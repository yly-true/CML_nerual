"""训练 Pendulum-v1 上的 Neural CML 模型。

整体流程分两步：
1. 随机探索采集转移数据
2. 用自监督方式学习 latent 动力学
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import random
import re
from types import SimpleNamespace

import numpy as np
import torch
from torch import optim
from tqdm import trange

from cml.cml_model import (
    CML_DERIVATIVE_DT,
    CML_DYNAMICS_MODE,
    CML_NETWORK_TYPE,
    CML_SNN_SURROGATE_SCALE,
    CML_SNN_TAU,
    CML_SNN_THRESHOLD,
    CML_SNN_TIMESTEPS,
    NeuralCML,
)
from cml.replay_buffer import ReplayBuffer
from cml.tasks import feature_dim, obs_to_features_from_env, reset_train_env, resolve_env_id, resolve_task_name
from cml.utils import (
    get_dims,
    make_env,
    resolve_device,
    save_checkpoint,
)


DEFAULT_CARTPOLE_RECON_WEIGHTS = [5.0, 1.0, 5.0, 5.0, 1.0]
DEFAULT_BIPEDALWALKER_RECON_WEIGHTS = [15.0, 15.0, 15.0, 15.0, 15.0] + [15.0] * 8
DEFAULT_MECANUM_RECON_WEIGHTS = [10.0, 10.0, 10.0] + [2.0] * 4


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="训练连续控制任务上的 Neural CML")
    parser.add_argument("--task", choices=("pendulum", "cartpole", "bipedalwalker", "mecanum"), default="cartpole")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-env-steps", type=int, default=300_000)
    parser.add_argument("--buffer-size", type=int, default=300_000)
    parser.add_argument("--updates", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--pred-weight", type=float, default=1.0)  #0.01
    parser.add_argument("--recon-loss-weight", "--recon-weight", dest="recon_weight", type=float, default=1.0)
    parser.add_argument("--recon-dim-weights", "--recon-weights", dest="recon_weights", type=float, nargs="*", default=None)
    parser.add_argument("--continuity-weight", type=float, default=0.2)
    parser.add_argument("--continuity-dt", type=float, default=0.05)
    parser.add_argument("--action-norm-weight", type=float, default=1e-4)
    parser.add_argument("--latent-norm-weight", type=float, default=1e-4)
    parser.add_argument("--network-type", choices=("snn", "mlp"), default=CML_NETWORK_TYPE)
    parser.add_argument("--dynamics-mode", choices=("auto", "latent", "obs_derivative"), default="auto")
    parser.add_argument("--derivative-dt", type=float, default=CML_DERIVATIVE_DT)
    parser.add_argument("--snn-timesteps", type=int, default=CML_SNN_TIMESTEPS)
    parser.add_argument("--snn-tau", type=float, default=CML_SNN_TAU)
    parser.add_argument("--snn-threshold", type=float, default=CML_SNN_THRESHOLD)
    parser.add_argument("--snn-surrogate-scale", type=float, default=CML_SNN_SURROGATE_SCALE)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Use CUDA mixed precision training.")
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--run-dir", type=str, default="runs")
    parser.add_argument("--load-run", type=str, default=None, help="checkpoint .pt path or run directory to continue from")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """固定随机种子，尽量提高结果可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def collect_random_data(
    env,
    buffer: ReplayBuffer,
    steps: int,
    seed: int,
    args: argparse.Namespace,
) -> None:
    """通过混合随机动作采集自监督训练数据。"""
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)

    gaussian_mean = np.zeros_like(action_low, dtype=np.float32)
    gaussian_std = np.maximum((action_high - action_low) / 8.0, 1e-6)
    gaussian_prob = 0.5
    mecanum_action_primitives = np.asarray(
        [
            [0.5, 0.5, 0.5, 0.5],
            [-0.5, -0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5, 0.5],
            [-0.5, 0.5, 0.5, -0.5],
            [0.5, -0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5, 0.5],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    segment_action = np.zeros_like(action_low, dtype=np.float32)
    segment_remaining = 0

    obs = reset_train_env(args.task, env, args, seed=seed)
    for _ in trange(steps, desc="Collecting random transitions"):
        if args.task == "mecanum":
            if segment_remaining <= 0:
                segment_remaining = int(np.random.randint(40, 161))
                if np.random.rand() < 0.75:
                    primitive = mecanum_action_primitives[np.random.randint(len(mecanum_action_primitives))]
                    segment_action = primitive[: action_low.size].copy()
                else:
                    segment_action = np.random.uniform(action_low, action_high).astype(np.float32)
            action = np.clip(segment_action, action_low, action_high).astype(np.float32)
            segment_remaining -= 1
        elif np.random.rand() < gaussian_prob:
            action = np.random.normal(loc=gaussian_mean, scale=gaussian_std).astype(np.float32)
            action = np.clip(action, action_low, action_high)
        else:
            action = np.random.uniform(action_low, action_high).astype(np.float32)
        obs_features = obs_to_features_from_env(args.task, obs, env)
        next_obs, _, terminated, truncated, _ = env.step(action)
        next_obs_features = obs_to_features_from_env(args.task, next_obs, env)
        done = terminated or truncated
        buffer.add(
            obs_features,
            action,
            next_obs_features,
            done,
        )
        obs = next_obs
        if done:
            obs = reset_train_env(args.task, env, args)


def build_model(args: argparse.Namespace, obs_dim: int, action_dim: int, device: torch.device) -> NeuralCML:
    """按参数构造模型并放到目标设备上。"""
    dynamics_mode = args.dynamics_mode
    if dynamics_mode == "auto":
        dynamics_mode = "obs_derivative" if args.task == "mecanum" else CML_DYNAMICS_MODE
    derivative_dim = obs_dim
    return NeuralCML(
        obs_dim=obs_dim,
        action_dim=action_dim,
        network_type=args.network_type,
        dynamics_mode=dynamics_mode,
        derivative_dt=args.derivative_dt,
        derivative_dim=derivative_dim,
        snn_timesteps=args.snn_timesteps,
        snn_tau=args.snn_tau,
        snn_threshold=args.snn_threshold,
        snn_surrogate_scale=args.snn_surrogate_scale,
    ).to(device)


def model_config(model: NeuralCML) -> dict[str, object]:
    """Return the model structure saved with each checkpoint."""
    return {
        "latent_dim": model.latent_dim,
        "hidden_dims": list(model.hidden_dims),
        "network_type": model.network_type,
        "dynamics_mode": model.dynamics_mode,
        "derivative_dt": model.derivative_dt,
        "derivative_dim": model.derivative_dim,
        "snn_timesteps": model.snn_timesteps,
        "snn_tau": model.snn_tau,
        "snn_threshold": model.snn_threshold,
        "snn_surrogate_scale": model.snn_surrogate_scale,
    }


def print_model_size(model: NeuralCML, obs_dim: int, action_dim: int) -> None:
    """打印当前网络参数，确认 cml_model.py 中的配置已生效。"""
    total_params = sum(param.numel() for param in model.parameters())
    param_mb = sum(param.numel() * param.element_size() for param in model.parameters()) / 1024 / 1024
    print(
        "Model size: "
        f"obs_dim={obs_dim}, action_dim={action_dim}, "
        f"latent_dim={model.latent_dim}, hidden_dims={list(model.hidden_dims)}, "
        f"network_type={model.network_type}, dynamics_mode={model.dynamics_mode}, "
        f"derivative_dt={model.derivative_dt}, derivative_dim={model.derivative_dim}, "
        f"snn_timesteps={model.snn_timesteps}, "
        f"snn_tau={model.snn_tau}, snn_threshold={model.snn_threshold}, "
        f"params={total_params:,}, param_memory={param_mb:.3f} MB"
    )


def resolve_load_checkpoint(load_run: str | None) -> Path | None:
    """Resolve --load-run as either a checkpoint file or a run directory."""
    if load_run is None:
        return None

    path = Path(load_run)
    if path.is_file():
        return path
    if path.is_dir():
        checkpoints = sorted(path.rglob("model_*.pt"), key=lambda item: item.stat().st_mtime, reverse=True)
        if not checkpoints:
            raise FileNotFoundError(f"No model_*.pt checkpoint found in {path}")
        return checkpoints[0]
    raise FileNotFoundError(f"--load-run does not exist: {path}")


def checkpoint_update_index(checkpoint_path: Path | None) -> int:
    """Infer the checkpoint step from names like model_50000.pt."""
    if checkpoint_path is None:
        return 0
    match = re.search(r"model_(\d+)\.pt$", checkpoint_path.name)
    if match is None:
        return 0
    return int(match.group(1))


def apply_checkpoint_model_args(args: argparse.Namespace, checkpoint_path: Path | None, device: torch.device) -> None:
    """Use saved architecture args when continuing from a checkpoint."""
    if checkpoint_path is None:
        return
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    args.network_type = saved_args.get("network_type", "mlp")
    args.dynamics_mode = saved_args.get("dynamics_mode", "latent")
    args.derivative_dt = float(saved_args.get("derivative_dt", CML_DERIVATIVE_DT))
    args.snn_timesteps = int(saved_args.get("snn_timesteps", CML_SNN_TIMESTEPS))
    args.snn_tau = float(saved_args.get("snn_tau", CML_SNN_TAU))
    args.snn_threshold = float(saved_args.get("snn_threshold", CML_SNN_THRESHOLD))
    args.snn_surrogate_scale = float(saved_args.get("snn_surrogate_scale", CML_SNN_SURROGATE_SCALE))


def load_model_checkpoint(
    checkpoint_path: Path | None,
    model: NeuralCML,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> int:
    """Restore model and optimizer state, returning the checkpoint update index."""
    if checkpoint_path is None:
        return 0
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer_state = ckpt.get("optimizer_state_dict")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    start_update = checkpoint_update_index(checkpoint_path)
    print(f"Loaded checkpoint: {checkpoint_path} (start_update={start_update})")
    return start_update


def train_model(
    model: NeuralCML,
    buffer: ReplayBuffer,
    optimizer: optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
    run_dir: Path,
    start_update: int = 0,
) -> None:
    """执行完整的自监督训练循环。"""
    recon_weights = build_recon_weights(args.recon_weights, model.obs_dim, device)
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    print(f"AMP mixed precision: {'enabled' if amp_enabled else 'disabled'}")
    pbar = trange(args.updates, desc="Training CML")  #带进度条的循环
    for local_step in pbar:
        update_idx = start_update + local_step + 1
        batch = buffer.sample(args.batch_size, device)  #从回放池里随机采样一个 batch
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            loss_out = model.loss(
                batch.obs,
                batch.actions,
                batch.next_obs,
                pred_weight=args.pred_weight,
                recon_weight=args.recon_weight,
                recon_weights=recon_weights,
                continuity_weight=args.continuity_weight,
                continuity_dt=args.continuity_dt,
                action_norm_weight=args.action_norm_weight,
                latent_norm_weight=args.latent_norm_weight,
            )
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss_out.total).backward()   #反向传播，计算梯度
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  #梯度裁剪，防止梯度爆炸
        scaler.step(optimizer)  #根据刚算出的梯度，真正修改参数
        scaler.update()

        if update_idx % 100 == 0:
            postfix = dict(
                total=f"{loss_out.total.item():.4f}",
                act_reg=f"{loss_out.action_norm.item():.3e}",
                state_reg=f"{loss_out.latent_norm.item():.3e}",
            )
            if getattr(model, "dynamics_mode", "latent") == "obs_derivative":
                postfix["step"] = f"{loss_out.pred.item():.3e}"
                postfix["s_dot"] = f"{loss_out.dot.item():.3e}"
            else:
                postfix["pred"] = f"{loss_out.pred.item():.3e}"
                postfix["recon"] = f"{loss_out.recon.item():.3e}"
            if loss_out.continuity.item() != 0.0:
                postfix["cont"] = f"{loss_out.continuity.item():.3e}"
            pbar.set_postfix(**postfix)

        if update_idx % args.save_every == 0 or local_step + 1 == args.updates:
            save_checkpoint(
                run_dir / f"model_{update_idx}.pt",
                model,
                optimizer,
                vars(args) | model_config(model),
            )


def prepare_dataset(args: argparse.Namespace):
    """创建环境并采样训练数据。"""
    args.task = resolve_task_name(args.task)
    args.env_id = resolve_env_id(args.task)
    env = make_env(args.env_id)
    raw_obs_dim, action_dim = get_dims(env)
    obs_dim = feature_dim(args.task, raw_obs_dim)
    if args.recon_weights is None and args.task == "cartpole":
        args.recon_weights = DEFAULT_CARTPOLE_RECON_WEIGHTS.copy()
    if args.recon_weights is None and args.task == "bipedalwalker":
        args.recon_weights = DEFAULT_BIPEDALWALKER_RECON_WEIGHTS.copy()
    if args.recon_weights is None and args.task == "mecanum":
        args.recon_weights = DEFAULT_MECANUM_RECON_WEIGHTS.copy()
    buffer = ReplayBuffer(obs_dim, action_dim, args.buffer_size)
    collect_random_data(env, buffer, args.total_env_steps, args.seed, args)

    print("Observation normalization: disabled")

    return SimpleNamespace(
        env=env,
        obs_dim=obs_dim,
        action_dim=action_dim,
        buffer=buffer,
    )


def build_run_dir(base_run_dir: str, env_id: str) -> Path:
    """按 任务名/时间戳 组织本次训练的输出目录。"""
    task_name = env_id.replace("/", "_").replace("-", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(base_run_dir) / task_name / timestamp


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    checkpoint_path = resolve_load_checkpoint(args.load_run)
    apply_checkpoint_model_args(args, checkpoint_path, device)

    dataset = prepare_dataset(args)  #准备buffer并且填充buffer
    model = build_model(args, dataset.obs_dim, dataset.action_dim, device)
    print_model_size(model, dataset.obs_dim, dataset.action_dim)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    start_update = load_model_checkpoint(checkpoint_path, model, optimizer, device)
    run_dir = checkpoint_path.parent if checkpoint_path is not None else build_run_dir(args.run_dir, args.env_id)
    train_model(
        model,
        dataset.buffer,
        optimizer,
        args,
        device,
        run_dir,
        start_update=start_update,
    )
    print(f"Saved checkpoints to {run_dir}")
    dataset.env.close()


def build_recon_weights(weights: list[float] | None, obs_dim: int, device: torch.device) -> torch.Tensor | None:
    """Build optional per-observation reconstruction weights."""
    if weights is None:
        return None
    if len(weights) != obs_dim:
        raise ValueError(f"--recon-dim-weights must contain {obs_dim} values for this task.")
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


if __name__ == "__main__":
    main()
