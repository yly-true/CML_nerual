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
from types import SimpleNamespace

import numpy as np
import torch
from torch import optim
from tqdm import trange

from cml_model import NeuralCML
from replay_buffer import ReplayBuffer
from tasks import ENV_ID, feature_dim, obs_to_features, reset_train_env
from utils import (
    compute_obs_stats,
    get_dims,
    make_env,
    normalize_replay_buffer,
    resolve_device,
    save_checkpoint,
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="训练 Pendulum-v1 上的 Neural CML")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-env-steps", type=int, default=300_000)
    parser.add_argument("--buffer-size", type=int, default=300_000)
    parser.add_argument("--random-pendulum-theta-range", type=float, default=np.pi)
    parser.add_argument("--random-pendulum-ang-vel-range", type=float, default=8.0)
    parser.add_argument("--updates", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--recon-weight", type=float, default=0.1)
    parser.add_argument("--action-norm-weight", type=float, default=1e-4)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run-dir", type=str, default="runs")
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

    obs = reset_train_env(env, args, seed=seed)
    for _ in trange(steps, desc="Collecting random transitions"):
        if np.random.rand() < gaussian_prob:
            action = np.random.normal(loc=gaussian_mean, scale=gaussian_std).astype(np.float32)
            action = np.clip(action, action_low, action_high)
        else:
            action = np.random.uniform(action_low, action_high).astype(np.float32)
        next_obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.add(
            obs_to_features(obs),
            action,
            obs_to_features(next_obs),
            done,
        )
        obs = next_obs
        if done:
            obs = reset_train_env(env, args)


def build_model(args: argparse.Namespace, obs_dim: int, action_dim: int, device: torch.device) -> NeuralCML:
    """按参数构造模型并放到目标设备上。"""
    _ = args
    return NeuralCML(
        obs_dim=obs_dim,
        action_dim=action_dim,
    ).to(device)


def model_config(model: NeuralCML) -> dict[str, object]:
    """Return the model structure saved with each checkpoint."""
    return {
        "latent_dim": model.latent_dim,
        "hidden_dims": list(model.hidden_dims),
    }


def print_model_size(model: NeuralCML, obs_dim: int, action_dim: int) -> None:
    """打印当前网络参数，确认 cml_model.py 中的配置已生效。"""
    total_params = sum(param.numel() for param in model.parameters())
    param_mb = sum(param.numel() * param.element_size() for param in model.parameters()) / 1024 / 1024
    print(
        "Model size: "
        f"obs_dim={obs_dim}, action_dim={action_dim}, "
        f"latent_dim={model.latent_dim}, hidden_dims={list(model.hidden_dims)}, "
        f"params={total_params:,}, param_memory={param_mb:.3f} MB"
    )


def train_model(
    model: NeuralCML,
    buffer: ReplayBuffer,
    optimizer: optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
    run_dir: Path,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
) -> None:
    """执行完整的自监督训练循环。"""
    pbar = trange(args.updates, desc="Training CML")  #带进度条的循环
    for step in pbar:
        update_idx = step + 1
        batch = buffer.sample(args.batch_size, device)  #从回放池里随机采样一个 batch
        loss_out = model.loss(
            batch.obs,
            batch.actions,
            batch.next_obs,
            recon_weight=args.recon_weight,
            action_norm_weight=args.action_norm_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss_out.total.backward()   #反向传播，计算梯度
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  #梯度裁剪，防止梯度爆炸
        optimizer.step()  #根据刚算出的梯度，真正修改参数

        if update_idx % 100 == 0:
            pbar.set_postfix(
                total=f"{loss_out.total.item():.4f}",
                pred=f"{loss_out.pred.item():.4f}",
                recon=f"{loss_out.recon.item():.4f}",
            )

        if update_idx % args.save_every == 0 or update_idx == args.updates:
            save_checkpoint(
                run_dir / f"model_{update_idx}.pt",
                model,
                optimizer,
                vars(args) | model_config(model),
                obs_mean=obs_mean,
                obs_std=obs_std,
            )


def prepare_dataset(args: argparse.Namespace):
    """创建环境、采样数据并完成标准化。"""
    args.env_id = ENV_ID
    env = make_env(args.env_id)
    raw_obs_dim, action_dim = get_dims(env)
    obs_dim = feature_dim(raw_obs_dim)
    buffer = ReplayBuffer(obs_dim, action_dim, args.buffer_size)
    collect_random_data(env, buffer, args.total_env_steps, args.seed, args)

    use_normalize = True
    if use_normalize:
        obs_mean_t, obs_std_t = compute_obs_stats(buffer.obs[: buffer.size])
        print(f"Observation mean: {obs_mean_t}")
        print(f"Observation std: {obs_std_t}")
        normalize_replay_buffer(buffer, obs_mean_t, obs_std_t)
    else:
        obs_mean_t = torch.zeros(obs_dim, dtype=torch.float32)
        obs_std_t = torch.ones(obs_dim, dtype=torch.float32)

    return SimpleNamespace(
        env=env,
        obs_dim=obs_dim,
        action_dim=action_dim,
        buffer=buffer,
        obs_mean=obs_mean_t.cpu().numpy(),
        obs_std=obs_std_t.cpu().numpy(),
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

    dataset = prepare_dataset(args)  #准备buffer并且填充buffer
    model = build_model(args, dataset.obs_dim, dataset.action_dim, device)
    print_model_size(model, dataset.obs_dim, dataset.action_dim)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    run_dir = build_run_dir(args.run_dir, args.env_id)
    train_model(
        model,
        dataset.buffer,
        optimizer,
        args,
        device,
        run_dir,
        dataset.obs_mean,
        dataset.obs_std,
    )
    print(f"Saved checkpoints to {run_dir}")
    dataset.env.close()


if __name__ == "__main__":
    main()
