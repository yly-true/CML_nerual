"""训练 InvertedPendulum 上的 Neural CML 模型。

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
    parser = argparse.ArgumentParser(description="训练 InvertedPendulum 上的 Neural CML")
    parser.add_argument("--env-id", type=str, default="InvertedPendulum-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-env-steps", type=int, default=50_000)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--updates", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--recon-weight", type=float, default=0.05)
    parser.add_argument("--action-norm-weight", type=float, default=1e-4)
    parser.add_argument("--residual-norm-weight", type=float, default=1e-3)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run-dir", type=str, default="runs")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """固定随机种子，尽量提高结果可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def collect_random_data(env, buffer: ReplayBuffer, steps: int, seed: int) -> None:
    """通过随机动作采集自监督训练数据。

    这里直接在函数内显式指定采样方式：
    - use_gaussian = False: 使用均匀采样
    - use_gaussian = True: 使用高斯采样
    """
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)

    use_gaussian = False
    gaussian_mean = 0.5 * (action_low + action_high)
    default_std = np.maximum((action_high - action_low) / 6.0, 1e-6)
    gaussian_std = default_std

    obs, _ = env.reset(seed=seed)
    for _ in trange(steps, desc="Collecting random transitions"):
        if use_gaussian:
            action = np.random.normal(loc=gaussian_mean, scale=gaussian_std).astype(np.float32)
            action = np.clip(action, action_low, action_high)
        else:
            action = np.random.uniform(action_low, action_high).astype(np.float32)
        next_obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.add(obs, action, next_obs, done)
        obs = next_obs
        if done:
            obs, _ = env.reset()


def build_model(args: argparse.Namespace, obs_dim: int, action_dim: int, device: torch.device) -> NeuralCML:
    """按参数构造模型并放到目标设备上。"""
    return NeuralCML(
        obs_dim=obs_dim,
        action_dim=action_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
    ).to(device)


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
            residual_norm_weight=args.residual_norm_weight,
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
                vars(args),
                obs_mean=obs_mean,
                obs_std=obs_std,
            )


def prepare_dataset(args: argparse.Namespace):
    """创建环境、采样数据并完成标准化。"""
    env = make_env(args.env_id)
    obs_dim, action_dim = get_dims(env)
    buffer = ReplayBuffer(obs_dim, action_dim, args.buffer_size)
    collect_random_data(env, buffer, args.total_env_steps, args.seed)

    use_normalize = False
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

    dataset = prepare_dataset(args)
    model = build_model(args, dataset.obs_dim, dataset.action_dim, device)
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
