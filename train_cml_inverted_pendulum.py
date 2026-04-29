"""Train a neural CML model on MuJoCo InvertedPendulum.

This script has two phases:
1. random motor-babbling data collection
2. self-supervised latent prediction learning

It does not use rewards for training the CML model.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random

import numpy as np
import torch
from torch import optim
from tqdm import trange

from cml_model import NeuralCML
from replay_buffer import ReplayBuffer
from utils import make_env, get_dims, save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="InvertedPendulum-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-env-steps", type=int, default=50_000)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--updates", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--recon-weight", type=float, default=0.05)
    parser.add_argument("--action-norm-weight", type=float, default=1e-4)
    parser.add_argument("--residual-norm-weight", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run-dir", type=str, default="runs/cml_inverted_pendulum")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def collect_random_data(env, buffer: ReplayBuffer, steps: int, seed: int) -> None:
    obs, _ = env.reset(seed=seed)
    for _ in trange(steps, desc="Collecting random transitions"):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        buffer.add(obs, action, next_obs, done)
        obs = next_obs
        if done:
            obs, _ = env.reset()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    env = make_env(args.env_id)
    obs_dim, action_dim = get_dims(env)

    buffer = ReplayBuffer(obs_dim, action_dim, args.buffer_size)
    collect_random_data(env, buffer, args.total_env_steps, args.seed)

    obs_mean = buffer.obs[: buffer.size].mean(axis=0).astype(np.float32)
    obs_std = buffer.obs[: buffer.size].std(axis=0).astype(np.float32) + 1e-6

    # Normalize the replay buffer in-place for stable training.
    buffer.obs[: buffer.size] = (buffer.obs[: buffer.size] - obs_mean) / obs_std
    buffer.next_obs[: buffer.size] = (buffer.next_obs[: buffer.size] - obs_mean) / obs_std

    model = NeuralCML(
        obs_dim=obs_dim,
        action_dim=action_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    pbar = trange(args.updates, desc="Training CML")
    for step in pbar:
        batch = buffer.sample(args.batch_size, device)
        out = model.loss(
            batch.obs,
            batch.actions,
            batch.next_obs,
            recon_weight=args.recon_weight,
            action_norm_weight=args.action_norm_weight,
            residual_norm_weight=args.residual_norm_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        out.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        if step % 100 == 0:
            pbar.set_postfix(
                total=f"{out.total.item():.4f}",
                pred=f"{out.pred.item():.4f}",
                recon=f"{out.recon.item():.4f}",
            )

    run_dir = Path(args.run_dir)
    save_checkpoint(
        run_dir / "model.pt",
        model,
        optimizer,
        vars(args),
        obs_mean=obs_mean,
        obs_std=obs_std,
    )
    print(f"Saved checkpoint to {run_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
