"""Evaluate the trained neural CML on MuJoCo InvertedPendulum."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from cml_model import NeuralCML, plan_action_random_shooting
from utils import make_env, get_dims, default_goal_obs, normalize_obs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--env-id", type=str, default="InvertedPendulum-v4")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--num-sequences", type=int, default=2048)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--action-cost", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    render_mode = "human" if args.render else None
    env = make_env(args.env_id, render_mode=render_mode)
    obs_dim, action_dim = get_dims(env)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_args = ckpt["args"]
    model = NeuralCML(
        obs_dim=obs_dim,
        action_dim=action_dim,
        latent_dim=model_args["latent_dim"],
        hidden_dim=model_args["hidden_dim"],
        depth=model_args["depth"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    obs_mean = ckpt["obs_mean"]
    obs_std = ckpt["obs_std"]

    action_low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=device)
    action_high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=device)

    raw_goal = default_goal_obs(env, obs_dim)
    goal = normalize_obs(raw_goal, obs_mean, obs_std)
    goal_t = torch.as_tensor(goal, dtype=torch.float32, device=device)

    returns = []
    lengths = []

    for ep in range(args.episodes):
        obs, _ = env.reset()
        ep_return = 0.0
        ep_len = 0

        for t in range(args.max_steps):
            norm_obs = normalize_obs(obs.astype(np.float32), obs_mean, obs_std)
            obs_t = torch.as_tensor(norm_obs, dtype=torch.float32, device=device)

            action_t = plan_action_random_shooting(
                model=model,
                obs=obs_t,
                goal_obs=goal_t,
                action_low=action_low,
                action_high=action_high,
                num_sequences=args.num_sequences,
                horizon=args.horizon,
                action_cost=args.action_cost,
            )
            action = action_t.detach().cpu().numpy()

            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            ep_len += 1

            if args.render:
                time.sleep(0.01)

            if terminated or truncated:
                break

        returns.append(ep_return)
        lengths.append(ep_len)
        print(f"episode={ep} return={ep_return:.1f} length={ep_len}")

    print(f"mean_return={np.mean(returns):.1f} +/- {np.std(returns):.1f}")
    print(f"mean_length={np.mean(lengths):.1f} +/- {np.std(lengths):.1f}")


if __name__ == "__main__":
    main()
