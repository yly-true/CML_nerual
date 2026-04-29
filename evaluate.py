"""评估训练好的 Neural CML 在 InvertedPendulum 上的表现。"""

from __future__ import annotations

import argparse
import time
from types import SimpleNamespace

import numpy as np
import torch

from cml_model import NeuralCML, plan_action_random_shooting
from utils import default_goal_obs, get_dims, make_env, normalize_obs, resolve_device


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="评估 InvertedPendulum 上的 Neural CML")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--env-id", type=str, default="InvertedPendulum-v4")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--num-sequences", type=int, default=8192)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--action-cost", type=float, default=0.003)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def load_model_and_stats(args: argparse.Namespace, obs_dim: int, action_dim: int, device: torch.device):
    """从 checkpoint 恢复模型和观测标准化参数。"""
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
    return model, ckpt["obs_mean"], ckpt["obs_std"]


def prepare_planner_inputs(env, obs_dim: int, obs_mean, obs_std, device: torch.device):
    """构造规划阶段反复使用的常量张量。"""
    action_low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=device)
    action_high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=device)
    raw_goal = default_goal_obs(env, obs_dim)
    goal = normalize_obs(raw_goal, obs_mean, obs_std)
    goal_t = torch.as_tensor(goal, dtype=torch.float32, device=device)
    return SimpleNamespace(action_low=action_low, action_high=action_high, goal_t=goal_t)


def run_episode(
    env,
    model: NeuralCML,
    planner_inputs,
    obs_mean,
    obs_std,
    args: argparse.Namespace,
    device: torch.device,
    use_normalize: bool,
) -> tuple[float, int]:
    """执行一条评估 episode。"""
    obs, _ = env.reset()
    ep_return = 0.0
    ep_len = 0

    for _ in range(args.max_steps):
        norm_obs = obs.astype(np.float32)
        if use_normalize:
            norm_obs = normalize_obs(norm_obs, obs_mean, obs_std)
        obs_t = torch.as_tensor(norm_obs, dtype=torch.float32, device=device)

        action_t = plan_action_random_shooting(
            model=model,
            obs=obs_t,
            goal_obs=planner_inputs.goal_t,
            action_low=planner_inputs.action_low,
            action_high=planner_inputs.action_high,
            num_sequences=args.num_sequences,
            horizon=args.horizon,
            action_cost=args.action_cost,
        )
        action = action_t.detach().cpu().numpy()

        obs, reward, terminated, truncated, _ = env.step(action)
        ep_return += float(reward)
        ep_len += 1

        if args.render:
            time.sleep(0.01)

        if terminated or truncated:
            break

    return ep_return, ep_len


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    use_normalize = False
    render_mode = "human" if args.render else None
    env = make_env(args.env_id, render_mode=render_mode)
    obs_dim, action_dim = get_dims(env)
    model, obs_mean, obs_std = load_model_and_stats(args, obs_dim, action_dim, device)
    if use_normalize:
        planner_inputs = prepare_planner_inputs(env, obs_dim, obs_mean, obs_std, device)
    else:
        planner_inputs = prepare_planner_inputs(
            env,
            obs_dim,
            np.zeros(obs_dim, dtype=np.float32),
            np.ones(obs_dim, dtype=np.float32),
            device,
        )

    returns = []
    lengths = []

    for ep in range(args.episodes):
        ep_return, ep_len = run_episode(
            env=env,
            model=model,
            planner_inputs=planner_inputs,
            obs_mean=obs_mean,
            obs_std=obs_std,
            args=args,
            device=device,
            use_normalize=use_normalize,
        )

        returns.append(ep_return)
        lengths.append(ep_len)
        print(f"episode={ep} return={ep_return:.1f} length={ep_len}")

    print(f"mean_return={np.mean(returns):.1f} +/- {np.std(returns):.1f}")
    print(f"mean_length={np.mean(lengths):.1f} +/- {np.std(lengths):.1f}")
    env.close()


if __name__ == "__main__":
    main()
