"""Visualize a learned Pendulum-v1 dynamics model as a virtual simulator."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import torch

from cml_model import CML_HIDDEN_DIMS, CML_LATENT_DIM, NeuralCML
from tasks import ENV_ID
from utils import get_dims, make_env, normalize_obs, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="可视化 Pendulum-v1 Neural CML 虚拟仿真器")
    parser.add_argument("--checkpoint", type=str, default=None, help="默认自动使用 runs 里的最新 checkpoint。")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--action-mode", choices=("random", "sine", "zero"), default="sine")
    parser.add_argument("--action-std", type=float, default=0.8)
    parser.add_argument("--sine-amplitude", type=float, default=1.5)
    parser.add_argument("--sine-period", type=float, default=50.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output", type=str, default=None, help="Optional .gif or .mp4 output path.")
    parser.add_argument("--show", action="store_true", help="Open an interactive matplotlib window.")
    return parser.parse_args()


def resolve_checkpoint(checkpoint: str | None) -> str:
    """Use the provided checkpoint or the newest saved Pendulum model."""
    if checkpoint is not None:
        return checkpoint

    candidates = sorted(
        Path("runs").glob("Pendulum_v1/*/model_*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No checkpoint found under runs/Pendulum_v1. Train a model first.")
    return str(candidates[0])


def load_model(checkpoint: str, action_dim: int, device: torch.device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    if "hidden_dims" in saved_args:
        hidden_dims = [int(dim) for dim in saved_args["hidden_dims"]]
    elif "hidden_dim" in saved_args and "depth" in saved_args:
        hidden_dims = [int(saved_args["hidden_dim"])] * int(saved_args["depth"])
    else:
        hidden_dims = [int(dim) for dim in CML_HIDDEN_DIMS]
    model_args = {
        "latent_dim": int(saved_args.get("latent_dim", CML_LATENT_DIM)),
        "hidden_dims": hidden_dims,
    }
    obs_dim = int(np.asarray(ckpt["obs_mean"]).shape[0])
    model = NeuralCML(
        obs_dim=obs_dim,
        action_dim=action_dim,
        latent_dim=model_args["latent_dim"],
        hidden_dims=model_args["hidden_dims"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, np.asarray(ckpt["obs_mean"], dtype=np.float32), np.asarray(ckpt["obs_std"], dtype=np.float32)


def denormalize_obs(obs: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return obs * (std + 1e-6) + mean


def generate_actions(args: argparse.Namespace, env) -> np.ndarray:
    rng = np.random.default_rng(args.seed)
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    actions = np.zeros((args.steps, action_low.size), dtype=np.float32)

    if args.action_mode == "zero":
        return actions

    if args.action_mode == "sine":
        t = np.arange(args.steps, dtype=np.float32)
        wave = args.sine_amplitude * np.sin(2.0 * np.pi * t / args.sine_period)
        actions[:, 0] = wave
        return np.clip(actions, action_low, action_high)

    actions = rng.normal(loc=0.0, scale=args.action_std, size=actions.shape).astype(np.float32)
    return np.clip(actions, action_low, action_high)


@torch.no_grad()
def predict_next_obs(
    model: NeuralCML,
    obs: np.ndarray,
    action: np.ndarray,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    norm_obs = normalize_obs(obs.astype(np.float32), obs_mean, obs_std)
    obs_t = torch.as_tensor(norm_obs, dtype=torch.float32, device=device).view(1, -1)
    action_t = torch.as_tensor(action, dtype=torch.float32, device=device).view(1, -1)
    state = model.encode(obs_t)
    next_state = model.transition(state, action_t)
    next_norm_obs = model.decode(next_state).detach().cpu().numpy()[0]
    return denormalize_obs(next_norm_obs, obs_mean, obs_std).astype(np.float32)


def rollout(args: argparse.Namespace, model: NeuralCML, obs_mean: np.ndarray, obs_std: np.ndarray, device: torch.device):
    env = make_env(ENV_ID)
    _, action_dim = get_dims(env)
    actions = generate_actions(args, env)

    real_obs, _ = env.reset(seed=args.seed)
    model_obs = np.asarray(real_obs, dtype=np.float32).copy()

    real_traj = [np.asarray(real_obs, dtype=np.float32)]
    model_traj = [model_obs.copy()]
    for action in actions:
        real_obs, _, terminated, truncated, _ = env.step(action)
        model_obs = predict_next_obs(model, model_obs, action, obs_mean, obs_std, device)
        real_traj.append(np.asarray(real_obs, dtype=np.float32))
        model_traj.append(model_obs.copy())
        if terminated or truncated:
            break

    env.close()
    used_steps = len(real_traj) - 1
    return np.asarray(real_traj), np.asarray(model_traj), actions[:used_steps], action_dim


def theta(obs: np.ndarray) -> np.ndarray:
    return np.arctan2(obs[..., 1], obs[..., 0])


def rod_xy(obs: np.ndarray) -> tuple[float, float]:
    angle = float(theta(obs))
    return float(np.sin(angle)), float(np.cos(angle))


def make_animation(real_traj: np.ndarray, model_traj: np.ndarray, actions: np.ndarray, args: argparse.Namespace) -> FuncAnimation:
    frames = len(real_traj)
    real_theta = theta(real_traj)
    model_theta = theta(model_traj)
    real_thetadot = real_traj[:, 2]
    model_thetadot = model_traj[:, 2]
    t = np.arange(frames)

    fig = plt.figure(figsize=(11, 5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.5])
    ax_pendulum = fig.add_subplot(gs[:, 0])
    ax_angle = fig.add_subplot(gs[0, 1])
    ax_vel = fig.add_subplot(gs[1, 1])

    ax_pendulum.set_title("Pendulum rollout: real vs model")
    ax_pendulum.set_xlim(-1.25, 1.25)
    ax_pendulum.set_ylim(-1.25, 1.25)
    ax_pendulum.set_aspect("equal")
    ax_pendulum.grid(True, alpha=0.25)
    ax_pendulum.plot([0], [0], marker="o", color="black")
    real_line, = ax_pendulum.plot([], [], color="#1f77b4", lw=4, label="real env")
    model_line, = ax_pendulum.plot([], [], color="#ff7f0e", lw=3, linestyle="--", label="model sim")
    real_bob, = ax_pendulum.plot([], [], marker="o", color="#1f77b4", ms=12)
    model_bob, = ax_pendulum.plot([], [], marker="o", color="#ff7f0e", ms=9)
    time_text = ax_pendulum.text(0.02, 0.96, "", transform=ax_pendulum.transAxes, va="top")
    ax_pendulum.legend(loc="lower right")

    ax_angle.set_title("Angle")
    ax_angle.plot(t, real_theta, color="#1f77b4", label="real")
    ax_angle.plot(t, model_theta, color="#ff7f0e", linestyle="--", label="model")
    angle_cursor = ax_angle.axvline(0, color="black", alpha=0.35)
    ax_angle.set_ylabel("theta")
    ax_angle.grid(True, alpha=0.25)
    ax_angle.legend(loc="upper right")

    action_t = np.arange(len(actions))
    ax_vel.set_title("Angular velocity and action")
    ax_vel.plot(t, real_thetadot, color="#1f77b4", label="real thetadot")
    ax_vel.plot(t, model_thetadot, color="#ff7f0e", linestyle="--", label="model thetadot")
    ax_action = ax_vel.twinx()
    ax_action.plot(action_t, actions[:, 0], color="#2ca02c", alpha=0.45, label="action")
    vel_cursor = ax_vel.axvline(0, color="black", alpha=0.35)
    ax_vel.set_xlabel("step")
    ax_vel.set_ylabel("thetadot")
    ax_action.set_ylabel("action")
    ax_vel.grid(True, alpha=0.25)

    handles_a, labels_a = ax_vel.get_legend_handles_labels()
    handles_b, labels_b = ax_action.get_legend_handles_labels()
    ax_vel.legend(handles_a + handles_b, labels_a + labels_b, loc="upper right")

    fig.tight_layout()

    def update(frame: int):
        rx, ry = rod_xy(real_traj[frame])
        mx, my = rod_xy(model_traj[frame])
        real_line.set_data([0, rx], [0, ry])
        model_line.set_data([0, mx], [0, my])
        real_bob.set_data([rx], [ry])
        model_bob.set_data([mx], [my])
        angle_cursor.set_xdata([frame, frame])
        vel_cursor.set_xdata([frame, frame])
        error = float(np.linalg.norm(real_traj[frame] - model_traj[frame]))
        time_text.set_text(f"step={frame}\nobs_error={error:.3f}")
        return real_line, model_line, real_bob, model_bob, angle_cursor, vel_cursor, time_text

    return FuncAnimation(fig, update, frames=frames, interval=1000 / args.fps, blit=True)


def save_animation(anim: FuncAnimation, output: str, fps: int) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".gif":
        anim.save(output_path, writer="pillow", fps=fps)
    elif suffix == ".mp4":
        anim.save(output_path, writer="ffmpeg", fps=fps)
    else:
        raise ValueError("--output must end with .gif or .mp4")


def main() -> None:
    args = parse_args()
    checkpoint = resolve_checkpoint(args.checkpoint)
    print(f"Using checkpoint: {checkpoint}")
    device = resolve_device(args.device)
    env = make_env(ENV_ID)
    _, action_dim = get_dims(env)
    env.close()

    model, obs_mean, obs_std = load_model(checkpoint, action_dim, device)
    real_traj, model_traj, actions, _ = rollout(args, model, obs_mean, obs_std, device)
    anim = make_animation(real_traj, model_traj, actions, args)

    if args.output is not None:
        save_animation(anim, args.output, args.fps)
        print(f"Saved visualization to {args.output}")

    if args.show or args.output is None:
        plt.show()


if __name__ == "__main__":
    main()
