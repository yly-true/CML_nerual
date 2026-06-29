"""Visualize a learned dynamics model as a virtual simulator."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import torch

from cml_model import CML_HIDDEN_DIMS, CML_LATENT_DIM, NeuralCML
from tasks import CARTPOLE, PENDULUM, feature_dim, obs_to_features, resolve_env_id, resolve_task_name
from utils import get_dims, make_env, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="可视化 Pendulum-v1 Neural CML 虚拟仿真器")
    parser.add_argument("--checkpoint", type=str, default=None, help="默认自动使用 runs 里的最新 checkpoint。")
    parser.add_argument("--task", choices=("pendulum", "cartpole"), default="pendulum")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--action-mode", choices=("random", "sine", "zero"), default="sine")
    parser.add_argument("--action-std", type=float, default=0.8)
    parser.add_argument("--sine-amplitude", type=float, default=3)
    parser.add_argument("--sine-period", type=float, default=30.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output", type=str, default=None, help="Optional .gif or .mp4 output path.")
    parser.add_argument("--show", action="store_true", help="Open an interactive matplotlib window.")
    return parser.parse_args()


def resolve_checkpoint(task_name: str, checkpoint: str | None) -> str:
    """Use the provided checkpoint or the newest saved model for a task."""
    if checkpoint is not None:
        return checkpoint

    env_id = resolve_env_id(task_name)
    run_name = env_id.replace("/", "_").replace("-", "_")
    candidates = sorted(
        Path("runs").glob(f"{run_name}/*/model_*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found under runs/{run_name}. Train a model first.")
    return str(candidates[0])


def load_model(checkpoint: str, obs_dim: int, action_dim: int, device: torch.device):
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
    model = NeuralCML(
        obs_dim=obs_dim,
        action_dim=action_dim,
        latent_dim=model_args["latent_dim"],
        hidden_dims=model_args["hidden_dims"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


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


def reset_visual_env_down(task_name: str, env, seed: int) -> np.ndarray:
    """Reset visualization rollout with the rod pointing downward."""
    env.reset(seed=seed)
    sim = env.unwrapped
    if task_name == PENDULUM:
        sim.state = np.asarray([np.pi, 0.0], dtype=np.float32)
        return np.asarray(sim._get_obs(), dtype=np.float32)
    if task_name == CARTPOLE:
        sim.state = np.asarray([0.0, np.pi - 1e-6, 0.0, 0.0], dtype=np.float32)
        return sim.state.copy()
    raise ValueError(f"Unsupported task: {task_name}")


@torch.no_grad()
def predict_next_obs(
    model: NeuralCML,
    obs: np.ndarray,
    action: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    obs_t = torch.as_tensor(obs.astype(np.float32), dtype=torch.float32, device=device).view(1, -1)
    action_t = torch.as_tensor(action, dtype=torch.float32, device=device).view(1, -1)
    state = model.encode(obs_t)
    next_state = model.transition(state, action_t)
    return model.decode(next_state).detach().cpu().numpy()[0].astype(np.float32)


def rollout(args: argparse.Namespace, model: NeuralCML, device: torch.device):
    task_name = resolve_task_name(args.task)
    env = make_env(resolve_env_id(task_name))
    _, action_dim = get_dims(env)
    actions = generate_actions(args, env)

    real_obs = reset_visual_env_down(task_name, env, args.seed)
    real_traj = [real_obs]
    model_traj = [obs_to_features(task_name, real_obs)]
    for action in actions:
        current_features = obs_to_features(task_name, real_obs)
        predicted_next_features = predict_next_obs(model, current_features, action, device)
        real_obs, _, terminated, truncated, _ = env.step(action)
        real_traj.append(np.asarray(real_obs, dtype=np.float32))
        model_traj.append(predicted_next_features.copy())
        if terminated or truncated:
            break

    env.close()
    used_steps = len(real_traj) - 1
    return np.asarray(real_traj), np.asarray(model_traj), actions[:used_steps], action_dim


def theta(task_name: str, obs: np.ndarray) -> np.ndarray:
    if task_name == CARTPOLE:
        return np.arctan2(obs[..., 3], obs[..., 2])
    return np.arctan2(obs[..., 1], obs[..., 0])


def rod_xy(task_name: str, obs: np.ndarray) -> tuple[float, float]:
    angle = float(theta(task_name, obs))
    return float(np.sin(angle)), float(np.cos(angle))


def make_animation(real_traj: np.ndarray, model_traj: np.ndarray, actions: np.ndarray, args: argparse.Namespace) -> FuncAnimation:
    task_name = resolve_task_name(args.task)
    frames = len(real_traj)
    real_features = obs_to_features(task_name, real_traj)
    real_theta = np.unwrap(theta(task_name, real_features))
    model_theta = np.unwrap(theta(task_name, model_traj))
    if task_name == CARTPOLE:
        lower_title = "Cart position and action"
        lower_ylabel = "x"
        real_lower = real_features[:, 0]
        model_lower = model_traj[:, 0]
        real_lower_label = "real x"
        model_lower_label = "model x"
    else:
        lower_title = "Angular velocity and action"
        lower_ylabel = "thetadot"
        real_lower = real_features[:, 2]
        model_lower = model_traj[:, 2]
        real_lower_label = "real thetadot"
        model_lower_label = "model thetadot"
    t = np.arange(frames)

    fig = plt.figure(figsize=(11, 5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.5])
    ax_pendulum = fig.add_subplot(gs[:, 0])
    ax_angle = fig.add_subplot(gs[0, 1])
    ax_vel = fig.add_subplot(gs[1, 1])

    ax_pendulum.set_title(f"{task_name} one-step prediction: real vs model")
    ax_pendulum.set_xlim(-4.8, 4.8) if task_name == CARTPOLE else ax_pendulum.set_xlim(-1.25, 1.25)
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
    ax_vel.set_title(lower_title)
    ax_vel.plot(t, real_lower, color="#1f77b4", label=real_lower_label)
    ax_vel.plot(t, model_lower, color="#ff7f0e", linestyle="--", label=model_lower_label)
    ax_action = ax_vel.twinx()
    ax_action.plot(action_t, actions[:, 0], color="#2ca02c", alpha=0.45, label="action")
    vel_cursor = ax_vel.axvline(0, color="black", alpha=0.35)
    ax_vel.set_xlabel("step")
    ax_vel.set_ylabel(lower_ylabel)
    ax_action.set_ylabel("action")
    ax_vel.grid(True, alpha=0.25)

    handles_a, labels_a = ax_vel.get_legend_handles_labels()
    handles_b, labels_b = ax_action.get_legend_handles_labels()
    ax_vel.legend(handles_a + handles_b, labels_a + labels_b, loc="upper right")

    fig.tight_layout()

    def update(frame: int):
        rx, ry = rod_xy(task_name, real_features[frame])
        mx, my = rod_xy(task_name, model_traj[frame])
        real_base_x = float(real_features[frame, 0]) if task_name == CARTPOLE else 0.0
        model_base_x = float(model_traj[frame, 0]) if task_name == CARTPOLE else 0.0
        real_line.set_data([real_base_x, real_base_x + rx], [0, ry])
        model_line.set_data([model_base_x, model_base_x + mx], [0, my])
        real_bob.set_data([real_base_x + rx], [ry])
        model_bob.set_data([model_base_x + mx], [my])
        angle_cursor.set_xdata([frame, frame])
        vel_cursor.set_xdata([frame, frame])
        error = float(np.linalg.norm(real_features[frame] - model_traj[frame]))
        time_text.set_text(f"step={frame}\none_step_error={error:.3f}")
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
    args.task = resolve_task_name(args.task)
    checkpoint = resolve_checkpoint(args.task, args.checkpoint)
    print(f"Using checkpoint: {checkpoint}")
    device = resolve_device(args.device)
    env = make_env(resolve_env_id(args.task))
    raw_obs_dim, action_dim = get_dims(env)
    obs_dim = feature_dim(args.task, raw_obs_dim)
    env.close()

    model = load_model(checkpoint, obs_dim, action_dim, device)
    real_traj, model_traj, actions, _ = rollout(args, model, device)
    anim = make_animation(real_traj, model_traj, actions, args)

    if args.output is not None:
        save_animation(anim, args.output, args.fps)
        print(f"Saved visualization to {args.output}")

    if args.show or args.output is None:
        plt.show()


if __name__ == "__main__":
    main()
