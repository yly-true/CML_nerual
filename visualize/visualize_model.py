"""Visualize a learned dynamics model as a virtual simulator."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import torch

from cml.cml_model import (
    CML_DERIVATIVE_DT,
    CML_DYNAMICS_MODE,
    CML_HIDDEN_DIMS,
    CML_LATENT_DIM,
    CML_SNN_SURROGATE_SCALE,
    CML_SNN_TAU,
    CML_SNN_THRESHOLD,
    CML_SNN_TIMESTEPS,
    NeuralCML,
)
from cml.tasks import (
    CARTPOLE,
    MECANUM,
    PENDULUM,
    SUPPORTED_TASKS,
    feature_dim,
    feature_names,
    obs_to_features,
    obs_to_features_from_env,
    resolve_env_id,
    resolve_task_name,
)
from cml.utils import get_dims, make_env, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="可视化 Pendulum-v1 Neural CML 虚拟仿真器")
    parser.add_argument("--checkpoint", type=str, default=None, help="默认自动使用 runs 里的最新 checkpoint。")
    parser.add_argument("--task", choices=SUPPORTED_TASKS, default="pendulum")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--action-mode", choices=("constant", "random", "sine", "zero"), default="sine")
    parser.add_argument(
        "--constant-action",
        type=float,
        nargs="*",
        default=[0.5],
        help="Constant action value. Provide one value or one per action dimension.",
    )
    parser.add_argument(
        "--constant-torque",
        type=float,
        nargs="*",
        default=None,
        help="Mecanum-only constant motor torque. Provide one value or one per wheel; converted by ctrl_scale.",
    )
    parser.add_argument("--action-std", type=float, default=0.8)
    parser.add_argument("--sine-amplitude", type=float, default=1)
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
        "network_type": saved_args.get("network_type", "mlp"),
        "dynamics_mode": saved_args.get("dynamics_mode", CML_DYNAMICS_MODE),
        "derivative_dt": float(saved_args.get("derivative_dt", CML_DERIVATIVE_DT)),
        "derivative_dim": int(saved_args.get("derivative_dim", obs_dim)),
        "snn_timesteps": int(saved_args.get("snn_timesteps", CML_SNN_TIMESTEPS)),
        "snn_tau": float(saved_args.get("snn_tau", CML_SNN_TAU)),
        "snn_threshold": float(saved_args.get("snn_threshold", CML_SNN_THRESHOLD)),
        "snn_surrogate_scale": float(saved_args.get("snn_surrogate_scale", CML_SNN_SURROGATE_SCALE)),
    }
    model = NeuralCML(
        obs_dim=obs_dim,
        action_dim=action_dim,
        latent_dim=model_args["latent_dim"],
        hidden_dims=model_args["hidden_dims"],
        network_type=model_args["network_type"],
        dynamics_mode=model_args["dynamics_mode"],
        derivative_dt=model_args["derivative_dt"],
        derivative_dim=model_args["derivative_dim"],
        snn_timesteps=model_args["snn_timesteps"],
        snn_tau=model_args["snn_tau"],
        snn_threshold=model_args["snn_threshold"],
        snn_surrogate_scale=model_args["snn_surrogate_scale"],
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

    if args.action_mode == "constant":
        constant_values = args.constant_action
        if args.constant_torque is not None:
            ctrl_scale = float(getattr(env.unwrapped, "ctrl_scale", 1.0))
            constant_values = [float(value) / ctrl_scale for value in args.constant_torque]
        if len(constant_values) == 0:
            raise ValueError("--constant-action requires at least one value.")
        if len(constant_values) == 1:
            constant_action = np.full(action_low.shape, constant_values[0], dtype=np.float32)
        elif len(constant_values) == action_low.size:
            constant_action = np.asarray(constant_values, dtype=np.float32)
        else:
            raise ValueError(f"--constant-action must contain 1 or {action_low.size} values.")
        actions[:, :] = constant_action
        return np.clip(actions, action_low, action_high)

    if args.action_mode == "sine":
        t = np.arange(args.steps, dtype=np.float32)
        phase = np.linspace(0.0, 2.0 * np.pi, action_low.size, endpoint=False, dtype=np.float32)
        wave = args.sine_amplitude * np.sin(2.0 * np.pi * t[:, None] / args.sine_period + phase[None, :])
        actions[:, :] = wave
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
    if task_name == MECANUM:
        obs, _ = env.reset(seed=seed)
        return np.asarray(obs, dtype=np.float32)
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
    next_state = model.transition(state, action_t, obs_t)
    return model.decode(next_state).detach().cpu().numpy()[0].astype(np.float32)


@torch.no_grad()
def predict_obs_derivative(
    model: NeuralCML,
    obs: np.ndarray,
    action: np.ndarray,
    device: torch.device,
) -> np.ndarray | None:
    if getattr(model, "dynamics_mode", "latent") != "obs_derivative":
        return None
    obs_t = torch.as_tensor(obs.astype(np.float32), dtype=torch.float32, device=device).view(1, -1)
    action_t = torch.as_tensor(action, dtype=torch.float32, device=device).view(1, -1)
    return model.obs_derivative(obs_t, action_t).detach().cpu().numpy()[0].astype(np.float32)


def rollout(args: argparse.Namespace, model: NeuralCML, device: torch.device):
    task_name = resolve_task_name(args.task)
    render_mode = "human" if task_name == MECANUM and args.show else None
    env = make_env(resolve_env_id(task_name), render_mode=render_mode)
    _, action_dim = get_dims(env)
    actions = generate_actions(args, env)

    real_obs = reset_visual_env_down(task_name, env, args.seed)
    initial_features = obs_to_features_from_env(task_name, real_obs, env)
    use_feature_traj = task_name == MECANUM
    real_traj = [initial_features if use_feature_traj else real_obs]
    model_traj = [initial_features]
    step_errors = [0.0]
    s_dot_errors = [np.nan]
    for action in actions:
        current_features = obs_to_features_from_env(task_name, real_obs, env)
        predicted_next_features = predict_next_obs(model, current_features, action, device)
        predicted_s_dot = predict_obs_derivative(model, current_features, action, device)
        real_obs, _, terminated, truncated, _ = env.step(action)
        if task_name == MECANUM and args.show:
            env.render()
            time.sleep(1.0 / max(args.fps, 1))
        next_features = obs_to_features_from_env(task_name, real_obs, env)
        real_traj.append(next_features if use_feature_traj else np.asarray(real_obs, dtype=np.float32))
        model_traj.append(predicted_next_features.copy())
        step_errors.append(float(np.sqrt(np.mean((next_features - predicted_next_features) ** 2))))
        if predicted_s_dot is None:
            s_dot_errors.append(np.nan)
        else:
            derivative_dim = int(getattr(model, "derivative_dim", len(current_features)))
            derivative_dt = float(getattr(model, "derivative_dt", 1.0))
            target_s_dot = (next_features[:derivative_dim] - current_features[:derivative_dim]) / derivative_dt
            s_dot_errors.append(float(np.sqrt(np.mean((target_s_dot - predicted_s_dot[:derivative_dim]) ** 2))))
        if terminated or truncated:
            break

    keep_env = env if task_name == MECANUM and args.show else None
    if keep_env is None:
        env.close()
    used_steps = len(real_traj) - 1
    diagnostics = {
        "step_errors": np.asarray(step_errors, dtype=np.float32),
        "s_dot_errors": np.asarray(s_dot_errors, dtype=np.float32),
    }
    return np.asarray(real_traj), np.asarray(model_traj), actions[:used_steps], action_dim, keep_env, diagnostics


def theta(task_name: str, obs: np.ndarray) -> np.ndarray:
    if task_name == CARTPOLE:
        return np.arctan2(obs[..., 3], obs[..., 2])
    return np.arctan2(obs[..., 1], obs[..., 0])


def rod_xy(task_name: str, obs: np.ndarray) -> tuple[float, float]:
    angle = float(theta(task_name, obs))
    return float(np.sin(angle)), float(np.cos(angle))


def make_animation(
    real_traj: np.ndarray,
    model_traj: np.ndarray,
    actions: np.ndarray,
    args: argparse.Namespace,
    diagnostics: dict[str, np.ndarray] | None = None,
) -> FuncAnimation:
    task_name = resolve_task_name(args.task)
    if task_name == MECANUM:
        return make_feature_animation(real_traj, model_traj, actions, args, diagnostics)

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
        step_rmse = diagnostics["step_errors"][frame] if diagnostics is not None else error
        s_dot_rmse = diagnostics["s_dot_errors"][frame] if diagnostics is not None else np.nan
        text = f"step={frame}\none_step_error={error:.3f}\nstep_rmse={step_rmse:.3f}"
        if not np.isnan(s_dot_rmse):
            text += f"\ns_dot_rmse={s_dot_rmse:.3f}"
        time_text.set_text(text)
        return real_line, model_line, real_bob, model_bob, angle_cursor, vel_cursor, time_text

    return FuncAnimation(fig, update, frames=frames, interval=1000 / args.fps, blit=True)


def make_feature_animation(
    real_traj: np.ndarray,
    model_traj: np.ndarray,
    actions: np.ndarray,
    args: argparse.Namespace,
    diagnostics: dict[str, np.ndarray] | None = None,
) -> FuncAnimation:
    """Animate one-step prediction errors for high-dimensional raw observations."""
    task_name = resolve_task_name(args.task)
    real_features = np.asarray(real_traj, dtype=np.float32)
    model_features = np.asarray(model_traj, dtype=np.float32)
    frames = len(real_features)
    t = np.arange(frames)

    labels = feature_names(task_name)
    feature_count = min(len(labels), real_features.shape[-1], model_features.shape[-1])
    action_count = actions.shape[-1] if len(actions) > 0 else 0

    fig_height = max(8.0, 1.15 * feature_count + 2.2)
    fig, axes = plt.subplots(feature_count + 1, 1, figsize=(12, fig_height), sharex=True)
    feature_axes = list(axes[:-1])
    ax_actions = axes[-1]
    feature_axes[0].set_title(f"{task_name} one-step prediction: real vs model")
    ax_actions.set_title("Actions")
    ax_actions.set_xlabel("step")
    ax_actions.set_ylabel("action")
    ax_actions.grid(True, alpha=0.25)

    real_colors = ["#0055ff", "#009e73", "#cc79a7", "#d55e00", "#000000"]
    model_colors = ["#ffb000", "#7a3cff", "#56b4e9", "#e60049", "#777777"]
    feature_cursors = []
    for idx in range(feature_count):
        label = labels[idx] if idx < len(labels) else f"obs_{idx}"
        ax_feature = feature_axes[idx]
        real_color = real_colors[idx % len(real_colors)]
        model_color = model_colors[idx % len(model_colors)]
        ax_feature.plot(t, real_features[:, idx], color=real_color, linewidth=2.2, label="real")
        ax_feature.plot(
            t,
            model_features[:, idx],
            color=model_color,
            linestyle=(0, (6, 2)),
            linewidth=1.9,
            alpha=0.95,
            label="model",
        )
        ax_feature.set_ylabel(label, rotation=0, ha="right", va="center")
        ax_feature.grid(True, alpha=0.25)
        ax_feature.tick_params(axis="y", labelsize=8)
        if idx == 0:
            ax_feature.legend(loc="upper right", ncol=2, fontsize=8)
        feature_cursors.append(ax_feature.axvline(0, color="black", linewidth=1.6, alpha=0.55))

    action_t = np.arange(len(actions))
    action_colors = ["#0072b2", "#f0e442", "#009e73", "#d55e00"]
    for idx in range(action_count):
        ax_actions.plot(
            action_t,
            actions[:, idx],
            color=action_colors[idx % len(action_colors)],
            linewidth=2.2,
            alpha=0.95,
            label=f"action {idx}",
        )

    action_cursor = ax_actions.axvline(0, color="black", linewidth=2.0, alpha=0.55)
    error_text = feature_axes[0].text(0.01, 0.94, "", transform=feature_axes[0].transAxes, va="top")
    ax_actions.legend(loc="upper right", ncol=2, fontsize=8)
    fig.tight_layout()

    def update(frame: int):
        for cursor in feature_cursors:
            cursor.set_xdata([frame, frame])
        action_cursor.set_xdata([frame, frame])
        error = float(np.linalg.norm(real_features[frame] - model_features[frame]))
        step_rmse = diagnostics["step_errors"][frame] if diagnostics is not None else error
        s_dot_rmse = diagnostics["s_dot_errors"][frame] if diagnostics is not None else np.nan
        text = f"step={frame}\none_step_error={error:.3f}\nstep_rmse={step_rmse:.3f}"
        if not np.isnan(s_dot_rmse):
            text += f"\ns_dot_rmse={s_dot_rmse:.3f}"
        error_text.set_text(text)
        return (*feature_cursors, action_cursor, error_text)

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
    real_traj, model_traj, actions, _, keep_env, diagnostics = rollout(args, model, device)
    try:
        anim = make_animation(real_traj, model_traj, actions, args, diagnostics)

        if args.output is not None:
            save_animation(anim, args.output, args.fps)
            print(f"Saved visualization to {args.output}")

        if args.show or args.output is None:
            plt.show()
    finally:
        if keep_env is not None:
            keep_env.close()


if __name__ == "__main__":
    main()
