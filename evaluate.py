"""评估训练好的 Neural CML 在 InvertedPendulum-v5 上的表现。"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from cml_model import NeuralCML, plan_action_random_shooting
from utils import get_dims, make_env, normalize_obs, pendulum_obs_to_features, resolve_device, upright_feature_target


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="评估 InvertedPendulum-v5 上的 Neural CML")
    parser.add_argument("--checkpoint", type=str, required=True)
    default_xml = str((Path(__file__).resolve().parent / "inverted_pendulum_v5.xml"))
    parser.add_argument("--env-id", type=str, default="InvertedPendulum-v5")
    parser.add_argument("--xml-file", type=str, default=default_xml)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--num-sequences", type=int, default=2048*4)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--action-cost", type=float, default=0)
    parser.add_argument("--action-sampling", type=str, choices=("uniform", "gaussian"), default="gaussian")
    parser.add_argument(
        "--target-cart",
        type=str,
        choices=("zero", "current", "sine"),
        default="sine",
        help="Cart position target: fixed 0, current position, or sinusoidal x over time.",
    )
    parser.add_argument("--target-x-amplitude", type=float, default=1.0, help="Amplitude for sine cart target.")
    parser.add_argument("--target-x-frequency", type=float, default=1.0, help="Frequency in Hz for sine cart target.")
    parser.add_argument("--target-x-phase", type=float, default=0.0, help="Phase in radians for sine cart target.")
    parser.add_argument("--target-x-offset", type=float, default=0.0, help="Offset for sine cart target.")
    parser.add_argument(
        "--inference-mode",
        type=str,
        choices=("obsend", "obstraj", "latentend", "latenttraj"),
        default="latentend",
        help="Random-shooting objective: obsend, obstraj, latentend, or latenttraj.",
    )
    parser.add_argument(
        "--obs-cost-weights",
        type=float,
        nargs="*",
        default=None,
        help="Optional feature weights: [x, sin(theta), cos(theta), xdot, thetadot].",
    )
    parser.add_argument("--init-cart-pos", type=float, default=0)
    parser.add_argument("--init-pole-angle", type=float, default=3.14)
    parser.add_argument("--init-cart-vel", type=float, default=0)
    parser.add_argument("--init-pole-ang-vel", type=float, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-sleep", type=float, default=0.0, help="Seconds to sleep after rendered frames.")
    parser.add_argument("--render-every", type=int, default=1, help="Render every N environment steps.")
    parser.add_argument("--log-every", type=int, default=20, help="Print obs every N steps. Use 0 to disable step logs.")
    parser.add_argument("--camera-name", type=str, default="track_cart", help="MuJoCo camera name used for rendering.")
    parser.add_argument("--reset-key", type=str, default="1", help="SSH terminal key that resets env to the init state.")
    parser.add_argument(
        "--render-backend",
        type=str,
        choices=("human", "passive"),
        default="passive",
        help="Use Gymnasium human rendering by default; passive keeps the MuJoCo passive viewer path.",
    )
    return parser.parse_args()


def objective_from_inference_mode(inference_mode: str) -> str:
    """Map short CLI names to cml_model scoring objectives."""
    objective_by_mode = {
        "obsend": "obs-terminal",
        "obstraj": "obs-trajectory",
        "latentend": "latent-terminal",
        "latenttraj": "latent-trajectory",
    }
    return objective_by_mode[inference_mode]


def print_model_size(model: NeuralCML, model_args: dict, obs_dim: int, action_dim: int) -> None:
    """打印当前加载模型的规模，便于确认 checkpoint 结构。"""
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    param_mb = sum(param.numel() * param.element_size() for param in model.parameters()) / 1024 / 1024
    print(
        "Model size: "
        f"obs_dim={obs_dim}, action_dim={action_dim}, "
        f"latent_dim={model_args['latent_dim']}, hidden_dim={model_args['hidden_dim']}, "
        f"depth={model_args['depth']}, "
        f"params={total_params:,}, trainable={trainable_params:,}, param_memory={param_mb:.3f} MB"
    )


def load_model_and_stats(args: argparse.Namespace, action_dim: int, device: torch.device):
    """从 checkpoint 恢复模型和观测标准化参数。"""
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_args = ckpt["args"]
    obs_dim = int(np.asarray(ckpt["obs_mean"]).shape[0])
    if obs_dim != 5:
        raise ValueError(
            "This swing-up evaluator expects a checkpoint trained with 5-D features "
            "[x, sin(theta), cos(theta), xdot, thetadot]. Retrain with the current "
            "train_cml_inverted_pendulum_v5.py."
        )
    model = NeuralCML(
        obs_dim=obs_dim,
        action_dim=action_dim,
        latent_dim=model_args["latent_dim"],
        hidden_dim=model_args["hidden_dim"],
        depth=model_args["depth"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print_model_size(model, model_args, obs_dim, action_dim)
    return model, ckpt["obs_mean"], ckpt["obs_std"]


def get_env_time(env, step_index: int) -> float:
    """Read simulator time, falling back to step_index * timestep if needed."""
    data = getattr(env.unwrapped, "data", None)
    sim_time = getattr(data, "time", None)
    if sim_time is not None:
        return float(sim_time)

    model = getattr(env.unwrapped, "model", None)
    timestep = getattr(getattr(model, "opt", None), "timestep", None)
    if timestep is not None:
        return float(step_index) * float(timestep)
    return float(step_index)


def target_cart_state(obs: np.ndarray, elapsed_time: float, args: argparse.Namespace) -> tuple[float, float]:
    """Return target cart position and velocity."""
    if args.target_cart == "current":
        return float(obs[0]), 0.0
    if args.target_cart == "sine":
        omega = 2.0 * math.pi * args.target_x_frequency
        angle = omega * elapsed_time + args.target_x_phase
        cart_pos = args.target_x_offset + args.target_x_amplitude * math.sin(angle)
        cart_vel = args.target_x_amplitude * omega * math.cos(angle)
        return cart_pos, cart_vel
    return 0.0, 0.0


def build_dynamic_target_features(obs: np.ndarray, elapsed_time: float, args: argparse.Namespace) -> np.ndarray:
    """Build the swing-up target in feature space."""
    cart_pos, cart_vel = target_cart_state(obs, elapsed_time, args)
    return upright_feature_target(obs, cart_pos=cart_pos, cart_vel=cart_vel, pole_ang_vel=0.0)


def prepare_planner_inputs(
    env,
    obs_dim: int,
    obs_mean,
    obs_std,
    device: torch.device,
    args: argparse.Namespace,
):
    """构造规划阶段反复使用的常量张量。"""
    action_low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=device)
    action_high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=device)
    obs_mean_t = torch.as_tensor(obs_mean, dtype=torch.float32, device=device)
    obs_std_t = torch.as_tensor(obs_std, dtype=torch.float32, device=device)
    obs_weights_t = None
    if args.obs_cost_weights is not None:
        if len(args.obs_cost_weights) != obs_dim:
            raise ValueError(f"--obs-cost-weights must contain {obs_dim} values for this environment.")
        obs_weights_t = torch.as_tensor(args.obs_cost_weights, dtype=torch.float32, device=device)
    return SimpleNamespace(
        action_low=action_low,
        action_high=action_high,
        obs_mean_t=obs_mean_t,
        obs_std_t=obs_std_t,
        obs_weights_t=obs_weights_t,
    )


def maybe_launch_mujoco_viewer(env, args: argparse.Namespace):
    """按需启动 MuJoCo 官方被动 viewer。"""
    if not args.render or args.render_backend != "passive":
        return nullcontext(None)

    try:
        import mujoco.viewer
    except ImportError as exc:
        raise RuntimeError("Official MuJoCo viewer requires the `mujoco` Python package.") from exc

    sim = env.unwrapped
    model = getattr(sim, "model", None)
    data = getattr(sim, "data", None)
    if model is None or data is None:
        raise RuntimeError("Current environment does not expose MuJoCo model/data for the official viewer.")
    return mujoco.viewer.launch_passive(model, data, show_left_ui=True, show_right_ui=True)


def configure_named_camera(model, cam, camera_name: str | None) -> None:
    """Switch a MuJoCo viewer camera to a named fixed/tracking XML camera."""
    if cam is None or not camera_name:
        return

    try:
        import mujoco
    except ImportError:
        return

    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id < 0:
        print(f"warning: camera '{camera_name}' not found in MuJoCo model.")
        return

    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = camera_id


def maybe_configure_passive_camera(env, viewer, args: argparse.Namespace) -> None:
    """Configure the official passive viewer camera if it is available."""
    if viewer is None:
        return
    configure_named_camera(env.unwrapped.model, getattr(viewer, "cam", None), args.camera_name)


def maybe_configure_human_camera(env, args: argparse.Namespace) -> None:
    """Configure Gymnasium's human viewer camera after the viewer is created."""
    renderer = getattr(env.unwrapped, "mujoco_renderer", None)
    viewer = getattr(renderer, "viewer", None)
    configure_named_camera(env.unwrapped.model, getattr(viewer, "cam", None), args.camera_name)


def maybe_render_human(env, args: argparse.Namespace) -> None:
    """Render one frame through Gymnasium's human render backend."""
    if args.render and args.render_backend == "human":
        env.render()
        maybe_configure_human_camera(env, args)


def maybe_sleep_after_render(args: argparse.Namespace) -> None:
    """Optional render pacing; default is no sleep for faster visualization."""
    if args.render_sleep > 0:
        time.sleep(args.render_sleep)


def format_obs_state(prefix: str, obs: np.ndarray) -> str:
    """Format InvertedPendulum observation values for logs."""
    obs = obs.astype(np.float32)
    if obs.shape[0] >= 4:
        return (
            f"{prefix}: "
            f"cart_pos={obs[0]:.4f}, "
            f"pole_angle={obs[1]:.4f}, "
            f"cart_vel={obs[2]:.4f}, "
            f"pole_ang_vel={obs[3]:.4f}"
        )
    return f"{prefix}: obs={obs}"


def maybe_set_initial_state(env, args: argparse.Namespace) -> None:
    """按命令行参数覆盖环境初始状态。"""
    overrides = (
        args.init_cart_pos,
        args.init_pole_angle,
        args.init_cart_vel,
        args.init_pole_ang_vel,
    )
    if all(value is None for value in overrides):
        return

    sim = env.unwrapped
    if not hasattr(sim, "set_state"):
        raise RuntimeError("Current environment does not expose set_state(qpos, qvel).")

    qpos = np.array(sim.data.qpos, dtype=np.float64, copy=True)
    qvel = np.array(sim.data.qvel, dtype=np.float64, copy=True)
    if args.init_cart_pos is not None:
        qpos[0] = args.init_cart_pos
    if args.init_pole_angle is not None:
        qpos[1] = args.init_pole_angle
    if args.init_cart_vel is not None:
        qvel[0] = args.init_cart_vel
    if args.init_pole_ang_vel is not None:
        qvel[1] = args.init_pole_ang_vel
    sim.set_state(qpos, qvel)


class TerminalKeyListener:
    """Non-blocking terminal key listener for SSH sessions.

    On Linux/macOS this reads one key at a time from the SSH terminal without
    requiring Enter. If stdin is not a TTY, the listener is disabled.
    """

    def __init__(self, reset_key: str = "1") -> None:
        self.reset_key = reset_key
        self.enabled = False
        self.fd = None
        self.old_settings = None
        self.termios = None

    def __enter__(self):
        if os.name == "nt":
            self.enabled = True
            print(f"keyboard listener enabled: press '{self.reset_key}' to reset.")
            return self

        if not sys.stdin.isatty():
            print("keyboard listener disabled: stdin is not a TTY.")
            return self

        import termios
        import tty

        self.termios = termios
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        self.enabled = True
        print(f"SSH key listener enabled: press '{self.reset_key}' to reset to init state.")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.enabled and os.name != "nt" and self.termios is not None and self.old_settings is not None:
            self.termios.tcsetattr(self.fd, self.termios.TCSADRAIN, self.old_settings)

    def _read_available_keys(self) -> list[str]:
        if not self.enabled:
            return []

        if os.name == "nt":
            try:
                import msvcrt
            except ImportError:
                return []

            keys = []
            while msvcrt.kbhit():
                keys.append(msvcrt.getwch())
            return keys

        import select

        keys = []
        while select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1)
            if not key:
                break
            keys.append(key)
        return keys

    def reset_requested(self) -> bool:
        return any(key == self.reset_key for key in self._read_available_keys())


def reset_env_to_initial_state(env, args: argparse.Namespace, viewer=None) -> np.ndarray:
    """Reset environment and reapply command-line initial state overrides."""
    env.reset()
    maybe_set_initial_state(env, args)
    obs = env.unwrapped._get_obs()
    print(format_obs_state("reset_obs", obs))
    if viewer is not None:
        maybe_configure_passive_camera(env, viewer, args)
        viewer.sync()
    else:
        maybe_render_human(env, args)
    return obs


def run_episode(
    env,
    model: NeuralCML,
    planner_inputs,
    obs_mean,
    obs_std,
    args: argparse.Namespace,
    device: torch.device,
    use_normalize: bool,
    viewer,
) -> tuple[float, int]:
    """执行一条评估 episode。"""
    obs = reset_env_to_initial_state(env, args, viewer)
    print(format_obs_state("initial_obs", obs))
    ep_return = 0.0
    ep_len = 0

    with TerminalKeyListener(args.reset_key) as key_listener:
        for _ in range(args.max_steps):
            if viewer is not None and not viewer.is_running():
                break

            if key_listener.reset_requested():
                obs = reset_env_to_initial_state(env, args, viewer)
                continue

            obs_features = pendulum_obs_to_features(obs)
            norm_obs = obs_features.astype(np.float32)
            if use_normalize:
                norm_obs = normalize_obs(norm_obs, obs_mean, obs_std)
            obs_t = torch.as_tensor(norm_obs, dtype=torch.float32, device=device)

            elapsed_time = get_env_time(env, ep_len)
            goal_features = build_dynamic_target_features(obs, elapsed_time, args)
            norm_goal = goal_features
            if use_normalize:
                norm_goal = normalize_obs(norm_goal, obs_mean, obs_std)
            goal_t = torch.as_tensor(norm_goal, dtype=torch.float32, device=device)
            objective = objective_from_inference_mode(args.inference_mode)

            action_t = plan_action_random_shooting(
                model=model,
                obs=obs_t,
                goal_obs=goal_t,
                action_low=planner_inputs.action_low,
                action_high=planner_inputs.action_high,
                num_sequences=args.num_sequences,
                horizon=args.horizon,
                action_cost=args.action_cost,
                sampling_dist=args.action_sampling,
                objective=objective,
                obs_mean=planner_inputs.obs_mean_t,
                obs_std=planner_inputs.obs_std_t,
                obs_weights=planner_inputs.obs_weights_t,
            )
            action = action_t.detach().cpu().numpy()

            obs, reward, terminated, truncated, _ = env.step(action)
            time.sleep(0.03)  # Small sleep to improve rendering smoothness; adjust as needed.
            ep_return += float(reward)
            ep_len += 1
            if args.log_every > 0 and ep_len % args.log_every == 0:
                target_x, target_xdot = target_cart_state(obs, get_env_time(env, ep_len), args)
                print(f"{format_obs_state('obs', obs)}, target_x={target_x:.4f}, target_xdot={target_xdot:.4f}")

            should_render = args.render_every > 0 and ep_len % args.render_every == 0
            if viewer is not None and should_render:
                viewer.sync()
                maybe_sleep_after_render(args)
            elif viewer is None and should_render:
                maybe_render_human(env, args)
                if args.render and args.render_backend == "human":
                    maybe_sleep_after_render(args)

            # if terminated or truncated:
            #     break

    return ep_return, ep_len


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    use_normalize = True
    render_mode = "human" if args.render and args.render_backend == "human" else None
    env = make_env(args.env_id, render_mode=render_mode, xml_file=args.xml_file)
    raw_obs_dim, action_dim = get_dims(env)
    if raw_obs_dim < 4:
        raise ValueError("Swing-up evaluation expects raw obs [x, theta, xdot, thetadot].")
    model, obs_mean, obs_std = load_model_and_stats(args, action_dim, device)
    obs_dim = int(np.asarray(obs_mean).shape[0])
    if use_normalize:
        planner_inputs = prepare_planner_inputs(env, obs_dim, obs_mean, obs_std, device, args)
    else:
        planner_inputs = prepare_planner_inputs(
            env,
            obs_dim,
            np.zeros(obs_dim, dtype=np.float32),
            np.ones(obs_dim, dtype=np.float32),
            device,
            args,
        )

    print(
        "inference: "
        f"mode={args.inference_mode}, planner=random, "
        f"objective={objective_from_inference_mode(args.inference_mode)}, "
        f"target=upright_features_cart_{args.target_cart}, action_cost={args.action_cost}, "
        f"num_sequences={args.num_sequences}, horizon={args.horizon}"
    )
    if args.target_cart == "sine":
        print(
            "target sine: "
            f"x={args.target_x_offset}+{args.target_x_amplitude}*sin("
            f"2*pi*{args.target_x_frequency}*t+{args.target_x_phase})"
        )
    if args.render:
        print(
            f"render: backend={args.render_backend}, render_every={args.render_every}, "
            f"render_sleep={args.render_sleep}, log_every={args.log_every}, "
            f"camera_name={args.camera_name}, reset_key={args.reset_key}"
        )

    returns = []
    lengths = []

    with maybe_launch_mujoco_viewer(env, args) as viewer:
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
                viewer=viewer,
            )

            returns.append(ep_return)
            lengths.append(ep_len)
            print(f"episode={ep} return={ep_return:.1f} length={ep_len}")

    print(f"mean_return={np.mean(returns):.1f} +/- {np.std(returns):.1f}")
    print(f"mean_length={np.mean(lengths):.1f} +/- {np.std(lengths):.1f}")
    env.close()


if __name__ == "__main__":
    main()
