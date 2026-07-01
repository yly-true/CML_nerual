"""评估训练好的 Neural CML。"""

from __future__ import annotations

import argparse
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from cml.cml_model import (
    CML_HIDDEN_DIMS,
    CML_LATENT_DIM,
    CML_SNN_SURROGATE_SCALE,
    CML_SNN_TAU,
    CML_SNN_THRESHOLD,
    CML_SNN_TIMESTEPS,
    NeuralCML,
    plan_action_cem,
    plan_action_random_shooting,
)
from cml.tasks import feature_dim, feature_names, format_obs, obs_to_features, reset_eval_env, resolve_env_id, resolve_task_name, target_features
from cml.utils import get_dims, make_env, resolve_device


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="评估连续控制任务上的 Neural CML")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--task", choices=("pendulum", "cartpole"), default="pendulum")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--planner", type=str, choices=("random", "cem"), default="cem")
    parser.add_argument("--num-sequences", type=int, default=2048)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--cem-iterations", type=int, default=4)
    parser.add_argument("--cem-elite-frac", type=float, default=0.1)
    parser.add_argument("--action-cost", type=float, default=0)
    parser.add_argument("--action-sampling", type=str, choices=("uniform", "gaussian"), default="gaussian")
    parser.add_argument(
        "--action-std",
        type=float,
        nargs="*",
        default=None,
        help="Gaussian action sampling std. Provide one value or one per action dimension. Defaults to action range / 4.",
    )
    parser.add_argument("--target-angular-velocity", type=float, default=0.0)
    parser.add_argument("--target-cart-position", type=float, default=0.0)
    parser.add_argument(
        "--inference-mode",
        type=str,
        choices=("obsend", "obstraj", "latentend", "latenttraj"),
        default="obsend",
                        help="MPC objective: obsend, obstraj, latentend, or latenttraj.",
    )
    parser.add_argument(
        "--obs-cost-weights",
        type=float,
        nargs="*",
        default=None,
        help="Optional feature weights in the selected task's feature order.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-sleep", type=float, default=0.0)
    parser.add_argument("--render-every", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--reset-key", type=str, default="1", help="Terminal key that resets env.")
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
        f"latent_dim={model_args['latent_dim']}, hidden_dims={model_args['hidden_dims']}, "
        f"network_type={model_args['network_type']}, snn_timesteps={model_args['snn_timesteps']}, "
        f"snn_tau={model_args['snn_tau']}, snn_threshold={model_args['snn_threshold']}, "
        f"params={total_params:,}, trainable={trainable_params:,}, param_memory={param_mb:.3f} MB"
    )


def resolve_hidden_dims(saved_args: dict) -> list[int]:
    """Read hidden layer widths from new or legacy checkpoint args."""
    if "hidden_dims" in saved_args:
        return [int(dim) for dim in saved_args["hidden_dims"]]
    if "hidden_dim" in saved_args and "depth" in saved_args:
        return [int(saved_args["hidden_dim"])] * int(saved_args["depth"])
    return [int(dim) for dim in CML_HIDDEN_DIMS]


def resolve_model_args(ckpt: dict) -> dict[str, object]:
    """Read model structure from checkpoint, falling back to cml_model.py defaults."""
    saved_args = ckpt.get("args", {})
    return {
        "latent_dim": int(saved_args.get("latent_dim", CML_LATENT_DIM)),
        "hidden_dims": resolve_hidden_dims(saved_args),
        "network_type": saved_args.get("network_type", "mlp"),
        "snn_timesteps": int(saved_args.get("snn_timesteps", CML_SNN_TIMESTEPS)),
        "snn_tau": float(saved_args.get("snn_tau", CML_SNN_TAU)),
        "snn_threshold": float(saved_args.get("snn_threshold", CML_SNN_THRESHOLD)),
        "snn_surrogate_scale": float(saved_args.get("snn_surrogate_scale", CML_SNN_SURROGATE_SCALE)),
    }


def load_model(args: argparse.Namespace, obs_dim: int, action_dim: int, device: torch.device):
    """从 checkpoint 恢复模型。"""
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_args = resolve_model_args(ckpt)
    model = NeuralCML(
        obs_dim=obs_dim,
        action_dim=action_dim,
        latent_dim=model_args["latent_dim"],
        hidden_dims=model_args["hidden_dims"],
        network_type=model_args["network_type"],
        snn_timesteps=model_args["snn_timesteps"],
        snn_tau=model_args["snn_tau"],
        snn_threshold=model_args["snn_threshold"],
        snn_surrogate_scale=model_args["snn_surrogate_scale"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print_model_size(model, model_args, obs_dim, action_dim)
    return model


def prepare_planner_inputs(
    env,
    obs_dim: int,
    device: torch.device,
    args: argparse.Namespace,
):
    """构造规划阶段反复使用的常量张量。"""
    action_low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=device)
    action_high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=device)
    obs_weights_t = None
    if args.obs_cost_weights is not None:
        if len(args.obs_cost_weights) != obs_dim:
            raise ValueError(f"--obs-cost-weights must contain {obs_dim} values for this task.")
        obs_weights_t = torch.as_tensor(args.obs_cost_weights, dtype=torch.float32, device=device)
    action_std_t = None
    if args.action_std is not None:
        if len(args.action_std) == 0:
            raise ValueError("--action-std requires at least one value.")
        if len(args.action_std) == 1:
            action_std = np.full(action_low.shape, args.action_std[0], dtype=np.float32)
        elif len(args.action_std) == action_low.numel():
            action_std = np.asarray(args.action_std, dtype=np.float32)
        else:
            raise ValueError(f"--action-std must contain 1 or {action_low.numel()} values.")
        if np.any(action_std < 0):
            raise ValueError("--action-std values must be non-negative.")
        action_std_t = torch.as_tensor(action_std, dtype=torch.float32, device=device)
    return SimpleNamespace(
        action_low=action_low,
        action_high=action_high,
        obs_weights_t=obs_weights_t,
        action_std_t=action_std_t,
    )


def maybe_render(env, args: argparse.Namespace) -> None:
    """Render one frame through Gymnasium's human render backend."""
    if args.render:
        env.render()
        if args.render_sleep > 0:
            time.sleep(args.render_sleep)


class TerminalKeyListener:
    """Non-blocking terminal key listener for resets."""

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
        print(f"key listener enabled: press '{self.reset_key}' to reset.")
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


def reset_env_to_initial_state(task_name: str, env, args: argparse.Namespace) -> np.ndarray:
    """Reset environment and render/log the initial state."""
    obs = reset_eval_env(task_name, env)
    print(format_obs(task_name, "reset_obs", obs))
    maybe_render(env, args)
    return obs


def plan_action_mpc(
    model: NeuralCML,
    obs_t: torch.Tensor,
    goal_t: torch.Tensor,
    planner_inputs,
    objective: str,
    args: argparse.Namespace,
) -> torch.Tensor:
    """Plan an action sequence with the learned model and execute the first action."""
    if args.planner == "cem":
        return plan_action_cem(
            model=model,
            obs=obs_t,
            goal_obs=goal_t,
            action_low=planner_inputs.action_low,
            action_high=planner_inputs.action_high,
            num_sequences=args.num_sequences,
            horizon=args.horizon,
            action_cost=args.action_cost,
            objective=objective,
            iterations=args.cem_iterations,
            elite_frac=args.cem_elite_frac,
            obs_weights=planner_inputs.obs_weights_t,
        )
    return plan_action_random_shooting(
        model=model,
        obs=obs_t,
        goal_obs=goal_t,
        action_low=planner_inputs.action_low,
        action_high=planner_inputs.action_high,
        num_sequences=args.num_sequences,
        horizon=args.horizon,
        action_cost=args.action_cost,
        sampling_dist=args.action_sampling,
        action_std=planner_inputs.action_std_t,
        objective=objective,
        obs_weights=planner_inputs.obs_weights_t,
    )


def run_episode(
    env,
    model: NeuralCML,
    planner_inputs,
    task_name: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[float, int]:
    """执行一条评估 episode。"""
    obs = reset_env_to_initial_state(task_name, env, args)
    print(format_obs(task_name, "initial_obs", obs))
    ep_return = 0.0
    ep_len = 0
    objective = objective_from_inference_mode(args.inference_mode)

    with TerminalKeyListener(args.reset_key) as key_listener:
        for _ in range(args.max_steps):
            if key_listener.reset_requested():
                obs = reset_env_to_initial_state(task_name, env, args)
                continue

            obs_features = obs_to_features(task_name, obs)
            obs_t = torch.as_tensor(obs_features.astype(np.float32), dtype=torch.float32, device=device)

            goal_features = target_features(task_name, args)
            goal_t = torch.as_tensor(goal_features.astype(np.float32), dtype=torch.float32, device=device)

            action_t = plan_action_mpc(
                model=model,
                obs_t=obs_t,
                goal_t=goal_t,
                planner_inputs=planner_inputs,
                objective=objective,
                args=args,
            )
            action = action_t.detach().cpu().numpy()

            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += float(reward)
            ep_len += 1
            if args.log_every > 0 and ep_len % args.log_every == 0:
                target = target_features(task_name, args)
                print(f"{format_obs(task_name, 'obs', obs)}, target={target}")

            if args.render_every > 0 and ep_len % args.render_every == 0:
                maybe_render(env, args)

            if terminated or truncated:
                break

    return ep_return, ep_len


def main() -> None:
    args = parse_args()
    task_name = resolve_task_name(args.task)
    env_id = resolve_env_id(task_name)
    device = resolve_device(args.device)
    render_mode = "human" if args.render else None
    env = make_env(env_id, render_mode=render_mode)
    raw_obs_dim, action_dim = get_dims(env)
    expected_obs_dim = feature_dim(task_name, raw_obs_dim)
    model = load_model(args, expected_obs_dim, action_dim, device)
    planner_inputs = prepare_planner_inputs(env, expected_obs_dim, device, args)

    print(
        "inference: "
        f"task={task_name}, env_id={env_id}, mode={args.inference_mode}, planner={args.planner}, "
        f"objective={objective_from_inference_mode(args.inference_mode)}, "
        f"features={feature_names(task_name)}, action_cost={args.action_cost}, "
        f"action_sampling={args.action_sampling}, action_std={args.action_std}, "
        f"num_sequences={args.num_sequences}, horizon={args.horizon}, "
        f"cem_iterations={args.cem_iterations}, cem_elite_frac={args.cem_elite_frac}"
    )
    print(f"target: {target_features(task_name, args)}")
    if args.render:
        print(
            f"render: render_every={args.render_every}, render_sleep={args.render_sleep}, "
            f"log_every={args.log_every}, reset_key={args.reset_key}"
        )

    returns = []
    lengths = []
    for ep in range(args.episodes):
        ep_return, ep_len = run_episode(
            env=env,
            model=model,
            planner_inputs=planner_inputs,
            task_name=task_name,
            args=args,
            device=device,
        )

        returns.append(ep_return)
        lengths.append(ep_len)
        print(f"episode={ep} return={ep_return:.1f} length={ep_len}")

    print(f"mean_return={np.mean(returns):.1f} +/- {np.std(returns):.1f}")
    print(f"mean_length={np.mean(lengths):.1f} +/- {np.std(lengths):.1f}")
    env.close()


if __name__ == "__main__":
    main()
