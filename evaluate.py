"""评估训练好的 Neural CML 在 InvertedPendulum-v5 上的表现。"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
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
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--num-sequences", type=int, default=2048)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--action-cost", type=float, default=0)
    parser.add_argument("--action-sampling", type=str, choices=("uniform", "gaussian"), default="gaussian")
    parser.add_argument(
        "--target-cart",
        type=str,
        choices=("zero", "current"),
        default="zero",
        help="Use cart position target 0 for swing-up, or keep the current cart position.",
    )
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
    parser.add_argument("--init-pole-angle", type=float, default=0)
    parser.add_argument("--init-cart-vel", type=float, default=0)
    parser.add_argument("--init-pole-ang-vel", type=float, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--render", action="store_true")
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
    return model, ckpt["obs_mean"], ckpt["obs_std"]


def build_dynamic_target_features(obs: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """Build the swing-up target in feature space."""
    cart_pos = float(obs[0]) if args.target_cart == "current" else 0.0
    return upright_feature_target(obs, cart_pos=cart_pos, cart_vel=0.0, pole_ang_vel=0.0)


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


def maybe_render_human(env, args: argparse.Namespace) -> None:
    """Render one frame through Gymnasium's human render backend."""
    if args.render and args.render_backend == "human":
        env.render()


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
    obs, _ = env.reset()
    maybe_set_initial_state(env, args)
    obs = env.unwrapped._get_obs()
    print(format_obs_state("initial_obs", obs))
    ep_return = 0.0
    ep_len = 0
    if viewer is not None:
        viewer.sync()
    else:
        maybe_render_human(env, args)

    for _ in range(args.max_steps):
        if viewer is not None and not viewer.is_running():
            break

        obs_features = pendulum_obs_to_features(obs)
        norm_obs = obs_features.astype(np.float32)
        if use_normalize:
            norm_obs = normalize_obs(norm_obs, obs_mean, obs_std)
        obs_t = torch.as_tensor(norm_obs, dtype=torch.float32, device=device)

        goal_features = build_dynamic_target_features(obs, args)
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
        ep_return += float(reward)
        ep_len += 1
        print(format_obs_state("obs", obs))

        if viewer is not None:
            viewer.sync()
            time.sleep(max(float(env.unwrapped.model.opt.timestep), 0.01))
        else:
            maybe_render_human(env, args)
            if args.render and args.render_backend == "human":
                time.sleep(max(float(env.unwrapped.model.opt.timestep), 0.01))

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
    if args.render:
        print(f"render: backend={args.render_backend}")

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
