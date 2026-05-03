"""Task-specific adapters for supported Gymnasium control tasks."""

from __future__ import annotations

import argparse
import math

import numpy as np


TASK_ENV_IDS = {
    "inverted_pendulum": "InvertedPendulum-v5",
    "mountain_car_continuous": "MountainCarContinuous-v0",
    "pendulum": "Pendulum-v1",
}


def resolve_task_name(task: str, env_id: str | None) -> str:
    """Resolve an explicit or auto task name."""
    if task != "auto":
        if task not in TASK_ENV_IDS:
            raise ValueError(f"Unsupported task: {task}")
        return task

    if env_id is None:
        return "inverted_pendulum"

    env_id_lower = env_id.lower()
    if "mountaincarcontinuous" in env_id_lower:
        return "mountain_car_continuous"
    if "pendulum" in env_id_lower and "inverted" not in env_id_lower:
        return "pendulum"
    if "invertedpendulum" in env_id_lower:
        return "inverted_pendulum"
    raise ValueError(
        f"Cannot infer task from env_id={env_id!r}. "
        "Use --task inverted_pendulum, mountain_car_continuous, or pendulum."
    )


def resolve_env_id(task_name: str, env_id: str | None) -> str:
    """Use the task default env id unless the user provided one."""
    return env_id or TASK_ENV_IDS[task_name]


def supports_passive_mujoco_viewer(task_name: str) -> bool:
    """Only the custom MuJoCo inverted-pendulum path supports passive viewer setup."""
    return task_name == "inverted_pendulum"


def feature_names(task_name: str) -> list[str]:
    """Return feature names in model observation order."""
    if task_name == "inverted_pendulum":
        return ["x", "sin(theta)", "cos(theta)", "xdot", "thetadot"]
    if task_name == "mountain_car_continuous":
        return ["position", "velocity"]
    if task_name == "pendulum":
        return ["cos(theta)", "sin(theta)", "thetadot"]
    raise ValueError(f"Unsupported task: {task_name}")


def feature_dim(task_name: str, raw_obs_dim: int) -> int:
    """Return model feature dimension and validate raw observation shape."""
    expected_raw_dim = {
        "inverted_pendulum": 4,
        "mountain_car_continuous": 2,
        "pendulum": 3,
    }[task_name]
    if raw_obs_dim < expected_raw_dim:
        raise ValueError(f"{task_name} expects raw obs dim >= {expected_raw_dim}, got {raw_obs_dim}.")
    if task_name == "inverted_pendulum":
        return 5
    return expected_raw_dim


def obs_to_features(task_name: str, obs) -> np.ndarray:
    """Convert raw environment observation to model features."""
    obs_arr = np.asarray(obs, dtype=np.float32)
    if task_name == "inverted_pendulum":
        features = np.zeros(obs_arr.shape[:-1] + (5,), dtype=np.float32)
        features[..., 0] = obs_arr[..., 0]
        features[..., 1] = np.sin(obs_arr[..., 1])
        features[..., 2] = np.cos(obs_arr[..., 1])
        features[..., 3] = obs_arr[..., 2]
        features[..., 4] = obs_arr[..., 3]
        return features
    if task_name in ("mountain_car_continuous", "pendulum"):
        return obs_arr.astype(np.float32)
    raise ValueError(f"Unsupported task: {task_name}")


def _inverted_pendulum_target(obs: np.ndarray, elapsed_time: float, args: argparse.Namespace) -> np.ndarray:
    if args.target_cart == "current":
        cart_pos = float(obs[0])
        cart_vel = 0.0
    elif args.target_cart == "sine":
        omega = 2.0 * math.pi * args.target_x_frequency
        angle = omega * elapsed_time + args.target_x_phase
        cart_pos = args.target_x_offset + args.target_x_amplitude * math.sin(angle)
        cart_vel = args.target_x_amplitude * omega * math.cos(angle)
    else:
        cart_pos = 0.0
        cart_vel = 0.0
    return np.asarray([cart_pos, 0.0, 1.0, cart_vel, 0.0], dtype=np.float32)


def target_features(task_name: str, obs: np.ndarray, elapsed_time: float, args: argparse.Namespace) -> np.ndarray:
    """Build the feature-space control target for a task."""
    if task_name == "inverted_pendulum":
        return _inverted_pendulum_target(obs, elapsed_time, args)
    if task_name == "mountain_car_continuous":
        return np.asarray([args.target_position, args.target_velocity], dtype=np.float32)
    if task_name == "pendulum":
        return np.asarray([1.0, 0.0, args.target_angular_velocity], dtype=np.float32)
    raise ValueError(f"Unsupported task: {task_name}")


def reset_inverted_pendulum_train(env, args: argparse.Namespace, seed: int | None = None) -> np.ndarray:
    """Reset inverted pendulum to a random full-angle state."""
    if seed is None:
        env.reset()
    else:
        env.reset(seed=seed)
    sim = env.unwrapped
    if not hasattr(sim, "set_state"):
        raise RuntimeError("Current environment does not expose set_state(qpos, qvel).")

    qpos = np.array(sim.data.qpos, dtype=np.float64, copy=True)
    qvel = np.array(sim.data.qvel, dtype=np.float64, copy=True)
    qpos[0] = np.random.uniform(-args.random_cart_pos_range, args.random_cart_pos_range)
    qpos[1] = np.random.uniform(-np.pi, np.pi)
    qvel[0] = np.random.uniform(-args.random_cart_vel_range, args.random_cart_vel_range)
    qvel[1] = np.random.uniform(-args.random_pole_ang_vel_range, args.random_pole_ang_vel_range)
    sim.set_state(qpos, qvel)
    return current_obs(env)


def reset_mountain_car_train(env, args: argparse.Namespace, seed: int | None = None) -> np.ndarray:
    """Reset MountainCarContinuous to a broad random position/velocity state."""
    if seed is None:
        env.reset()
    else:
        env.reset(seed=seed)

    sim = env.unwrapped
    if not hasattr(sim, "state"):
        raise RuntimeError("Current MountainCar environment does not expose state.")

    pos_low = args.random_mountain_car_pos_low
    pos_high = args.random_mountain_car_pos_high
    if pos_low >= pos_high:
        raise ValueError("--random-mountain-car-pos-low must be smaller than --random-mountain-car-pos-high.")

    velocity_range = abs(args.random_mountain_car_vel_range)
    position = np.random.uniform(pos_low, pos_high)
    velocity = np.random.uniform(-velocity_range, velocity_range)
    sim.state = np.asarray([position, velocity], dtype=np.float32)
    return current_obs(env)


def reset_pendulum_train(env, args: argparse.Namespace, seed: int | None = None) -> np.ndarray:
    """Reset Pendulum to a broad random angle/angular-velocity state."""
    if seed is None:
        env.reset()
    else:
        env.reset(seed=seed)

    sim = env.unwrapped
    if not hasattr(sim, "state"):
        raise RuntimeError("Current Pendulum environment does not expose state.")

    theta_range = abs(args.random_pendulum_theta_range)
    angular_velocity_range = abs(args.random_pendulum_ang_vel_range)
    theta = np.random.uniform(-theta_range, theta_range)
    theta_dot = np.random.uniform(-angular_velocity_range, angular_velocity_range)
    sim.state = np.asarray([theta, theta_dot], dtype=np.float32)
    return current_obs(env)


def reset_train_env(task_name: str, env, args: argparse.Namespace, seed: int | None = None) -> np.ndarray:
    """Reset an environment for random transition collection."""
    if task_name == "inverted_pendulum":
        return reset_inverted_pendulum_train(env, args, seed=seed)
    if task_name == "mountain_car_continuous":
        return reset_mountain_car_train(env, args, seed=seed)
    if task_name == "pendulum":
        return reset_pendulum_train(env, args, seed=seed)
    obs, _ = env.reset(seed=seed)
    return np.asarray(obs, dtype=np.float32)


def set_inverted_pendulum_eval_state(env, args: argparse.Namespace) -> None:
    """Apply optional inverted-pendulum initial-state overrides."""
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


def reset_eval_env(task_name: str, env, args: argparse.Namespace) -> np.ndarray:
    """Reset an environment for evaluation."""
    obs, _ = env.reset()
    if task_name == "inverted_pendulum":
        set_inverted_pendulum_eval_state(env, args)
        return current_obs(env)
    return np.asarray(obs, dtype=np.float32)


def current_obs(env) -> np.ndarray:
    """Return current observation for envs that expose different helper names."""
    sim = env.unwrapped
    if hasattr(sim, "_get_obs"):
        return np.asarray(sim._get_obs(), dtype=np.float32)
    if hasattr(sim, "state"):
        return np.asarray(sim.state, dtype=np.float32)
    raise RuntimeError("Current environment does not expose current observation.")


def format_obs(task_name: str, prefix: str, obs: np.ndarray) -> str:
    """Format raw task observation values for logs."""
    obs_arr = np.asarray(obs, dtype=np.float32)
    if task_name == "inverted_pendulum":
        return (
            f"{prefix}: "
            f"cart_pos={obs_arr[0]:.4f}, "
            f"pole_angle={obs_arr[1]:.4f}, "
            f"cart_vel={obs_arr[2]:.4f}, "
            f"pole_ang_vel={obs_arr[3]:.4f}"
        )
    if task_name == "mountain_car_continuous":
        return f"{prefix}: position={obs_arr[0]:.4f}, velocity={obs_arr[1]:.4f}"
    if task_name == "pendulum":
        theta = math.atan2(float(obs_arr[1]), float(obs_arr[0]))
        return (
            f"{prefix}: "
            f"cos(theta)={obs_arr[0]:.4f}, "
            f"sin(theta)={obs_arr[1]:.4f}, "
            f"theta={theta:.4f}, "
            f"thetadot={obs_arr[2]:.4f}"
        )
    return f"{prefix}: obs={obs_arr}"
