"""Pendulum-v1 task helpers."""

from __future__ import annotations

import argparse
import math

import numpy as np


TASK_NAME = "pendulum"
ENV_ID = "Pendulum-v1"


def feature_names() -> list[str]:
    """Return feature names in model observation order."""
    return ["cos(theta)", "sin(theta)", "thetadot"]


def feature_dim(raw_obs_dim: int) -> int:
    """Return model feature dimension and validate raw observation shape."""
    expected_raw_dim = 3
    if raw_obs_dim < expected_raw_dim:
        raise ValueError(f"Pendulum-v1 expects raw obs dim >= {expected_raw_dim}, got {raw_obs_dim}.")
    return expected_raw_dim


def obs_to_features(obs) -> np.ndarray:
    """Pendulum-v1 already exposes [cos(theta), sin(theta), thetadot]."""
    return np.asarray(obs, dtype=np.float32)


def target_features(args: argparse.Namespace) -> np.ndarray:
    """Build the upright pendulum target in feature space."""
    return np.asarray([1.0, 0.0, args.target_angular_velocity], dtype=np.float32)


def reset_train_env(env, args: argparse.Namespace, seed: int | None = None) -> np.ndarray:
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


def reset_eval_env(env) -> np.ndarray:
    """Reset the Pendulum evaluation environment."""
    obs, _ = env.reset()
    return np.asarray(obs, dtype=np.float32)


def current_obs(env) -> np.ndarray:
    """Return the current Pendulum observation."""
    sim = env.unwrapped
    if hasattr(sim, "_get_obs"):
        return np.asarray(sim._get_obs(), dtype=np.float32)
    if hasattr(sim, "state"):
        theta, theta_dot = np.asarray(sim.state, dtype=np.float32)
        return np.asarray([np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32)
    raise RuntimeError("Current environment does not expose current observation.")


def format_obs(prefix: str, obs: np.ndarray) -> str:
    """Format Pendulum observation values for logs."""
    obs_arr = np.asarray(obs, dtype=np.float32)
    theta = math.atan2(float(obs_arr[1]), float(obs_arr[0]))
    return (
        f"{prefix}: "
        f"cos(theta)={obs_arr[0]:.4f}, "
        f"sin(theta)={obs_arr[1]:.4f}, "
        f"theta={theta:.4f}, "
        f"thetadot={obs_arr[2]:.4f}"
    )
