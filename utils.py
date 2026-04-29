"""Utilities for MuJoCo CML training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import gymnasium as gym
import numpy as np
import torch


def make_env(env_id: str, render_mode: str | None = None):
    return gym.make(env_id, render_mode=render_mode)


def get_dims(env) -> tuple[int, int]:
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    return obs_dim, action_dim


def default_goal_obs(env, obs_dim: int) -> np.ndarray:
    """Target observation for InvertedPendulum.

    Gymnasium's InvertedPendulum observation is usually:
    [cart_position, pole_angle, cart_velocity, pole_angular_velocity].
    The upright equilibrium is therefore near all zeros.
    """
    goal = np.zeros(obs_dim, dtype=np.float32)
    return goal


def save_checkpoint(path: Path, model, optimizer, args: Dict[str, Any], obs_mean, obs_std) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "args": args,
            "obs_mean": obs_mean,
            "obs_std": obs_std,
        },
        path,
    )
    with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(args, f, indent=2)


def normalize_obs(obs: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (obs - mean) / (std + 1e-6)
