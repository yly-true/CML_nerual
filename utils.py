"""项目共用工具函数。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import gymnasium as gym
import numpy as np
import torch


def make_env(env_id: str, render_mode: str | None = None):
    """创建 Gymnasium 环境。"""
    return gym.make(env_id, render_mode=render_mode)


def get_dims(env) -> tuple[int, int]:
    """读取观测维度和动作维度。"""
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    return obs_dim, action_dim


def default_goal_obs(env, obs_dim: int) -> np.ndarray:
    """返回 InvertedPendulum 的默认目标观测。

    该环境的观测通常接近：
    [小车位置, 杆角度, 小车速度, 杆角速度]
    因此竖直平衡点可以近似看作全零向量。
    """
    goal = np.zeros(obs_dim, dtype=np.float32)
    return goal


def resolve_device(device_name: str) -> torch.device:
    """根据用户输入和 CUDA 可用性选择实际设备。"""
    if device_name == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device(device_name)
    return torch.device("cpu")


def compute_obs_stats(obs):
    """计算观测归一化所需的均值和标准差。"""
    if isinstance(obs, torch.Tensor):
        obs_mean = obs.mean(dim=0)
        obs_std = obs.std(dim=0, unbiased=False) + 1e-6
        return obs_mean, obs_std
    obs_mean = obs.mean(axis=0).astype(np.float32)
    obs_std = obs.std(axis=0).astype(np.float32) + 1e-6
    return obs_mean, obs_std


def normalize_obs(obs, mean, std):
    """按给定统计量对单条或批量观测做标准化。"""
    return (obs - mean) / (std + 1e-6)


def normalize_replay_buffer(buffer, mean, std) -> None:
    """原地标准化 replay buffer 中已采集到的观测。"""
    buffer.obs[: buffer.size] = normalize_obs(buffer.obs[: buffer.size], mean, std)
    buffer.next_obs[: buffer.size] = normalize_obs(buffer.next_obs[: buffer.size], mean, std)


def save_checkpoint(path: Path, model, optimizer, args: Dict[str, Any], obs_mean, obs_std) -> None:
    """保存模型参数、优化器状态和归一化统计量。"""
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
