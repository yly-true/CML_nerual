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
    if env_id == "ContinuousCartPole-v0":
        from cml.tasks import make_continuous_cartpole_env

        return make_continuous_cartpole_env(render_mode=render_mode)
    make_kwargs = {}
    if render_mode is not None:
        make_kwargs["render_mode"] = render_mode
    return gym.make(env_id, **make_kwargs)


def get_dims(env) -> tuple[int, int]:
    """读取观测维度和动作维度。"""
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    return obs_dim, action_dim


def resolve_device(device_name: str) -> torch.device:
    """根据用户输入和 CUDA 可用性选择实际设备。"""
    if device_name == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device(device_name)
    return torch.device("cpu")


def save_checkpoint(path: Path, model, optimizer, args: Dict[str, Any]) -> None:
    """保存模型参数和优化器状态。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "args": args,
        },
        path,
    )
    with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(args, f, indent=2)
