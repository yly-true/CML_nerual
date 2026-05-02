"""项目共用工具函数。"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict

import gymnasium as gym
import numpy as np
import torch


def make_env(env_id: str, render_mode: str | None = None, xml_file: str | None = None):
    """创建 Gymnasium 环境。"""
    make_kwargs = {}
    if render_mode is not None:
        make_kwargs["render_mode"] = render_mode
    if xml_file is not None:
        xml_path = Path(xml_file)
        if not xml_path.is_absolute():
            xml_path = xml_path.resolve()
        # MuJoCo 在 Windows 上有时无法稳定读取包含非 ASCII 字符的路径。
        # 若路径里含中文等字符，就复制到纯 ASCII 临时目录后再加载。
        try:
            str(xml_path).encode("ascii")
            safe_xml_path = xml_path
        except UnicodeEncodeError:
            cache_dir = Path(tempfile.gettempdir()) / "mujoco_xml_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            suffix = hashlib.md5(str(xml_path).encode("utf-8")).hexdigest()[:8]
            safe_xml_path = cache_dir / f"{xml_path.stem}_{suffix}{xml_path.suffix}"
            shutil.copy2(xml_path, safe_xml_path)
        make_kwargs["xml_file"] = str(safe_xml_path)
    return gym.make(env_id, **make_kwargs)


def get_dims(env) -> tuple[int, int]:
    """读取观测维度和动作维度。"""
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    return obs_dim, action_dim


def pendulum_obs_to_features(obs) -> np.ndarray:
    """Convert raw [x, theta, xdot, thetadot] obs to periodic angle features.

    Returned order:
    [x, sin(theta), cos(theta), xdot, thetadot]
    """
    obs_arr = np.asarray(obs, dtype=np.float32)
    features = np.zeros(obs_arr.shape[:-1] + (5,), dtype=np.float32)
    features[..., 0] = obs_arr[..., 0]
    features[..., 1] = np.sin(obs_arr[..., 1])
    features[..., 2] = np.cos(obs_arr[..., 1])
    features[..., 3] = obs_arr[..., 2]
    features[..., 4] = obs_arr[..., 3]
    return features


def upright_feature_target(
    raw_obs,
    cart_pos: float = 0.0,
    cart_vel: float = 0.0,
    pole_ang_vel: float = 0.0,
) -> np.ndarray:
    """Build a feature-space upright target for swing-up control."""
    _ = raw_obs
    return np.asarray([cart_pos, 0.0, 1.0, cart_vel, pole_ang_vel], dtype=np.float32)


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
