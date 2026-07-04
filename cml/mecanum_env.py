"""Gymnasium wrapper for the MuJoCo mecanum-wheel Summit XL model."""

from __future__ import annotations

import os
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


def _default_model_path() -> Path:
    root = os.environ.get("MECANUM_MUJOCO_ROOT")
    candidates = []
    if root:
        candidates.append(Path(root))
    candidates.extend(
        [
            Path(r"E:\mujoco_mecanum"),
            Path(__file__).resolve().parents[2] / "mujoco_mecanum",
            Path(__file__).resolve().parents[1] / "mujoco_mecanum",
        ]
    )
    for candidate in candidates:
        model_path = candidate / "robots" / "summit_xl_description" / "summit_xls.xml"
        if model_path.is_file():
            return model_path
    raise FileNotFoundError(
        "Could not find mujoco_mecanum model. Set MECANUM_MUJOCO_ROOT to the cloned repo path."
    )


class MecanumDriveEnv(gym.Env):
    """Summit XL mecanum drive exposed as a compact continuous-control task."""

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        render_mode: str | None = None,
        model_path: str | None = None,
        frame_skip: int = 10,
        ctrl_scale: float = 10.0,
        max_episode_steps: int = 1000,
    ) -> None:
        self.render_mode = render_mode
        self.model_path = Path(model_path) if model_path is not None else _default_model_path()
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.frame_skip = int(frame_skip)
        self.ctrl_scale = float(ctrl_scale)
        self.max_episode_steps = int(max_episode_steps)
        self.steps = 0
        self.viewer = None

        self.observation_space = spaces.Box(
            low=np.full(3 + self.model.nu, -np.inf, dtype=np.float32),
            high=np.full(3 + self.model.nu, np.inf, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        qvel = self.data.qvel
        wheel_vel = np.asarray(
            [
                qvel[6],   # front_right_wheel_rolling_joint
                qvel[19],  # front_left_wheel_rolling_joint
                qvel[32],  # back_right_wheel_rolling_joint
                qvel[45],  # back_left_wheel_rolling_joint
            ],
            dtype=np.float32,
        )
        base_vel = np.asarray([qvel[0], qvel[1], qvel[5]], dtype=np.float32)
        return np.concatenate([base_vel, wheel_vel]).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        _ = options
        mujoco.mj_resetData(self.model, self.data)
        self.steps = 0
        return self._get_obs(), {}

    def step(self, action):
        action_arr = np.asarray(action, dtype=np.float32)
        action_arr = np.clip(action_arr, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action_arr * self.ctrl_scale
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1

        obs = self._get_obs()
        reward = -float(obs[0] * obs[0] + obs[1] * obs[1] + obs[2] * obs[2])
        terminated = False
        truncated = self.steps >= self.max_episode_steps
        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode != "human":
            return None
        if self.viewer is None:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()
        return None

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
