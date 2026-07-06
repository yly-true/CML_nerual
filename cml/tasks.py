"""Task helpers for supported continuous-control systems."""

from __future__ import annotations

import argparse
import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces


PENDULUM = "pendulum"
CARTPOLE = "cartpole"
MECANUM = "mecanum"
SUPPORTED_TASKS = (PENDULUM, CARTPOLE, MECANUM)

TASK_ENV_IDS = {
    PENDULUM: "Pendulum-v1",
    CARTPOLE: "ContinuousCartPole-v0",
    MECANUM: "MecanumDrive-v0",
}

TASK_DERIVATIVE_DT = {
    PENDULUM: 0.05,
    CARTPOLE: 0.05,
    MECANUM: 0.02,
}

FEATURE_NAMES = {
    PENDULUM: ["cos(theta)", "sin(theta)", "thetadot"],
    CARTPOLE: ["x", "xdot", "cos(theta)", "sin(theta)", "thetadot"],
    MECANUM: [
        "body_vx",
        "body_vy",
        "body_yaw_rate",
        "front_right_wheel_speed",
        "front_left_wheel_speed",
        "back_right_wheel_speed",
        "back_left_wheel_speed",
    ],
}


def resolve_task_name(task: str) -> str:
    if task not in TASK_ENV_IDS:
        raise ValueError(f"Unsupported task: {task}. Supported tasks: {', '.join(SUPPORTED_TASKS)}")
    return task


def resolve_env_id(task_name: str) -> str:
    return TASK_ENV_IDS[task_name]


def derivative_dt(task_name: str) -> float:
    resolve_task_name(task_name)
    return TASK_DERIVATIVE_DT[task_name]


def feature_names(task_name: str) -> list[str]:
    resolve_task_name(task_name)
    return FEATURE_NAMES[task_name]


def feature_dim(task_name: str, raw_obs_dim: int) -> int:
    expected_raw_dim_by_task = {
        PENDULUM: 3,
        CARTPOLE: 4,
        MECANUM: 7,
    }
    resolve_task_name(task_name)
    expected_raw_dim = expected_raw_dim_by_task[task_name]
    if raw_obs_dim < expected_raw_dim:
        raise ValueError(f"{task_name} expects raw obs dim >= {expected_raw_dim}, got {raw_obs_dim}.")
    return len(FEATURE_NAMES[task_name])


def obs_to_features(task_name: str, obs) -> np.ndarray:
    obs_arr = np.asarray(obs, dtype=np.float32)
    if task_name == PENDULUM:
        return obs_arr
    if task_name == CARTPOLE:
        features = np.zeros(obs_arr.shape[:-1] + (5,), dtype=np.float32)
        features[..., 0] = obs_arr[..., 0]
        features[..., 1] = obs_arr[..., 2]
        features[..., 2] = np.cos(obs_arr[..., 1])
        features[..., 3] = np.sin(obs_arr[..., 1])
        features[..., 4] = obs_arr[..., 3]
        return features
    if task_name == MECANUM:
        return obs_arr
    raise ValueError(f"Unsupported task: {task_name}")


def obs_to_features_from_env(task_name: str, obs, env) -> np.ndarray:
    """Convert observation to model features, using env internals when needed."""
    _ = env
    return obs_to_features(task_name, obs)


def target_features(task_name: str, args: argparse.Namespace) -> np.ndarray:
    if task_name == PENDULUM:
        return np.asarray([1.0, 0.0, args.target_angular_velocity], dtype=np.float32)
    if task_name == CARTPOLE:
        return np.asarray([args.target_cart_position, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    if task_name == MECANUM:
        target = np.zeros(7, dtype=np.float32)
        target[0] = args.target_mecanum_vx
        target[1] = args.target_mecanum_vy
        target[2] = args.target_mecanum_yaw_rate
        return target
    raise ValueError(f"Unsupported task: {task_name}")


def reset_train_env(task_name: str, env, args: argparse.Namespace, seed: int | None = None) -> np.ndarray:
    if task_name == PENDULUM:
        return reset_pendulum_train(env, args, seed=seed)
    if task_name == CARTPOLE:
        return reset_cartpole_train(env, args, seed=seed)
    if task_name == MECANUM:
        return reset_mecanum_train(env, args, seed=seed)
    raise ValueError(f"Unsupported task: {task_name}")


def reset_pendulum_train(env, args: argparse.Namespace, seed: int | None = None) -> np.ndarray:
    """Reset Pendulum to the upright equilibrium state."""
    _ = args
    if seed is None:
        env.reset()
    else:
        env.reset(seed=seed)

    sim = env.unwrapped
    if not hasattr(sim, "state"):
        raise RuntimeError("Current Pendulum environment does not expose state.")

    theta = 0.0
    theta_dot = 0.0
    sim.state = np.asarray([theta, theta_dot], dtype=np.float32)
    return current_obs(PENDULUM, env)


def reset_cartpole_train(env, args: argparse.Namespace, seed: int | None = None) -> np.ndarray:
    """Reset the cart-pole to x=0, upright pole, and zero velocities."""
    _ = args
    if seed is None:
        env.reset()
    else:
        env.reset(seed=seed)

    sim = env.unwrapped
    cart_pos = 0.0
    pole_angle = 0.0
    cart_vel = 0.0
    pole_ang_vel = 0.0
    sim.state = np.asarray([cart_pos, pole_angle, cart_vel, pole_ang_vel], dtype=np.float32)
    return current_obs(CARTPOLE, env)


def reset_mecanum_train(env, args: argparse.Namespace, seed: int | None = None) -> np.ndarray:
    """Reset the mecanum base to the origin."""
    _ = args
    if seed is None:
        obs, _ = env.reset()
    else:
        obs, _ = env.reset(seed=seed)
    return np.asarray(obs, dtype=np.float32)


def reset_eval_env(task_name: str, env) -> np.ndarray:
    obs, _ = env.reset()
    return np.asarray(obs, dtype=np.float32)


def current_obs(task_name: str, env) -> np.ndarray:
    sim = env.unwrapped
    if task_name == PENDULUM:
        if hasattr(sim, "_get_obs"):
            return np.asarray(sim._get_obs(), dtype=np.float32)
        if hasattr(sim, "state"):
            theta, theta_dot = np.asarray(sim.state, dtype=np.float32)
            return np.asarray([np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32)
    if task_name == CARTPOLE and hasattr(sim, "state"):
        return np.asarray(sim.state, dtype=np.float32)
    raise RuntimeError("Current environment does not expose current observation.")


def format_obs(task_name: str, prefix: str, obs: np.ndarray) -> str:
    obs_arr = np.asarray(obs, dtype=np.float32)
    if task_name == PENDULUM:
        theta = math.atan2(float(obs_arr[1]), float(obs_arr[0]))
        return (
            f"{prefix}: "
            f"cos(theta)={obs_arr[0]:.4f}, "
            f"sin(theta)={obs_arr[1]:.4f}, "
            f"theta={theta:.4f}, "
            f"thetadot={obs_arr[2]:.4f}"
        )
    if task_name == CARTPOLE:
        return (
            f"{prefix}: "
            f"x={obs_arr[0]:.4f}, "
            f"theta={obs_arr[1]:.4f}, "
            f"xdot={obs_arr[2]:.4f}, "
            f"thetadot={obs_arr[3]:.4f}"
        )
    if task_name == MECANUM:
        return (
            f"{prefix}: "
            f"body_vx={obs_arr[0]:.4f}, "
            f"body_vy={obs_arr[1]:.4f}, "
            f"body_yaw_rate={obs_arr[2]:.4f}"
        )
    raise ValueError(f"Unsupported task: {task_name}")


class ContinuousCartPoleEnv(gym.Env):
    """Classic single-link cart-pole with continuous horizontal force."""

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, render_mode: str | None = None) -> None:
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.05
        self.x_threshold = 4.8
        self.theta_threshold_radians = math.pi
        self.max_episode_steps = 500
        self.render_mode = render_mode
        high = np.asarray(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.asarray([-self.force_mag], dtype=np.float32),
            high=np.asarray([self.force_mag], dtype=np.float32),
            dtype=np.float32,
        )
        self.state = np.zeros(4, dtype=np.float32)
        self.steps = 0
        self._fig = None
        self._ax = None

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,)).astype(np.float32)
        self.steps = 0
        return self.state.copy(), {}

    def step(self, action):
        force = float(np.clip(np.asarray(action, dtype=np.float32)[0], -self.force_mag, self.force_mag))
        x, theta, x_dot, theta_dot = [float(v) for v in self.state]
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = angle_normalize(theta + self.tau * theta_dot)
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = np.asarray([x, theta, x_dot, theta_dot], dtype=np.float32)
        self.steps += 1
        terminated = bool(abs(x) > self.x_threshold or abs(theta) > self.theta_threshold_radians)
        truncated = self.steps >= self.max_episode_steps
        reward = float(-(x**2 + 10.0 * angle_normalize(theta) ** 2 + 0.1 * x_dot**2 + 0.1 * theta_dot**2 + 0.001 * force**2))
        if self.render_mode == "human":
            self.render()
        return self.state.copy(), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode != "human":
            return None
        import matplotlib.pyplot as plt

        if self._fig is None:
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(6, 3))
        assert self._ax is not None
        x, theta, _, _ = self.state
        pole_x = x + math.sin(theta)
        pole_y = math.cos(theta)
        self._ax.clear()
        self._ax.set_xlim(-self.x_threshold, self.x_threshold)
        self._ax.set_ylim(-1.2, 1.2)
        self._ax.set_aspect("equal")
        self._ax.grid(True, alpha=0.25)
        self._ax.plot([-self.x_threshold, self.x_threshold], [0, 0], color="black", lw=1)
        self._ax.add_patch(plt.Rectangle((x - 0.2, -0.1), 0.4, 0.2, fill=False, lw=2))
        self._ax.plot([x, pole_x], [0, pole_y], lw=3)
        self._ax.plot([pole_x], [pole_y], marker="o")
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        return None

    def close(self):
        if self._fig is not None:
            import matplotlib.pyplot as plt

            plt.close(self._fig)
            self._fig = None
            self._ax = None


def angle_normalize(angle: float) -> float:
    return ((angle + math.pi) % (2.0 * math.pi)) - math.pi


def make_continuous_cartpole_env(render_mode: str | None = None):
    return ContinuousCartPoleEnv(render_mode=render_mode)
