"""连续控制版本的 Neural CML 组件。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F

CML_LATENT_DIM = 256
CML_HIDDEN_DIMS = [128, 128]
CML_NETWORK_TYPE = "mlp"
CML_DYNAMICS_MODE = "latent"
CML_DERIVATIVE_DT = 0.02
CML_SNN_TIMESTEPS = 16
CML_SNN_TAU = 2.0
CML_SNN_THRESHOLD = 0.5
CML_SNN_SURROGATE_SCALE = 10.0

# CML_LATENT_DIM = 128
# CML_HIDDEN_DIMS = [64, 64]

# CML_LATENT_DIM = 512
# CML_HIDDEN_DIMS = [16, 64, 64]
# 1.同类任务下别人怎么做
# 2.多任务，比如组里的脉轮   ②
# 3.MLP更换为SNN，神经元换为LIF   ①



class SurrogateSpike(torch.autograd.Function):
    """Heaviside spike with a smooth surrogate gradient."""

    @staticmethod
    def forward(ctx, membrane_minus_threshold: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.save_for_backward(membrane_minus_threshold)
        ctx.scale = scale
        return (membrane_minus_threshold >= 0).to(membrane_minus_threshold.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (membrane_minus_threshold,) = ctx.saved_tensors
        scale = ctx.scale
        grad = 1.0 / (scale * membrane_minus_threshold.abs() + 1.0).pow(2)
        return grad_output * grad, None


def spike_fn(membrane_minus_threshold: torch.Tensor, scale: float) -> torch.Tensor:
    return SurrogateSpike.apply(membrane_minus_threshold, scale)


class LIFBlock(nn.Module):
    """Linear layer followed by a leaky integrate-and-fire neuron."""

    def __init__(self, input_dim: int, output_dim: int, tau: float, threshold: float, surrogate_scale: float) -> None:
        super().__init__()
        if tau <= 1.0:
            raise ValueError("SNN tau must be greater than 1.0.")
        self.linear = nn.Linear(input_dim, output_dim)
        self.decay = 1.0 - 1.0 / tau
        self.threshold = threshold
        self.surrogate_scale = surrogate_scale

    def forward(self, x: torch.Tensor, membrane: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        current = self.linear(x)
        if membrane is None:
            membrane = torch.zeros_like(current)
        membrane = self.decay * membrane + current
        spike = spike_fn(membrane - self.threshold, self.surrogate_scale)
        membrane = membrane * (1.0 - spike.detach())
        return spike, membrane


class SNN(nn.Module):
    """Static-input feed-forward SNN with LIF hidden layers and averaged readout."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        timesteps: int = CML_SNN_TIMESTEPS,
        tau: float = CML_SNN_TAU,
        threshold: float = CML_SNN_THRESHOLD,
        surrogate_scale: float = CML_SNN_SURROGATE_SCALE,
    ) -> None:
        super().__init__()
        if timesteps < 1:
            raise ValueError("SNN timesteps must be at least 1.")
        self.timesteps = int(timesteps)
        self.tau = float(tau)
        self.threshold = float(threshold)
        self.surrogate_scale = float(surrogate_scale)

        blocks = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            blocks.append(LIFBlock(last_dim, hidden_dim, tau, threshold, surrogate_scale))
            last_dim = hidden_dim
        self.blocks = nn.ModuleList(blocks)
        self.readout = nn.Linear(last_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        membranes: list[torch.Tensor | None] = [None] * len(self.blocks)
        output = torch.zeros((*x.shape[:-1], self.readout.out_features), dtype=x.dtype, device=x.device)
        for _ in range(self.timesteps):
            spikes = x
            for index, block in enumerate(self.blocks):
                spikes, membranes[index] = block(spikes, membranes[index])
            output = output + self.readout(spikes)
        return output / self.timesteps


def mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> nn.Sequential:
    """构建硬件友好的 ReLU MLP。"""
    layers = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        layers += [
            nn.Linear(last_dim, hidden_dim),
            nn.ReLU(),
        ]
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


def build_network(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    network_type: str = CML_NETWORK_TYPE,
    snn_timesteps: int = CML_SNN_TIMESTEPS,
    snn_tau: float = CML_SNN_TAU,
    snn_threshold: float = CML_SNN_THRESHOLD,
    snn_surrogate_scale: float = CML_SNN_SURROGATE_SCALE,
) -> nn.Module:
    """Build either a ReLU MLP or an LIF SNN with the same I/O shape."""
    if network_type == "mlp":
        return mlp(input_dim, hidden_dims, output_dim)
    if network_type == "snn":
        return SNN(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            timesteps=snn_timesteps,
            tau=snn_tau,
            threshold=snn_threshold,
            surrogate_scale=snn_surrogate_scale,
        )
    raise ValueError(f"Unsupported network_type: {network_type}")


@dataclass
class CMLLossOutput:
    """训练时返回的各项损失，便于日志输出。"""

    total: torch.Tensor
    pred: torch.Tensor
    recon: torch.Tensor
    dot: torch.Tensor
    continuity: torch.Tensor
    action_norm: torch.Tensor
    latent_norm: torch.Tensor


class NeuralCML(nn.Module):
    """受 CML 启发的 latent dynamics 模型。

    对于观测 o_t 和动作 a_t：

        s_t = encoder(o_t)
        delta_t = action_encoder([o_t, a_t])
        s_next_hat = s_t + delta_t

    这里保留了论文里“动作在状态空间中对应位移”的核心想法，
    同时让动作位移依赖当前 latent state，以适配连续控制任务。
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int = CML_LATENT_DIM,
        hidden_dims: Sequence[int] | None = None,
        network_type: str = CML_NETWORK_TYPE,
        dynamics_mode: str = CML_DYNAMICS_MODE,
        derivative_dt: float = CML_DERIVATIVE_DT,
        derivative_dim: int | None = None,
        snn_timesteps: int = CML_SNN_TIMESTEPS,
        snn_tau: float = CML_SNN_TAU,
        snn_threshold: float = CML_SNN_THRESHOLD,
        snn_surrogate_scale: float = CML_SNN_SURROGATE_SCALE,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = CML_HIDDEN_DIMS
        hidden_dims = tuple(int(dim) for dim in hidden_dims)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        if dynamics_mode not in ("latent", "obs_derivative"):
            raise ValueError(f"Unsupported dynamics_mode: {dynamics_mode}")
        self.dynamics_mode = dynamics_mode
        self.derivative_dt = float(derivative_dt)
        self.derivative_dim = obs_dim if derivative_dim is None else int(derivative_dim)
        if self.derivative_dim < 1 or self.derivative_dim > obs_dim:
            raise ValueError(f"derivative_dim must be in [1, {obs_dim}], got {self.derivative_dim}.")
        self.latent_dim = obs_dim if dynamics_mode == "obs_derivative" else latent_dim
        self.hidden_dims = hidden_dims
        self.network_type = network_type
        self.snn_timesteps = int(snn_timesteps)
        self.snn_tau = float(snn_tau)
        self.snn_threshold = float(snn_threshold)
        self.snn_surrogate_scale = float(snn_surrogate_scale)

        if self.dynamics_mode == "latent":
            self.encoder = build_network(
                obs_dim,
                hidden_dims,
                self.latent_dim,
                network_type,
                self.snn_timesteps,
                self.snn_tau,
                self.snn_threshold,
                self.snn_surrogate_scale,
            )
            self.decoder = build_network(
                self.latent_dim,
                hidden_dims,
                obs_dim,
                network_type,
                self.snn_timesteps,
                self.snn_tau,
                self.snn_threshold,
                self.snn_surrogate_scale,
            )
            self.action_encoder = build_network(
                obs_dim + action_dim,
                hidden_dims,
                self.latent_dim,
                network_type,
                self.snn_timesteps,
                self.snn_tau,
                self.snn_threshold,
                self.snn_surrogate_scale,
            )
            self.derivative_net = None
        else:
            self.encoder = None
            self.decoder = None
            self.action_encoder = None
            self.derivative_net = build_network(
                obs_dim + action_dim,
                hidden_dims,
                self.derivative_dim,
                network_type,
                self.snn_timesteps,
                self.snn_tau,
                self.snn_threshold,
                self.snn_surrogate_scale,
            )

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """将观测编码到 latent state。"""
        if self.dynamics_mode == "obs_derivative":
            return obs
        if self.encoder is None:
            raise RuntimeError("Encoder is not available for this dynamics mode.")
        return self.encoder(obs)

    def decode(self, state: torch.Tensor) -> torch.Tensor:
        """将 latent state 解码回观测空间。"""
        if self.dynamics_mode == "obs_derivative":
            return state
        if self.decoder is None:
            raise RuntimeError("Decoder is not available for this dynamics mode.")
        return self.decoder(state)

    def action_delta(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """根据当前 observation feature 和动作预测 latent 空间中的位移向量。"""
        if self.action_encoder is None:
            raise RuntimeError("action_delta is only available in latent dynamics mode.")
        action_in = torch.cat([obs, action], dim=-1)
        return self.action_encoder(action_in)

    def obs_derivative(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict d(obs[:derivative_dim]) / dt from current observation and action."""
        if self.derivative_net is None:
            raise RuntimeError("obs_derivative is only available in obs_derivative dynamics mode.")
        derivative_in = torch.cat([obs, action], dim=-1)
        return self.derivative_net(derivative_in)

    def predict_next_obs(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict the next observation for the configured dynamics mode."""
        if self.dynamics_mode == "latent":
            state = self.encode(obs)
            return self.decode(self.transition(state, action, obs))

        obs_dot = self.obs_derivative(obs, action)
        next_obs = obs.clone()
        next_obs[..., : self.derivative_dim] = obs[..., : self.derivative_dim] + self.derivative_dt * obs_dot
        return next_obs

    def transition(self, state: torch.Tensor, action: torch.Tensor, obs: torch.Tensor | None = None) -> torch.Tensor:
        """执行一步 latent 转移。

        action_encoder uses [obs, action]. During imagined rollouts, obs is
        decoded from the current latent state when a real observation is not
        available.
        """
        if self.dynamics_mode == "obs_derivative":
            if obs is None:
                obs = state
            return self.predict_next_obs(obs, action)
        if obs is None:
            obs = self.decode(state)
        return state + self.action_delta(obs, action)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回当前 latent state 和预测的下一 latent state。"""
        state = self.encode(obs)
        next_state_hat = self.transition(state, action, obs)
        return state, next_state_hat

    def loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
        pred_weight: float = 1.0,
        recon_weight: float = 0.05,
        recon_weights: torch.Tensor | None = None,
        continuity_weight: float = 0.0,
        continuity_dt: float = 0.05,
        action_norm_weight: float = 1e-4,
        latent_norm_weight: float = 1e-4,
    ) -> CMLLossOutput:
        """计算训练损失。

        包含三部分：
        1. latent 下一状态预测误差
        2. 观测重建误差
        3. action delta 的范数正则
        """
        if self.dynamics_mode == "obs_derivative":
            state, next_obs_hat = self.forward(obs, action)
            obs_dot_hat = self.obs_derivative(obs, action)
            obs_dot_target = (next_obs[..., : self.derivative_dim] - obs[..., : self.derivative_dim]) / self.derivative_dt
            derivative_weights = None
            if recon_weights is not None:
                derivative_weights = recon_weights[: self.derivative_dim]
            pred = weighted_mse(next_obs_hat, next_obs, recon_weights)
            recon = weighted_mse(obs_dot_hat, obs_dot_target, derivative_weights)
            continuity = pred.new_zeros(())
            action_norm = obs_dot_hat.pow(2).mean()
            latent_norm = state.pow(2).mean()
            total = (
                pred_weight * pred
                + recon_weight * recon
                + continuity_weight * continuity
                + action_norm_weight * action_norm
                + latent_norm_weight * latent_norm
            )
            return CMLLossOutput(total, pred, recon, recon, continuity, action_norm, latent_norm)

        state, next_state_hat = self.forward(obs, action)
        with torch.no_grad():
            next_state_target = self.encode(next_obs)

        # 主目标：让预测的下一 latent state 靠近目标编码。
        pred = F.mse_loss(next_state_hat, next_state_target)

        # 重建项防止 latent 只会做转移预测，却丢失观测信息。
        obs_hat = self.decode(state)
        next_obs_hat = self.decode(next_state_hat)
        recon = weighted_mse(obs_hat, obs, recon_weights) + weighted_mse(next_obs_hat, next_obs, recon_weights)
        continuity = cartpole_continuity_loss(obs, next_obs_hat, continuity_dt) if continuity_weight > 0.0 else pred.new_zeros(())

        # 正则项约束 latent 几何不要过度发散。
        action_delta = self.action_delta(obs, action)
        action_norm = action_delta.pow(2).mean()
        latent_norm = 0.5 * state.pow(2).mean() + 0.5 * next_state_hat.pow(2).mean()

        total = (
            pred_weight * pred
            + recon_weight * recon
            + continuity_weight * continuity
            + action_norm_weight * action_norm
            + latent_norm_weight * latent_norm
        )
        dot = pred.new_zeros(())
        return CMLLossOutput(total, pred, recon, dot, continuity, action_norm, latent_norm)


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    """MSE with optional per-observation-dimension weights."""
    err = (pred - target).pow(2)
    if weights is not None:
        err = err * weights.to(device=pred.device, dtype=pred.dtype).view(1, -1)
    return err.mean()


def cartpole_continuity_loss(obs: torch.Tensor, next_obs_hat: torch.Tensor, dt: float = 0.05) -> torch.Tensor:
    """Kinematic continuity prior for cartpole features.

    Feature order is [x, xdot, cos(theta), sin(theta), thetadot].
    The force affects acceleration, but x and theta should still advance
    smoothly from the current velocities over one environment step.
    """
    if obs.shape[-1] != 5 or next_obs_hat.shape[-1] != 5:
        return next_obs_hat.new_zeros(())

    x_target = obs[..., 0] + dt * obs[..., 1]
    theta = torch.atan2(obs[..., 3], obs[..., 2])
    theta_target = theta + dt * obs[..., 4]
    cos_target = torch.cos(theta_target)
    sin_target = torch.sin(theta_target)

    x_loss = F.mse_loss(next_obs_hat[..., 0], x_target)
    angle_loss = 0.5 * F.mse_loss(next_obs_hat[..., 2], cos_target) + 0.5 * F.mse_loss(next_obs_hat[..., 3], sin_target)
    return x_loss + angle_loss


@torch.no_grad()
def sample_action_sequences(
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    num_sequences: int,
    horizon: int,
    action_dim: int,
    device: torch.device,
    sampling_dist: str = "gaussian",
    action_std: torch.Tensor | None = None,
) -> torch.Tensor:
    """在动作边界内采样候选动作序列。"""
    shape = (num_sequences, horizon, action_dim)
    low = action_low.view(1, 1, action_dim)
    high = action_high.view(1, 1, action_dim)

    if sampling_dist == "uniform":
        u = torch.rand(shape, device=device)
        return low + u * (high - low)

    if sampling_dist == "gaussian":
        mean = torch.zeros(shape, device=device)
        if action_std is None:
            std = ((action_high - action_low) / 4.0).view(1, 1, action_dim)
        else:
            std = action_std.to(device=device, dtype=torch.float32).view(1, 1, action_dim)
        std = std.expand(shape)
        actions = torch.normal(mean=mean, std=std)
        return torch.clamp(actions, min=low, max=high)

    raise ValueError(f"Unsupported sampling_dist: {sampling_dist}")


def plan_action_continuous(
    model: NeuralCML,
    obs: torch.Tensor,
    goal_obs: torch.Tensor,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    action_cost: float = 1e-3,
    opt_steps: int = 32,
    opt_lr: float = 0.05,
) -> torch.Tensor:
    """Optimize the current continuous action with the CML utility.

    The optimized objective is:

        U(a) = (s_goal - s)^T (transition(s, a) - s) - lambda * ||a||^2

    This is a single-step continuous CML action choice, not sequence sampling
    or MPC. The model weights are treated as fixed; only the action tensor is
    optimized.
    """
    action_dim = action_low.shape[-1]
    low = action_low.view(1, action_dim)
    high = action_high.view(1, action_dim)
    requires_grad = [param.requires_grad for param in model.parameters()]

    try:
        for param in model.parameters():
            param.requires_grad_(False)

        with torch.no_grad():
            state = model.encode(obs.view(1, -1)).detach()
            goal_state = model.encode(goal_obs.view(1, -1)).detach()
            direction = goal_state - state
            init = torch.zeros((1, action_dim), dtype=torch.float32, device=obs.device)
            init = torch.clamp(init, min=low, max=high)

        with torch.enable_grad():
            action = init.detach().clone().requires_grad_(True)
            optimizer = torch.optim.Adam([action], lr=opt_lr)

            for _ in range(opt_steps):
                optimizer.zero_grad(set_to_none=True)
                delta = model.transition(state, action, obs.view(1, -1)) - state
                utility = (direction * delta).sum(dim=-1) - action_cost * action.pow(2).sum(dim=-1)
                loss = -utility.mean()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    action.clamp_(min=low, max=high)

        return action.detach().view(action_dim)
    finally:
        for param, old_requires_grad in zip(model.parameters(), requires_grad):
            param.requires_grad_(old_requires_grad)


@torch.no_grad()
def _normalize_weights(weights: torch.Tensor | None, dim: int, device: torch.device) -> torch.Tensor:
    """Return per-dimension weights for observation/state costs."""
    if weights is None:
        return torch.ones(dim, dtype=torch.float32, device=device)
    return weights.to(device=device, dtype=torch.float32).view(dim)


@torch.no_grad()
def score_action_sequences(
    model: NeuralCML,
    obs: torch.Tensor,
    goal_obs: torch.Tensor,
    actions: torch.Tensor,
    action_cost: float = 0.001,
    objective: str = "latent-terminal",
    obs_weights: torch.Tensor | None = None,
    terminal_weight: float = 1.0,
) -> torch.Tensor:
    """Score candidate action sequences using learned CML latent rollouts.

    Supported objectives:
    - latent-terminal: original behavior, only terminal latent distance
    - latent-trajectory: accumulated latent distance along the rollout
    - obs-terminal: terminal decoded-observation distance
    - obs-trajectory: accumulated decoded-observation distance
    """
    device = obs.device
    num_sequences, horizon, _ = actions.shape

    obs_batch = obs.unsqueeze(0).repeat(num_sequences, 1)
    goal_batch = goal_obs.unsqueeze(0).repeat(num_sequences, 1)

    state = model.encode(obs_batch)
    rollout_obs = obs_batch
    use_latent = objective.startswith("latent")
    use_obs = objective.startswith("obs")
    use_trajectory = objective.endswith("trajectory")
    use_terminal = objective.endswith("terminal")
    if not ((use_latent or use_obs) and (use_trajectory or use_terminal)):
        raise ValueError(f"Unsupported objective: {objective}")

    if use_latent:
        goal_state = model.encode(goal_batch)
        weights = _normalize_weights(None, model.latent_dim, device)
    else:
        goal_raw = goal_batch
        weights = _normalize_weights(obs_weights, model.obs_dim, device)

    distance_cost = torch.zeros(num_sequences, device=device)
    total_action_cost = torch.zeros(num_sequences, device=device)
    for t in range(horizon):
        a_t = actions[:, t, :]
        state = model.transition(state, a_t, rollout_obs)
        rollout_obs = model.decode(state)
        total_action_cost += a_t.pow(2).mean(dim=-1)

        if use_trajectory:
            if use_latent:
                step_distance = ((state - goal_state).pow(2) * weights.view(1, -1)).mean(dim=-1)
            else:
                step_distance = ((rollout_obs - goal_raw).pow(2) * weights.view(1, -1)).mean(dim=-1)
            distance_cost += step_distance

    if use_terminal:
        if use_latent:
            terminal_distance = ((state - goal_state).pow(2) * weights.view(1, -1)).mean(dim=-1)
        else:
            terminal_distance = ((rollout_obs - goal_raw).pow(2) * weights.view(1, -1)).mean(dim=-1)
        distance_cost = terminal_weight * terminal_distance
    else:
        distance_cost = distance_cost / max(horizon, 1)

    score = -distance_cost - action_cost * total_action_cost
    return score


def action_sequence_cost(
    model: NeuralCML,
    obs: torch.Tensor,
    goal_obs: torch.Tensor,
    actions: torch.Tensor,
    action_cost: float = 0.001,
    objective: str = "obs-trajectory",
    obs_weights: torch.Tensor | None = None,
    terminal_weight: float = 1.0,
    action_smooth_cost: float = 0.0,
) -> torch.Tensor:
    """Differentiable MPC cost for a single action sequence.

    This is nonlinear MPC over the learned neural dynamics. The cost has the
    usual quadratic tracking/action form, but the dynamics are not linearized
    into a QP.
    """
    if actions.dim() != 2:
        raise ValueError("actions must have shape [horizon, action_dim].")

    horizon = actions.shape[0]
    state = model.encode(obs.view(1, -1))
    rollout_obs = obs.view(1, -1)
    goal_batch = goal_obs.view(1, -1)
    use_latent = objective.startswith("latent")
    use_obs = objective.startswith("obs")
    use_trajectory = objective.endswith("trajectory")
    use_terminal = objective.endswith("terminal")
    if not ((use_latent or use_obs) and (use_trajectory or use_terminal)):
        raise ValueError(f"Unsupported objective: {objective}")

    if use_latent:
        goal_state = model.encode(goal_batch)
        weights = torch.ones(model.latent_dim, dtype=torch.float32, device=obs.device)
    else:
        weights = _normalize_weights(obs_weights, model.obs_dim, obs.device)

    distance_cost = obs.new_zeros(())
    for t in range(horizon):
        action_t = actions[t].view(1, -1)
        state = model.transition(state, action_t, rollout_obs)
        rollout_obs = model.decode(state)
        if use_trajectory:
            if use_latent:
                step_distance = ((state - goal_state).pow(2) * weights.view(1, -1)).mean()
            else:
                step_distance = ((rollout_obs - goal_batch).pow(2) * weights.view(1, -1)).mean()
            distance_cost = distance_cost + step_distance

    if use_terminal:
        if use_latent:
            distance_cost = terminal_weight * ((state - goal_state).pow(2) * weights.view(1, -1)).mean()
        else:
            distance_cost = terminal_weight * ((rollout_obs - goal_batch).pow(2) * weights.view(1, -1)).mean()
    else:
        distance_cost = distance_cost / max(horizon, 1)

    control_cost = action_cost * actions.pow(2).mean()
    smooth_cost = obs.new_zeros(())
    if action_smooth_cost > 0.0 and horizon > 1:
        smooth_cost = action_smooth_cost * (actions[1:] - actions[:-1]).pow(2).mean()
    return distance_cost + control_cost + smooth_cost


def plan_action_gradient_mpc(
    model: NeuralCML,
    obs: torch.Tensor,
    goal_obs: torch.Tensor,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    horizon: int = 8,
    action_cost: float = 0.001,
    objective: str = "obs-trajectory",
    opt_steps: int = 32,
    opt_lr: float = 0.05,
    obs_weights: torch.Tensor | None = None,
    terminal_weight: float = 1.0,
    action_smooth_cost: float = 0.0,
) -> torch.Tensor:
    """Optimize a finite-horizon action sequence by gradient-based NMPC."""
    action_dim = action_low.shape[-1]
    low = action_low.view(1, action_dim)
    high = action_high.view(1, action_dim)
    action_span = torch.clamp(high - low, min=1e-6)
    requires_grad = [param.requires_grad for param in model.parameters()]

    try:
        for param in model.parameters():
            param.requires_grad_(False)

        raw_actions = torch.zeros((horizon, action_dim), dtype=torch.float32, device=obs.device, requires_grad=True)
        optimizer = torch.optim.Adam([raw_actions], lr=opt_lr)

        for _ in range(opt_steps):
            optimizer.zero_grad(set_to_none=True)
            actions = low + action_span * torch.sigmoid(raw_actions)
            cost = action_sequence_cost(
                model=model,
                obs=obs,
                goal_obs=goal_obs,
                actions=actions,
                action_cost=action_cost,
                objective=objective,
                obs_weights=obs_weights,
                terminal_weight=terminal_weight,
                action_smooth_cost=action_smooth_cost,
            )
            cost.backward()
            optimizer.step()

        with torch.no_grad():
            actions = low + action_span * torch.sigmoid(raw_actions)
            return actions[0].detach()
    finally:
        for param, old_requires_grad in zip(model.parameters(), requires_grad):
            param.requires_grad_(old_requires_grad)


@torch.no_grad()
def plan_action_random_shooting(
    model: NeuralCML,
    obs: torch.Tensor,
    goal_obs: torch.Tensor,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    num_sequences: int = 1024,
    horizon: int = 8,
    action_cost: float = 0.001,
    sampling_dist: str = "gaussian",
    action_std: torch.Tensor | None = None,
    objective: str = "latent-terminal",
    obs_weights: torch.Tensor | None = None,
    terminal_weight: float = 1.0,
) -> torch.Tensor:
    """用短时域 random shooting 选择下一步动作。"""
    device = obs.device
    action_dim = action_low.shape[-1]
    actions = sample_action_sequences(
        action_low=action_low,
        action_high=action_high,
        num_sequences=num_sequences,
        horizon=horizon,
        action_dim=action_dim,
        device=device,
        sampling_dist=sampling_dist,
        action_std=action_std,
    )
    score = score_action_sequences(
        model=model,
        obs=obs,
        goal_obs=goal_obs,
        actions=actions,
        action_cost=action_cost,
        objective=objective,
        obs_weights=obs_weights,
        terminal_weight=terminal_weight,
    )
    best = torch.argmax(score)
    return actions[best, 0, :]


@torch.no_grad()
def plan_action_cem(
    model: NeuralCML,
    obs: torch.Tensor,
    goal_obs: torch.Tensor,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    num_sequences: int = 1024,
    horizon: int = 8,
    action_cost: float = 0.001,
    objective: str = "latent-trajectory",
    iterations: int = 4,
    elite_frac: float = 0.1,
    init_std_scale: float = 0.5,
    min_std: float = 1e-3,
    obs_weights: torch.Tensor | None = None,
    terminal_weight: float = 1.0,
) -> torch.Tensor:
    """Cross-Entropy Method MPC over CML latent rollouts."""
    device = obs.device
    action_dim = action_low.shape[-1]
    shape = (num_sequences, horizon, action_dim)
    low = action_low.view(1, 1, action_dim)
    high = action_high.view(1, 1, action_dim)
    mean = torch.zeros((horizon, action_dim), dtype=torch.float32, device=device)
    std = ((action_high - action_low) * init_std_scale).view(1, action_dim).repeat(horizon, 1)
    elite_count = max(1, int(num_sequences * elite_frac))

    best_actions = None
    best_score = None
    for _ in range(iterations):
        actions = mean.view(1, horizon, action_dim) + std.view(1, horizon, action_dim) * torch.randn(shape, device=device)
        actions = torch.clamp(actions, min=low, max=high)
        score = score_action_sequences(
            model=model,
            obs=obs,
            goal_obs=goal_obs,
            actions=actions,
            action_cost=action_cost,
            objective=objective,
            obs_weights=obs_weights,
            terminal_weight=terminal_weight,
        )
        elite_idx = torch.topk(score, elite_count).indices
        elites = actions[elite_idx]
        mean = elites.mean(dim=0)
        std = torch.clamp(elites.std(dim=0, unbiased=False), min=min_std)

        iter_best = torch.argmax(score)
        if best_score is None or score[iter_best] > best_score:
            best_score = score[iter_best]
            best_actions = actions[iter_best]

    if best_actions is None:
        return mean[0]
    return best_actions[0]
