"""连续控制版本的 Neural CML 组件。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


def mlp(input_dim: int, hidden_dim: int, output_dim: int, depth: int = 2) -> nn.Sequential:
    """构建带 LayerNorm 的 MLP，用于稳定 latent 预测。"""
    layers = []
    last_dim = input_dim
    for _ in range(depth):
        layers += [
            nn.Linear(last_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        ]
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


@dataclass
class CMLLossOutput:
    """训练时返回的各项损失，便于日志输出。"""

    total: torch.Tensor
    pred: torch.Tensor
    recon: torch.Tensor
    action_norm: torch.Tensor
    residual_norm: torch.Tensor


class NeuralCML(nn.Module):
    """受 CML 启发的 latent dynamics 模型。

    对于观测 o_t 和动作 a_t：

        s_t = encoder(o_t)
        delta_t = action_encoder(a_t)
        s_next_hat = s_t + delta_t + residual([s_t, a_t])

    这里保留了论文里“动作在状态空间中对应位移”的核心想法，
    同时加入神经编码器、解码器和残差项，以适配连续控制任务。
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        depth: int = 2,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        self.encoder = mlp(obs_dim, hidden_dim, latent_dim, depth)
        self.decoder = mlp(latent_dim, hidden_dim, obs_dim, depth)
        self.action_encoder = mlp(action_dim, hidden_dim, latent_dim, depth)
        self.residual = mlp(latent_dim + action_dim, hidden_dim, latent_dim, depth)

        # 让残差分支从接近 0 开始，初始阶段更贴近原始 CML 的加法结构。
        final = self.residual[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """将观测编码到 latent state。"""
        return self.encoder(obs)

    def decode(self, state: torch.Tensor) -> torch.Tensor:
        """将 latent state 解码回观测空间。"""
        return self.decoder(state)

    def action_delta(self, action: torch.Tensor) -> torch.Tensor:
        """将动作编码为 latent 空间中的位移向量。"""
        return self.action_encoder(action)

    def transition(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """执行一步 latent 转移。"""
        residual_in = torch.cat([state, action], dim=-1)
        return state + self.action_delta(action) + self.residual(residual_in)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回当前 latent state 和预测的下一 latent state。"""
        state = self.encode(obs)
        next_state_hat = self.transition(state, action)
        return state, next_state_hat

    def loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
        pred_weight: float = 1.0,
        recon_weight: float = 0.05,
        action_norm_weight: float = 1e-4,
        residual_norm_weight: float = 1e-3,
    ) -> CMLLossOutput:
        """计算训练损失。

        包含三部分：
        1. latent 下一状态预测误差
        2. 观测重建误差
        3. action / residual 的范数正则
        """
        state, next_state_hat = self.forward(obs, action)
        with torch.no_grad():
            next_state_target = self.encode(next_obs)

        # 主目标：让预测的下一 latent state 靠近目标编码。
        pred = F.mse_loss(next_state_hat, next_state_target)

        # 重建项防止 latent 只会做转移预测，却丢失观测信息。
        obs_hat = self.decode(state)
        next_obs_hat = self.decode(next_state_hat)
        recon = 0.5 * F.mse_loss(obs_hat, obs) + 0.5 * F.mse_loss(next_obs_hat, next_obs)

        # 正则项约束 latent 几何不要过度发散。
        action_delta = self.action_delta(action)
        residual_in = torch.cat([state, action], dim=-1)
        residual_delta = self.residual(residual_in)
        action_norm = action_delta.pow(2).mean()
        residual_norm = residual_delta.pow(2).mean()

        total = (
            pred_weight * pred
            + recon_weight * recon
            + action_norm_weight * action_norm
            + residual_norm_weight * residual_norm
        )
        return CMLLossOutput(total, pred, recon, action_norm, residual_norm)


@torch.no_grad()
def sample_action_sequences(
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    num_sequences: int,
    horizon: int,
    action_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """在动作边界内均匀采样候选动作序列。"""
    shape = (num_sequences, horizon, action_dim)
    u = torch.rand(shape, device=device)
    return action_low + u * (action_high - action_low)


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
) -> torch.Tensor:
    """用短时域 random shooting 选择下一步动作。

    评分逻辑是：预测轨迹末端越接近目标 latent state 越好，
    同时对动作幅度施加一个轻微代价。
    这可以看作把论文中的单步 utility 扩展到了连续动作和多步预测。
    """
    device = obs.device
    action_dim = action_low.shape[-1]

    obs_batch = obs.unsqueeze(0).repeat(num_sequences, 1)
    goal_batch = goal_obs.unsqueeze(0).repeat(num_sequences, 1)

    state = model.encode(obs_batch)
    goal_state = model.encode(goal_batch)
    actions = sample_action_sequences(
        action_low=action_low,
        action_high=action_high,
        num_sequences=num_sequences,
        horizon=horizon,
        action_dim=action_dim,
        device=device,
    )

    total_action_cost = torch.zeros(num_sequences, device=device)
    for t in range(horizon):
        # 将每条候选序列在 latent 空间中向前 rollout。
        a_t = actions[:, t, :]
        state = model.transition(state, a_t)
        total_action_cost += a_t.pow(2).mean(dim=-1)

    latent_distance = (state - goal_state).pow(2).mean(dim=-1)
    score = -latent_distance - action_cost * total_action_cost
    best = torch.argmax(score)
    return actions[best, 0, :]
