"""用于自监督转移动力学学习的简单 PyTorch 回放缓冲区。"""

from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class Batch:
    """一次训练采样得到的小批量数据。

    形状约定：
    - obs: (batch_size, obs_dim)
    - actions: (batch_size, action_dim)
    - next_obs: (batch_size, obs_dim)
    - dones: (batch_size, 1)
    """

    obs: torch.Tensor
    actions: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    """按环形队列方式存储转移样本。

    底层张量形状：
    - self.obs: (capacity, obs_dim)
    - self.actions: (capacity, action_dim)
    - self.next_obs: (capacity, obs_dim)
    - self.dones: (capacity, 1)
    """

    def __init__(self, obs_dim: int, action_dim: int, capacity: int) -> None:
        # 每一行对应一条 transition 的当前观测 obs_t。
        self.obs = torch.zeros((capacity, obs_dim), dtype=torch.float32)
        # 每一行对应一条 transition 的动作 action_t。
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32)
        # 每一行对应一条 transition 的下一观测 next_obs_t。
        self.next_obs = torch.zeros((capacity, obs_dim), dtype=torch.float32)
        # done 标记保留成二维张量，方便后续与 batch 维对齐。
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32)
        self.capacity = capacity
        self.size = 0
        self.ptr = 0

    def add(self, obs, action, next_obs, done: bool) -> None:
        """写入一条转移，满了以后覆盖最旧样本。

        输入形状：
        - obs: (obs_dim,)
        - action: (action_dim,)
        - next_obs: (obs_dim,)
        - done: 标量布尔值
        """
        self.obs[self.ptr] = torch.as_tensor(obs, dtype=torch.float32)
        self.actions[self.ptr] = torch.as_tensor(action, dtype=torch.float32)
        self.next_obs[self.ptr] = torch.as_tensor(next_obs, dtype=torch.float32)
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> Batch:
        """随机采样一个 batch，并直接转成目标设备上的张量。

        返回形状：
        - obs: (batch_size, obs_dim)
        - actions: (batch_size, action_dim)
        - next_obs: (batch_size, obs_dim)
        - dones: (batch_size, 1)
        """
        if self.size == 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")
        idx = torch.randint(0, self.size, (batch_size,))
        return Batch(
            obs=self.obs[idx].to(device),
            actions=self.actions[idx].to(device),
            next_obs=self.next_obs[idx].to(device),
            dones=self.dones[idx].to(device),
        )
