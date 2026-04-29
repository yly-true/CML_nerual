"""A minimal replay buffer for self-supervised transition learning."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class Batch:
    obs: torch.Tensor
    actions: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    def __init__(self, obs_dim: int, action_dim: int, capacity: int) -> None:
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.capacity = capacity
        self.size = 0
        self.ptr = 0

    def add(self, obs, action, next_obs, done: bool) -> None:
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> Batch:
        if self.size == 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")
        idx = np.random.randint(0, self.size, size=batch_size)
        return Batch(
            obs=torch.as_tensor(self.obs[idx], device=device),
            actions=torch.as_tensor(self.actions[idx], device=device),
            next_obs=torch.as_tensor(self.next_obs[idx], device=device),
            dones=torch.as_tensor(self.dones[idx], device=device),
        )
