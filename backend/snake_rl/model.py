"""Q-network used by the DQN agent.

Enhancements over the original 2-layer MLP:
    * 3-layer MLP with wider hidden size
    * Dueling architecture: separate value and advantage heads, combined
      via the standard ``Q = V + (A - mean(A))`` formulation. Dueling
      consistently improves stability for small action spaces.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingQNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 256, output_size: int = 3):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 2, 1),
        )
        self.adv_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.feature(x)
        value = self.value_head(h)
        adv = self.adv_head(h)
        return value + (adv - adv.mean(dim=-1, keepdim=True))

    def save(self, path: str | os.PathLike) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str | os.PathLike, map_location: str | torch.device = "cpu") -> None:
        self.load_state_dict(torch.load(path, map_location=map_location))
