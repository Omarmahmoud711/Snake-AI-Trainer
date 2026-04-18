"""DQN agent with target network + Double DQN updates.

These two upgrades over vanilla DQN are the smallest reliable change you
can make to improve learning stability and final score.

Components:
    * Replay buffer (uniform sampling, capped)
    * Target network synced every ``target_sync_steps`` updates
    * Double DQN target: ``r + gamma * Q_target(s', argmax_a Q_online(s', a))``
    * ε-greedy exploration with exponential decay
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .model import DuelingQNet
from .state import STATE_SIZE


@dataclass
class AgentConfig:
    state_size: int = STATE_SIZE
    hidden_size: int = 256
    n_actions: int = 3
    lr: float = 1e-3
    gamma: float = 0.95
    batch_size: int = 256
    memory_size: int = 100_000
    min_memory_for_train: int = 1_000
    target_sync_steps: int = 500
    learn_every: int = 4              # do a gradient update every N env steps
    eps_start: float = 1.0
    eps_end: float = 0.02
    eps_decay_episodes: float = 250.0  # ~exponential half-life in episodes
    grad_clip: float = 1.0


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next, done = zip(*batch)
        return (
            np.asarray(s, dtype=np.float32),
            np.asarray(a, dtype=np.int64),
            np.asarray(r, dtype=np.float32),
            np.asarray(s_next, dtype=np.float32),
            np.asarray(done, dtype=np.float32),
        )


class DQNAgent:
    def __init__(self, config: Optional[AgentConfig] = None, device: Optional[str] = None):
        self.cfg = config or AgentConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.online = DuelingQNet(self.cfg.state_size, self.cfg.hidden_size, self.cfg.n_actions).to(self.device)
        self.target = DuelingQNet(self.cfg.state_size, self.cfg.hidden_size, self.cfg.n_actions).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optim = torch.optim.Adam(self.online.parameters(), lr=self.cfg.lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.memory = ReplayBuffer(self.cfg.memory_size)
        self.train_steps = 0
        self.env_steps = 0
        self.episodes = 0

    # ---------- exploration ----------

    @property
    def epsilon(self) -> float:
        if self.cfg.eps_decay_episodes <= 0:
            return self.cfg.eps_end
        decay = np.exp(-self.episodes / self.cfg.eps_decay_episodes)
        return float(self.cfg.eps_end + (self.cfg.eps_start - self.cfg.eps_end) * decay)

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        if (not greedy) and random.random() < self.epsilon:
            return random.randrange(self.cfg.n_actions)
        with torch.no_grad():
            tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q = self.online(tensor)
        return int(q.argmax(dim=-1).item())

    # ---------- learning ----------

    def remember(self, s, a, r, s_next, done):
        self.memory.push(s, a, r, s_next, done)

    def train_step(self) -> Optional[float]:
        self.env_steps += 1
        if self.cfg.learn_every > 1 and (self.env_steps % self.cfg.learn_every) != 0:
            return None
        if len(self.memory) < max(self.cfg.batch_size, self.cfg.min_memory_for_train):
            return None
        s, a, r, s_next, done = self.memory.sample(self.cfg.batch_size)
        s_t = torch.from_numpy(s).to(self.device)
        a_t = torch.from_numpy(a).to(self.device)
        r_t = torch.from_numpy(r).to(self.device)
        s_next_t = torch.from_numpy(s_next).to(self.device)
        done_t = torch.from_numpy(done).to(self.device)

        q_pred = self.online(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Double DQN: action selection by online net, evaluation by target net
            next_actions = self.online(s_next_t).argmax(dim=-1, keepdim=True)
            next_q = self.target(s_next_t).gather(1, next_actions).squeeze(1)
            target = r_t + (1.0 - done_t) * self.cfg.gamma * next_q

        loss = self.loss_fn(q_pred, target)
        self.optim.zero_grad()
        loss.backward()
        if self.cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.grad_clip)
        self.optim.step()

        self.train_steps += 1
        if self.train_steps % self.cfg.target_sync_steps == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def end_episode(self) -> None:
        self.episodes += 1

    # ---------- persistence ----------

    def save(self, path) -> None:
        self.online.save(path)

    def load(self, path) -> None:
        self.online.load(path, map_location=self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.online.to(self.device)
        self.target.to(self.device)
