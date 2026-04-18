"""Headless training entrypoint.

Usage:
    python -m backend.snake_rl.train --mode walls --episodes 1500
    python -m backend.snake_rl.train --all --episodes 5000 --device cuda
    python -m backend.snake_rl.train --all --max-seconds 14400 --device cuda

Best-model checkpointing: during training we track the rolling mean score
over the last ``--rolling-window`` episodes and save weights whenever this
metric reaches a new high. This protects against the typical late-training
score dip caused by the agent exploring tail-risk states.

For ``duel`` mode, both snakes share the same online network (self-play).
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .agent import AgentConfig, DQNAgent
from .game import SnakeGame
from .state import encode_state


MODES = ["walls", "no_walls", "rocks", "duel"]
MODELS_DIR = Path(__file__).resolve().parents[2] / "backend" / "models"


def train_mode(
    mode: str,
    episodes: int = 5000,
    device: str = "cpu",
    out_path: Optional[Path] = None,
    log_every: int = 50,
    rolling_window: int = 100,
    seed: int = 0,
    max_seconds: Optional[float] = None,
    eps_decay_episodes: Optional[float] = None,
) -> Path:
    print(f"\n=== mode={mode!r}  episodes={episodes}  device={device}  budget={max_seconds}s ===", flush=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = AgentConfig()
    if eps_decay_episodes is not None:
        cfg.eps_decay_episodes = eps_decay_episodes
    elif episodes >= 2000:
        # stretch exploration to roughly first ~25% of the run
        cfg.eps_decay_episodes = max(250.0, episodes / 5.0)
    agent = DQNAgent(cfg, device=device)

    game = SnakeGame(mode=mode, seed=seed)
    out_path = out_path or (MODELS_DIR / f"{mode}.pth")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_score = 0
    best_mean = -1.0
    rolling = deque(maxlen=rolling_window)
    t0 = time.time()
    saved_at_episode = 0

    for ep in range(1, episodes + 1):
        game.reset()
        while True:
            states = []
            actions = []
            for i in range(game.num_snakes):
                if game.snakes[i].alive:
                    s = encode_state(game, i)
                    a = agent.select_action(s)
                    states.append(s)
                    actions.append(a)
                else:
                    states.append(None)
                    actions.append(0)

            result = game.step(actions)

            for i in range(game.num_snakes):
                if states[i] is None:
                    continue
                s_next = encode_state(game, i)
                done_i = result.done or not game.snakes[i].alive
                agent.remember(states[i], actions[i], result.rewards[i], s_next, float(done_i))

            agent.train_step()

            if result.done:
                break

        agent.end_episode()
        score = max(s.score for s in game.snakes)
        rolling.append(score)
        if score > best_score:
            best_score = score

        # checkpoint by rolling mean once we have a full window
        if len(rolling) >= rolling_window:
            mean_score = float(np.mean(rolling))
            if mean_score > best_mean:
                best_mean = mean_score
                saved_at_episode = ep
                agent.save(out_path)

        if ep % log_every == 0:
            mean = float(np.mean(rolling)) if rolling else 0.0
            elapsed = time.time() - t0
            print(
                f"  ep {ep:5d}/{episodes}  "
                f"score={score:3d}  best={best_score:3d}  "
                f"mean{rolling_window}={mean:5.2f}  "
                f"best_mean={best_mean:5.2f}  "
                f"eps={agent.epsilon:.3f}  "
                f"buffer={len(agent.memory):6d}  "
                f"elapsed={elapsed:6.1f}s",
                flush=True,
            )

        if max_seconds is not None and (time.time() - t0) >= max_seconds:
            print(f"  -> hit time budget ({max_seconds:.0f}s) at ep {ep}, stopping mode early", flush=True)
            break

    # if we never reached a full window (very short run), save the final state anyway
    if best_mean < 0:
        agent.save(out_path)
        saved_at_episode = ep

    elapsed = time.time() - t0
    print(
        f"saved -> {out_path}  best_score={best_score}  best_mean={best_mean:.2f}  "
        f"saved_at_ep={saved_at_episode}  total_elapsed={elapsed:.1f}s",
        flush=True,
    )
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, help="single mode to train")
    parser.add_argument("--all", action="store_true", help="train all four modes sequentially")
    parser.add_argument("--episodes", type=int, default=5000, help="max episodes per mode")
    parser.add_argument("--max-seconds", type=float, default=None, help="optional wall-clock budget per mode")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:<idx>")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rolling-window", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    if not args.mode and not args.all:
        parser.error("specify --mode <mode> or --all")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    modes = MODES if args.all else [args.mode]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for m in modes:
        train_mode(
            m,
            episodes=args.episodes,
            device=device,
            seed=args.seed,
            log_every=args.log_every,
            rolling_window=args.rolling_window,
            max_seconds=args.max_seconds,
        )


if __name__ == "__main__":
    main()
