"""Vectorized headless trainer.

Runs ``--num-envs`` independent SnakeGame instances in the same process.
At each tick we:

    1) collect the live snakes' states into one (B, state_size) tensor,
    2) batch-forward the online network to pick all actions at once,
    3) step every env, push transitions into a single shared replay buffer,
    4) optionally do a learner step.

This turns the workload from "1 env / 1 tiny forward" into "B envs / 1 fat
forward", which is what makes a GPU worthwhile for tiny networks like ours.

Best-by-rolling-mean checkpointing is applied per ``--mode``; weights land
at ``backend/models/<mode>.pth``.

Usage:
    python -m backend.snake_rl.vec_train --mode walls --num-envs 64 --max-seconds 1800 --device cuda
    python -m backend.snake_rl.vec_train --mode duel  --num-envs 32 --max-seconds 2400 --device cuda
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from .agent import AgentConfig, DQNAgent
from .game import SnakeGame
from .state import STATE_SIZE, encode_state


MODES = ["walls", "no_walls", "rocks", "duel"]
MODELS_DIR = Path(__file__).resolve().parents[2] / "backend" / "models"


def vec_train(
    mode: str,
    num_envs: int = 64,
    episodes: int = 100_000,
    device: str = "cpu",
    out_path: Optional[Path] = None,
    log_every: int = 200,
    rolling_window: int = 200,
    max_seconds: Optional[float] = None,
    seed: int = 0,
    eps_decay_episodes: Optional[float] = None,
    learner_updates_per_step: int = 1,
) -> Path:
    print(
        f"\n=== mode={mode!r}  num_envs={num_envs}  device={device}  "
        f"max_episodes={episodes}  budget={max_seconds}s ===",
        flush=True,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = AgentConfig()
    if eps_decay_episodes is not None:
        cfg.eps_decay_episodes = eps_decay_episodes
    else:
        cfg.eps_decay_episodes = max(500.0, episodes / 8.0)
    cfg.batch_size = 512
    cfg.learn_every = max(1, num_envs // 8)  # one learn per ~8 transitions per env
    agent = DQNAgent(cfg, device=device)

    games: List[SnakeGame] = [SnakeGame(mode=mode, seed=seed + 1000 * i) for i in range(num_envs)]
    out_path = out_path or (MODELS_DIR / f"{mode}.pth")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rolling = deque(maxlen=rolling_window)
    best_mean = -1.0
    best_score = 0
    finished_episodes = 0
    saved_at_episode = 0

    t0 = time.time()
    last_log_t = t0
    total_env_steps = 0

    while True:
        # ---- batch action selection -------------------------------------------------
        # Collect all alive snakes (snake_idx, env_idx, state)
        slots: List[tuple] = []  # (env_idx, snake_idx)
        states: List[np.ndarray] = []
        for env_idx, g in enumerate(games):
            for s_idx in range(g.num_snakes):
                if g.snakes[s_idx].alive:
                    slots.append((env_idx, s_idx))
                    states.append(encode_state(g, s_idx))

        if not slots:
            # all envs dead simultaneously: shouldn't happen because we reset
            # immediately, but keep the loop defensive
            for g in games:
                g.reset()
            continue

        states_np = np.stack(states, axis=0)
        with torch.no_grad():
            t = torch.from_numpy(states_np).to(agent.device, non_blocking=True)
            qvals = agent.online(t)
            greedy = qvals.argmax(dim=-1).cpu().numpy()
        eps = agent.epsilon
        rand_mask = np.random.rand(len(slots)) < eps
        rand_actions = np.random.randint(0, cfg.n_actions, size=len(slots))
        actions_flat = np.where(rand_mask, rand_actions, greedy)

        # group actions by env
        env_actions: List[List[int]] = [[0] * g.num_snakes for g in games]
        env_states: List[List[Optional[np.ndarray]]] = [[None] * g.num_snakes for g in games]
        for k, (env_idx, s_idx) in enumerate(slots):
            env_actions[env_idx][s_idx] = int(actions_flat[k])
            env_states[env_idx][s_idx] = states_np[k]

        # ---- step every env ---------------------------------------------------------
        for env_idx, g in enumerate(games):
            result = g.step(env_actions[env_idx])
            for s_idx in range(g.num_snakes):
                s_old = env_states[env_idx][s_idx]
                if s_old is None:
                    continue
                s_next = encode_state(g, s_idx)
                done_i = result.done or not g.snakes[s_idx].alive
                agent.remember(s_old, env_actions[env_idx][s_idx], result.rewards[s_idx], s_next, float(done_i))
            total_env_steps += 1
            if result.done:
                ep_score = max(s.score for s in g.snakes)
                rolling.append(ep_score)
                finished_episodes += 1
                if ep_score > best_score:
                    best_score = ep_score
                if len(rolling) >= rolling_window:
                    mean_score = float(np.mean(rolling))
                    if mean_score > best_mean:
                        best_mean = mean_score
                        saved_at_episode = finished_episodes
                        agent.save(out_path)
                agent.end_episode()
                g.reset()

        # ---- learner ----------------------------------------------------------------
        for _ in range(learner_updates_per_step):
            agent.train_step()

        # ---- bookkeeping ------------------------------------------------------------
        now = time.time()
        if (finished_episodes and finished_episodes % log_every == 0) or (now - last_log_t > 30):
            mean = float(np.mean(rolling)) if rolling else 0.0
            elapsed = now - t0
            envs_per_sec = total_env_steps / max(elapsed, 1e-3)
            print(
                f"  ep {finished_episodes:6d}  "
                f"best={best_score:3d}  "
                f"mean{rolling_window}={mean:5.2f}  "
                f"best_mean={best_mean:5.2f}  "
                f"eps={eps:.3f}  "
                f"buf={len(agent.memory):6d}  "
                f"steps/s={envs_per_sec:6.0f}  "
                f"saved_at_ep={saved_at_episode:6d}  "
                f"elapsed={elapsed:6.0f}s",
                flush=True,
            )
            last_log_t = now

        if finished_episodes >= episodes:
            print(f"  -> hit episode cap ({episodes}), stopping mode", flush=True)
            break
        if max_seconds is not None and (now - t0) >= max_seconds:
            print(f"  -> hit time budget ({max_seconds:.0f}s) at ep {finished_episodes}, stopping mode", flush=True)
            break

    if best_mean < 0:
        agent.save(out_path)
        saved_at_episode = finished_episodes

    elapsed = time.time() - t0
    print(
        f"saved -> {out_path}  best_score={best_score}  best_mean={best_mean:.2f}  "
        f"saved_at_ep={saved_at_episode}  episodes={finished_episodes}  "
        f"total_elapsed={elapsed:.1f}s",
        flush=True,
    )
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=200_000)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rolling-window", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--learner-updates", type=int, default=1, help="learner steps per env tick")
    parser.add_argument("--eps-decay-episodes", type=float, default=None)
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    vec_train(
        mode=args.mode,
        num_envs=args.num_envs,
        episodes=args.episodes,
        device=device,
        seed=args.seed,
        log_every=args.log_every,
        rolling_window=args.rolling_window,
        max_seconds=args.max_seconds,
        eps_decay_episodes=args.eps_decay_episodes,
        learner_updates_per_step=args.learner_updates,
    )


if __name__ == "__main__":
    main()
