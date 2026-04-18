"""State encoding for the DQN agent.

The state is a fixed-length feature vector per snake. We deliberately keep
it compact (no full grid) so a small MLP suffices and training is fast,
while still carrying enough mode-aware signal:

    [0..2]   danger straight / right / left (one step ahead)
    [3..5]   danger 2 steps ahead in the same three directions
    [6..9]   one-hot of current heading (R, D, L, U)
    [10..13] food relative position flags (left, right, up, down)
    [14]     normalized chebyshev distance to food
    [15..17] (duel only) opponent head proximity in straight / right / left
    [18..20] (rocks only) nearest-rock proximity in straight / right / left
"""

from __future__ import annotations

import numpy as np

from .game import DIR_VECTORS, GRID_H, GRID_W, Direction, SnakeGame


STATE_SIZE = 21  # max width; unused slots stay at zero


def _candidate_squares(head, direction: Direction):
    dx_s, dy_s = DIR_VECTORS[direction]
    dx_r, dy_r = DIR_VECTORS[Direction((int(direction) + 1) % 4)]
    dx_l, dy_l = DIR_VECTORS[Direction((int(direction) - 1) % 4)]
    hx, hy = head
    return (
        (hx + dx_s, hy + dy_s),
        (hx + dx_r, hy + dy_r),
        (hx + dx_l, hy + dy_l),
        (hx + 2 * dx_s, hy + 2 * dy_s),
        (hx + 2 * dx_r, hy + 2 * dy_r),
        (hx + 2 * dx_l, hy + 2 * dy_l),
    )


def encode_state(game: SnakeGame, snake_idx: int = 0) -> np.ndarray:
    snake = game.snakes[snake_idx]
    state = np.zeros(STATE_SIZE, dtype=np.float32)
    if not snake.alive:
        return state

    head = snake.head
    direction = snake.direction
    s1, r1, l1, s2, r2, l2 = _candidate_squares(head, direction)

    state[0] = float(game.is_collision_for(snake_idx, s1))
    state[1] = float(game.is_collision_for(snake_idx, r1))
    state[2] = float(game.is_collision_for(snake_idx, l1))
    state[3] = float(game.is_collision_for(snake_idx, s2))
    state[4] = float(game.is_collision_for(snake_idx, r2))
    state[5] = float(game.is_collision_for(snake_idx, l2))

    state[6 + int(direction)] = 1.0  # one-hot of heading

    food = game.food
    fx, fy = food
    hx, hy = head
    # food relative direction (handle wrap if no walls)
    dx = fx - hx
    dy = fy - hy
    if game.mode in ("no_walls", "rocks", "duel"):
        # take shortest signed delta on a torus
        if dx > GRID_W // 2:
            dx -= GRID_W
        elif dx < -GRID_W // 2:
            dx += GRID_W
        if dy > GRID_H // 2:
            dy -= GRID_H
        elif dy < -GRID_H // 2:
            dy += GRID_H
    state[10] = float(dx < 0)   # food to the left
    state[11] = float(dx > 0)   # food to the right
    state[12] = float(dy < 0)   # food above
    state[13] = float(dy > 0)   # food below
    state[14] = max(abs(dx), abs(dy)) / max(GRID_W, GRID_H)

    if game.mode == "duel":
        opp_idx = 1 - snake_idx
        opp = game.snakes[opp_idx]
        if opp.alive:
            ohx, ohy = opp.head
            for slot, target in zip((15, 16, 17), (s1, r1, l1)):
                tx, ty = target
                # how close (in chebyshev) is opponent head?
                d = max(abs(tx - ohx), abs(ty - ohy))
                state[slot] = max(0.0, 1.0 - d / 5.0)

    if game.mode == "rocks" and game.rocks:
        for slot, target in zip((18, 19, 20), (s1, r1, l1)):
            tx, ty = target
            best = 1e9
            for rx, ry, _ in game.rocks:
                d = abs(tx - rx) + abs(ty - ry)
                if d < best:
                    best = d
            state[slot] = max(0.0, 1.0 - best / 6.0)

    return state
