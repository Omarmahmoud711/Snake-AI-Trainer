"""Headless Snake game engine supporting four modes.

Modes:
    - "walls":     classic; snake dies on boundary.
    - "no_walls":  snake wraps around the board edges.
    - "rocks":     no walls; rocks spawn periodically and kill on contact.
    - "duel":      two snakes share the board; collision with self / wall /
                   the other snake = death for that snake.

The engine is pure-Python with no pygame dependency so it can be used both
for headless training and for streaming game state to a web frontend.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Tuple


GRID_W = 32  # number of columns
GRID_H = 24  # number of rows
BLOCK = 20   # pixel size of a cell (used by frontend)

ROCK_SPAWN_EVERY = 25     # frames between rock spawns in "rocks" mode
ROCK_LIFETIME = 220       # frames a rock stays before disappearing
INITIAL_LENGTH = 3


class Direction(IntEnum):
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3


# clockwise unit vectors aligned with Direction
DIR_VECTORS: Tuple[Tuple[int, int], ...] = (
    (1, 0),    # RIGHT
    (0, 1),    # DOWN
    (-1, 0),   # LEFT
    (0, -1),   # UP
)


@dataclass
class Snake:
    body: List[Tuple[int, int]]
    direction: Direction
    alive: bool = True
    score: int = 0
    steps_since_food: int = 0

    @property
    def head(self) -> Tuple[int, int]:
        return self.body[0]


@dataclass
class StepResult:
    rewards: List[float]
    done: bool          # whole episode finished
    scores: List[int]
    info: dict = field(default_factory=dict)


class SnakeGame:
    """Grid-based Snake environment.

    Action space (per snake): 3 discrete actions
        0 = go straight
        1 = turn right (clockwise)
        2 = turn left  (counter-clockwise)
    """

    def __init__(self, mode: str = "walls", seed: Optional[int] = None):
        if mode not in {"walls", "no_walls", "rocks", "duel"}:
            raise ValueError(f"unknown mode: {mode}")
        self.mode = mode
        self.rng = random.Random(seed)
        self.num_snakes = 2 if mode == "duel" else 1
        self.snakes: List[Snake] = []
        self.food: Tuple[int, int] = (0, 0)
        self.rocks: List[Tuple[int, int, int]] = []  # (x, y, frames_left)
        self.frame = 0
        self.reset()

    # ---------- public api ----------

    def reset(self) -> None:
        self.frame = 0
        self.rocks = []
        self.snakes = []
        if self.num_snakes == 1:
            mid_x, mid_y = GRID_W // 2, GRID_H // 2
            body = [(mid_x - i, mid_y) for i in range(INITIAL_LENGTH)]
            self.snakes.append(Snake(body=body, direction=Direction.RIGHT))
        else:
            # duel: place snakes on opposite sides facing each other
            left_body = [(GRID_W // 4 - i, GRID_H // 2) for i in range(INITIAL_LENGTH)]
            right_body = [(3 * GRID_W // 4 + i, GRID_H // 2) for i in range(INITIAL_LENGTH)]
            self.snakes.append(Snake(body=left_body, direction=Direction.RIGHT))
            self.snakes.append(Snake(body=right_body, direction=Direction.LEFT))
        self._place_food()

    def step(self, actions: List[int]) -> StepResult:
        """Advance one tick. ``actions`` length must equal number of snakes.

        Dead snakes ignore their action. The episode ends when all snakes are
        dead, or (single-snake) when the timeout for the current snake fires.
        """
        if len(actions) != self.num_snakes:
            raise ValueError(f"expected {self.num_snakes} actions, got {len(actions)}")
        self.frame += 1

        # rocks: tick lifetime, optionally spawn
        if self.mode == "rocks":
            self._tick_rocks()

        rewards = [0.0] * self.num_snakes

        # 1) compute new heads (only for alive snakes)
        new_heads: List[Optional[Tuple[int, int]]] = [None] * self.num_snakes
        for i, snake in enumerate(self.snakes):
            if not snake.alive:
                continue
            new_dir = self._apply_turn(snake.direction, actions[i])
            snake.direction = new_dir
            dx, dy = DIR_VECTORS[new_dir]
            nx, ny = snake.head[0] + dx, snake.head[1] + dy
            if self.mode in ("no_walls", "rocks", "duel"):
                nx %= GRID_W
                ny %= GRID_H
            new_heads[i] = (nx, ny)

        # 2) collision detection (do all checks before mutating bodies)
        new_dead: List[bool] = [False] * self.num_snakes
        for i, snake in enumerate(self.snakes):
            if not snake.alive:
                continue
            head = new_heads[i]
            assert head is not None
            if self._is_lethal(head, snake_idx=i, candidate_heads=new_heads):
                new_dead[i] = True

        # 3) head-on collision in duel: both die if heads swap or collide
        if self.num_snakes == 2 and self.snakes[0].alive and self.snakes[1].alive:
            h0, h1 = new_heads[0], new_heads[1]
            if h0 is not None and h1 is not None and h0 == h1:
                new_dead[0] = True
                new_dead[1] = True

        # 4) apply moves / kills / food
        for i, snake in enumerate(self.snakes):
            if not snake.alive:
                continue
            if new_dead[i]:
                snake.alive = False
                rewards[i] += -10.0
                continue
            head = new_heads[i]
            assert head is not None
            snake.body.insert(0, head)
            if head == self.food:
                snake.score += 1
                snake.steps_since_food = 0
                rewards[i] += 10.0
                self._place_food()
            else:
                snake.body.pop()
                snake.steps_since_food += 1
                # small per-step shaping: tiny penalty to discourage looping
                rewards[i] += -0.01

            # timeout if snake spends way too long without eating
            if snake.steps_since_food > 80 * len(snake.body):
                snake.alive = False
                rewards[i] += -10.0

        all_dead = all(not s.alive for s in self.snakes)
        scores = [s.score for s in self.snakes]
        return StepResult(rewards=rewards, done=all_dead, scores=scores)

    # ---------- helpers ----------

    @staticmethod
    def _apply_turn(direction: Direction, action: int) -> Direction:
        if action == 0:        # straight
            return direction
        if action == 1:        # turn right (clockwise)
            return Direction((int(direction) + 1) % 4)
        if action == 2:        # turn left (counter-clockwise)
            return Direction((int(direction) - 1) % 4)
        raise ValueError(f"bad action {action}")

    def _place_food(self) -> None:
        occupied = set()
        for s in self.snakes:
            occupied.update(s.body)
        for rx, ry, _ in self.rocks:
            occupied.add((rx, ry))
        free = [(x, y) for x in range(GRID_W) for y in range(GRID_H) if (x, y) not in occupied]
        if not free:
            self.food = (-1, -1)
            return
        self.food = self.rng.choice(free)

    def _tick_rocks(self) -> None:
        self.rocks = [(x, y, t - 1) for x, y, t in self.rocks if t > 1]
        if self.frame % ROCK_SPAWN_EVERY == 0:
            occupied = {(x, y) for x, y, _ in self.rocks}
            for s in self.snakes:
                occupied.update(s.body)
                # don't spawn directly in front of the head either
                hx, hy = s.head
                dx, dy = DIR_VECTORS[s.direction]
                occupied.add(((hx + dx) % GRID_W, (hy + dy) % GRID_H))
            occupied.add(self.food)
            free = [(x, y) for x in range(GRID_W) for y in range(GRID_H) if (x, y) not in occupied]
            if free:
                rx, ry = self.rng.choice(free)
                self.rocks.append((rx, ry, ROCK_LIFETIME))

    def _is_lethal(
        self,
        pt: Tuple[int, int],
        snake_idx: int,
        candidate_heads: Optional[List[Optional[Tuple[int, int]]]] = None,
    ) -> bool:
        x, y = pt
        # walls
        if self.mode == "walls":
            if x < 0 or x >= GRID_W or y < 0 or y >= GRID_H:
                return True
        # rocks
        if self.mode == "rocks":
            for rx, ry, _ in self.rocks:
                if rx == x and ry == y:
                    return True
        # self body (excluding tail tip which will move) and other snakes
        for i, snake in enumerate(self.snakes):
            if not snake.alive:
                continue
            body = snake.body
            # Tail tip will move out unless that snake is about to eat the food.
            # Conservatively treat tail as free if that snake didn't eat this turn.
            if candidate_heads is not None and candidate_heads[i] is not None:
                will_eat = candidate_heads[i] == self.food
                check_body = body if will_eat else body[:-1]
            else:
                check_body = body
            for j, seg in enumerate(check_body):
                if i == snake_idx and j == 0:
                    continue  # don't count own current head
                if seg == pt:
                    return True
        return False

    # ---------- introspection used by state extractor ----------

    def is_collision_for(self, snake_idx: int, pt: Tuple[int, int]) -> bool:
        """Whether *pt* would currently kill snake ``snake_idx``.

        Used by the state encoder to compute "danger" features. Treats the
        existing bodies (minus tails) as obstacles in addition to walls/rocks.
        """
        x, y = pt
        if self.mode in ("no_walls", "rocks", "duel"):
            x %= GRID_W
            y %= GRID_H
            pt = (x, y)
        elif self.mode == "walls" and (x < 0 or x >= GRID_W or y < 0 or y >= GRID_H):
            return True
        if self.mode == "rocks":
            for rx, ry, _ in self.rocks:
                if rx == x and ry == y:
                    return True
        for snake in self.snakes:
            if not snake.alive:
                continue
            for seg in snake.body[:-1]:
                if seg == pt:
                    return True
        return False

    def to_dict(self) -> dict:
        """Serialize current state for the frontend."""
        return {
            "mode": self.mode,
            "grid": {"w": GRID_W, "h": GRID_H, "block": BLOCK},
            "food": {"x": self.food[0], "y": self.food[1]},
            "rocks": [{"x": x, "y": y, "ttl": t} for x, y, t in self.rocks],
            "snakes": [
                {
                    "body": [{"x": x, "y": y} for x, y in s.body],
                    "direction": int(s.direction),
                    "alive": s.alive,
                    "score": s.score,
                }
                for s in self.snakes
            ],
            "frame": self.frame,
        }
