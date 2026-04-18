"""FastAPI server for the Snake-RL web app.

Routes
------
GET  /                       single-page app shell
GET  /api/status             which trained models are available
WS   /ws/learn/{mode}        live training stream (server controls episode loop)
WS   /ws/play/{mode}         server plays a pre-trained agent (or two, for duel)
WS   /ws/compete             user vs AI in duel mode (user sends key strokes)

WebSocket protocol (server -> client) sends JSON messages of the form::

    {"type": "frame", "state": {...}, "stats": {...}}
    {"type": "episode_end", "stats": {...}}
    {"type": "error", "message": "..."}
    {"type": "info", "message": "..."}

Client -> server (during compete mode)::

    {"type": "input", "action": 0|1|2}   # straight / right-turn / left-turn
    {"type": "speed", "value": <int fps>}
    {"type": "pause", "value": true|false}
    {"type": "reset"}
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .snake_rl.agent import AgentConfig, DQNAgent
from .snake_rl.game import Direction, SnakeGame
from .snake_rl.state import encode_state


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
MODELS_DIR = ROOT / "backend" / "models"
MODES = ["walls", "no_walls", "rocks", "duel"]

app = FastAPI(title="Snake-RL")
app.mount("/static", StaticFiles(directory=FRONTEND / "static"), name="static")


def pick_device() -> str:
    """Pick CPU by default; only use CUDA if explicitly available AND idle.

    The web server is interactive — we don't want to grab a GPU just to render
    snake frames. Training uses its own device flag.
    """
    return "cpu"


@app.get("/")
async def index():
    return FileResponse(FRONTEND / "templates" / "index.html")


@app.get("/api/status")
async def status():
    available = {m: (MODELS_DIR / f"{m}.pth").exists() for m in MODES}
    return JSONResponse({"modes": MODES, "available": available})


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


async def safe_send(ws: WebSocket, payload: dict) -> bool:
    try:
        await ws.send_text(json.dumps(payload))
        return True
    except Exception:
        return False


def load_agent(mode: str) -> Optional[DQNAgent]:
    weights = MODELS_DIR / f"{mode}.pth"
    if not weights.exists():
        return None
    agent = DQNAgent(AgentConfig(), device=pick_device())
    agent.load(weights)
    agent.online.eval()
    return agent


class ClientControls:
    """Mutable knobs the client can flip during a session."""

    def __init__(self, fps: int = 12):
        self.fps = max(1, min(fps, 240))
        self.paused = False
        self.reset_requested = False
        self.user_action: Optional[int] = None  # for compete mode

    def update_from(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "speed":
            self.fps = max(1, min(int(msg.get("value", 12)), 240))
        elif kind == "pause":
            self.paused = bool(msg.get("value", False))
        elif kind == "reset":
            self.reset_requested = True
        elif kind == "input":
            self.user_action = int(msg.get("action", 0))


async def consume_client_messages(ws: WebSocket, controls: ClientControls):
    """Background task: read client messages and update controls."""
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            controls.update_from(msg)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


# ---------------------------------------------------------------------------
# /ws/learn/{mode} — train live and stream every frame
# ---------------------------------------------------------------------------


@app.websocket("/ws/learn/{mode}")
async def ws_learn(ws: WebSocket, mode: str):
    await ws.accept()
    if mode not in MODES:
        await safe_send(ws, {"type": "error", "message": f"unknown mode: {mode}"})
        await ws.close()
        return

    controls = ClientControls(fps=20)
    reader = asyncio.create_task(consume_client_messages(ws, controls))

    cfg = AgentConfig()
    agent = DQNAgent(cfg, device=pick_device())
    game = SnakeGame(mode=mode)
    episode = 0
    best_score = 0
    total_score = 0

    await safe_send(ws, {
        "type": "info",
        "message": f"training started on {mode} (device={agent.device})",
    })

    try:
        while True:
            episode += 1
            game.reset()
            ep_score = 0
            steps = 0
            while True:
                if controls.reset_requested:
                    controls.reset_requested = False
                    break

                while controls.paused:
                    await asyncio.sleep(0.05)

                states, actions = [], []
                for i in range(game.num_snakes):
                    if game.snakes[i].alive:
                        s = encode_state(game, i)
                        a = agent.select_action(s)
                    else:
                        s, a = None, 0
                    states.append(s)
                    actions.append(a)

                result = game.step(actions)
                steps += 1

                for i in range(game.num_snakes):
                    if states[i] is None:
                        continue
                    s_next = encode_state(game, i)
                    done_i = result.done or not game.snakes[i].alive
                    agent.remember(states[i], actions[i], result.rewards[i], s_next, float(done_i))
                agent.train_step()

                ep_score = max(s.score for s in game.snakes)

                ok = await safe_send(ws, {
                    "type": "frame",
                    "state": game.to_dict(),
                    "stats": {
                        "mode": mode,
                        "episode": episode,
                        "step": steps,
                        "score": ep_score,
                        "best": best_score,
                        "epsilon": round(agent.epsilon, 3),
                        "buffer": len(agent.memory),
                        "phase": "learn",
                    },
                })
                if not ok:
                    return

                # pacing
                await asyncio.sleep(1.0 / max(1, controls.fps))

                if result.done:
                    break

            agent.end_episode()
            best_score = max(best_score, ep_score)
            total_score += ep_score
            await safe_send(ws, {
                "type": "episode_end",
                "stats": {
                    "episode": episode,
                    "score": ep_score,
                    "best": best_score,
                    "mean": round(total_score / episode, 2),
                    "epsilon": round(agent.epsilon, 3),
                },
            })
    except WebSocketDisconnect:
        pass
    finally:
        reader.cancel()


# ---------------------------------------------------------------------------
# /ws/play/{mode} — load weights and stream greedy play
# ---------------------------------------------------------------------------


@app.websocket("/ws/play/{mode}")
async def ws_play(ws: WebSocket, mode: str):
    await ws.accept()
    if mode not in MODES:
        await safe_send(ws, {"type": "error", "message": f"unknown mode: {mode}"})
        await ws.close()
        return
    agent = load_agent(mode)
    if agent is None:
        await safe_send(ws, {
            "type": "error",
            "message": (
                f"no trained model for mode '{mode}'. "
                "run: python -m backend.snake_rl.train --all  to train all four."
            ),
        })
        await ws.close()
        return

    controls = ClientControls(fps=12)
    reader = asyncio.create_task(consume_client_messages(ws, controls))
    game = SnakeGame(mode=mode)
    episode = 0
    best_score = 0

    try:
        while True:
            episode += 1
            game.reset()
            ep_score = 0
            steps = 0
            while True:
                if controls.reset_requested:
                    controls.reset_requested = False
                    break
                while controls.paused:
                    await asyncio.sleep(0.05)

                actions = []
                for i in range(game.num_snakes):
                    if game.snakes[i].alive:
                        s = encode_state(game, i)
                        a = agent.select_action(s, greedy=True)
                    else:
                        a = 0
                    actions.append(a)
                result = game.step(actions)
                steps += 1
                ep_score = max(s.score for s in game.snakes)

                ok = await safe_send(ws, {
                    "type": "frame",
                    "state": game.to_dict(),
                    "stats": {
                        "mode": mode,
                        "episode": episode,
                        "step": steps,
                        "score": ep_score,
                        "best": best_score,
                        "phase": "play",
                    },
                })
                if not ok:
                    return
                await asyncio.sleep(1.0 / max(1, controls.fps))
                if result.done:
                    break

            best_score = max(best_score, ep_score)
            await safe_send(ws, {
                "type": "episode_end",
                "stats": {"episode": episode, "score": ep_score, "best": best_score},
            })
    except WebSocketDisconnect:
        pass
    finally:
        reader.cancel()


# ---------------------------------------------------------------------------
# /ws/compete — user controls snake 0, AI controls snake 1 (duel mode)
# ---------------------------------------------------------------------------


def _direction_to_action(current: Direction, desired: Direction) -> int:
    """Translate an absolute desired direction into a relative turn (0/1/2).

    If the desired direction is opposite the current one, keep going straight.
    """
    if current == desired:
        return 0
    if Direction((int(current) + 1) % 4) == desired:
        return 1  # right turn
    if Direction((int(current) - 1) % 4) == desired:
        return 2  # left turn
    return 0  # opposite => can't reverse, stay straight


@app.websocket("/ws/compete")
async def ws_compete(ws: WebSocket):
    await ws.accept()
    agent = load_agent("duel") or load_agent("walls")
    if agent is None:
        await safe_send(ws, {
            "type": "error",
            "message": "no trained model available. run training first.",
        })
        await ws.close()
        return

    controls = ClientControls(fps=10)
    reader = asyncio.create_task(consume_client_messages(ws, controls))
    game = SnakeGame(mode="duel")
    desired_dir: Direction = game.snakes[0].direction
    episode = 0

    # rebind input handling: in compete mode controls.user_action carries an
    # absolute direction (0=R,1=D,2=L,3=U)
    async def consume(ws: WebSocket):
        nonlocal desired_dir
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                kind = msg.get("type")
                if kind == "input":
                    desired_dir = Direction(int(msg.get("action", int(desired_dir))) % 4)
                else:
                    controls.update_from(msg)
        except WebSocketDisconnect:
            pass

    reader.cancel()
    reader = asyncio.create_task(consume(ws))

    try:
        while True:
            episode += 1
            game.reset()
            desired_dir = game.snakes[0].direction
            steps = 0
            while True:
                if controls.reset_requested:
                    controls.reset_requested = False
                    break
                while controls.paused:
                    await asyncio.sleep(0.05)

                actions = []
                # snake 0 = human
                if game.snakes[0].alive:
                    actions.append(_direction_to_action(game.snakes[0].direction, desired_dir))
                else:
                    actions.append(0)
                # snake 1 = AI
                if game.snakes[1].alive:
                    s = encode_state(game, 1)
                    actions.append(agent.select_action(s, greedy=True))
                else:
                    actions.append(0)

                result = game.step(actions)
                steps += 1

                ok = await safe_send(ws, {
                    "type": "frame",
                    "state": game.to_dict(),
                    "stats": {
                        "mode": "duel",
                        "episode": episode,
                        "step": steps,
                        "score_player": game.snakes[0].score,
                        "score_ai": game.snakes[1].score,
                        "phase": "compete",
                    },
                })
                if not ok:
                    return
                await asyncio.sleep(1.0 / max(1, controls.fps))
                if result.done:
                    break

            outcome = "draw"
            if game.snakes[0].score > game.snakes[1].score:
                outcome = "player"
            elif game.snakes[1].score > game.snakes[0].score:
                outcome = "ai"
            await safe_send(ws, {
                "type": "episode_end",
                "stats": {
                    "episode": episode,
                    "score_player": game.snakes[0].score,
                    "score_ai": game.snakes[1].score,
                    "winner": outcome,
                },
            })
    except WebSocketDisconnect:
        pass
    finally:
        reader.cancel()
