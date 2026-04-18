"""Render before/after GIFs for each game mode.

For each mode we generate two GIFs:
    - <mode>_before.gif  — policy with freshly initialised (untrained) weights
    - <mode>_after.gif   — policy loaded from backend/models/<mode>.pth

Both GIFs are rendered in Python with PIL, using the same visual language
as the browser client (garden grass cells, red-brick frame for the walls
mode, brick-chunk rocks, solid-body snake with a cute round head, apple).

Outputs land in assets/gifs/.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

# path shim so we can run this from the repo root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.snake_rl.agent import AgentConfig, DQNAgent
from backend.snake_rl.game import BLOCK, GRID_H, GRID_W, SnakeGame
from backend.snake_rl.state import encode_state


# ---------- palette (matches frontend/static/app.js) ----------
GRASS_A     = (199, 226, 167)
GRASS_B     = (181, 214, 146)
GRASS_EDGE  = (154, 197, 120)
APPLE       = (209, 59, 59)
APPLE_HI    = (255, 138, 135)
APPLE_STEM  = (91, 58, 31)
APPLE_LEAF  = (79, 154, 63)
BRICK_RED   = (166, 54, 47)
BRICK_RED_HI = (201, 74, 65)
BRICK_RED_LO = (120, 40, 33)
MORTAR      = (214, 200, 176)
SNAKE_A     = (61, 126, 73)
SNAKE_A_HEAD = (98, 169, 110)
SNAKE_A_EDGE = (32, 74, 42)
SNAKE_B     = (121, 87, 181)
SNAKE_B_HEAD = (162, 133, 214)
SNAKE_B_EDGE = (67, 43, 130)
DEAD        = (162, 168, 164)
DEAD_HEAD   = (193, 198, 194)
DEAD_EDGE   = (107, 112, 108)
EYE         = (255, 255, 255)
PUPIL       = (17, 17, 17)

WALL_PAD_CELLS = 1


# ---------- brick helpers ----------

def paint_brick(draw: ImageDraw.ImageDraw, x: float, y: float, w: float, h: float) -> None:
    # vertical red gradient in 3 stops, approximated by striping
    stops = 6
    for i in range(stops):
        t = i / (stops - 1) if stops > 1 else 0
        if t < 0.5:
            a = t / 0.5
            c = (
                int(BRICK_RED_HI[0] + (BRICK_RED[0] - BRICK_RED_HI[0]) * a),
                int(BRICK_RED_HI[1] + (BRICK_RED[1] - BRICK_RED_HI[1]) * a),
                int(BRICK_RED_HI[2] + (BRICK_RED[2] - BRICK_RED_HI[2]) * a),
            )
        else:
            a = (t - 0.5) / 0.5
            c = (
                int(BRICK_RED[0] + (BRICK_RED_LO[0] - BRICK_RED[0]) * a),
                int(BRICK_RED[1] + (BRICK_RED_LO[1] - BRICK_RED[1]) * a),
                int(BRICK_RED[2] + (BRICK_RED_LO[2] - BRICK_RED[2]) * a),
            )
        y0 = y + (h * i) / stops
        y1 = y + (h * (i + 1)) / stops
        draw.rectangle([x, y0, x + w, y1], fill=c)
    # soft bottom shadow
    draw.rectangle([x, y + h - 1, x + w, y + h], fill=(0, 0, 0, 40))


def draw_brick_frame(img: Image.Image, gw: int, gh: int, block: int, pad: int) -> None:
    draw = ImageDraw.Draw(img)
    total_w = gw * block + 2 * pad
    total_h = gh * block + 2 * pad
    draw.rectangle([0, 0, total_w, pad], fill=MORTAR)
    draw.rectangle([0, total_h - pad, total_w, total_h], fill=MORTAR)
    draw.rectangle([0, pad, pad, total_h - pad], fill=MORTAR)
    draw.rectangle([total_w - pad, pad, total_w, total_h - pad], fill=MORTAR)

    bw = block * 0.85
    bh = pad * 0.5

    def paint_strip(x0: int, y0: int, w: int, h: int) -> None:
        strip = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        sd = ImageDraw.Draw(strip)
        rows = int(math.ceil(h / bh)) + 1
        for row in range(rows):
            yy = row * bh
            offset = -bw / 2 if (row % 2) else 0
            cols = int(math.ceil(w / bw)) + 2
            for col in range(-1, cols):
                xx = col * bw + offset
                sd.rectangle([xx + 1, yy + 1, xx + bw - 1, yy + bh - 1], fill=BRICK_RED)
                # simulate gradient via a subtle highlight band
                sd.rectangle([xx + 1, yy + 1, xx + bw - 1, yy + bh * 0.35], fill=BRICK_RED_HI)
                sd.rectangle([xx + 1, yy + bh - 2, xx + bw - 1, yy + bh - 1], fill=BRICK_RED_LO)
        img.paste(strip, (x0, y0), strip)

    paint_strip(0, 0, total_w, pad)
    paint_strip(0, total_h - pad, total_w, pad)
    paint_strip(0, pad, pad, total_h - 2 * pad)
    paint_strip(total_w - pad, pad, pad, total_h - 2 * pad)

    # outer + inner thin borders
    draw.rectangle([0, 0, total_w - 1, total_h - 1], outline=(0, 0, 0, 64))
    draw.rectangle([pad - 1, pad - 1, pad + gw * block, pad + gh * block], outline=(0, 0, 0, 60))


# ---------- garden / food / rocks ----------

def draw_garden(img: Image.Image, gw: int, gh: int, block: int, origin: Tuple[int, int]) -> None:
    draw = ImageDraw.Draw(img)
    ox, oy = origin
    for gx in range(gw):
        for gy in range(gh):
            c = GRASS_A if ((gx + gy) % 2) == 0 else GRASS_B
            draw.rectangle(
                [ox + gx * block, oy + gy * block, ox + (gx + 1) * block, oy + (gy + 1) * block],
                fill=c,
            )
    draw.rectangle(
        [ox, oy, ox + gw * block - 1, oy + gh * block - 1],
        outline=GRASS_EDGE, width=2,
    )


def draw_apple(img: Image.Image, fx: int, fy: int, block: int, origin: Tuple[int, int]) -> None:
    if fx < 0: return
    ox, oy = origin
    draw = ImageDraw.Draw(img)
    cx = ox + fx * block + block / 2
    cy = oy + fy * block + block / 2
    r = block * 0.36
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=APPLE)
    # highlight
    draw.ellipse([cx - r * 0.5, cy - r * 0.5, cx - r * 0.1, cy - r * 0.1], fill=(255, 255, 255, 180))
    # stem
    draw.line([(cx, cy - r), (cx + r * 0.15, cy - r - block * 0.18)], fill=APPLE_STEM, width=2)
    # leaf (small filled triangle-ish)
    draw.polygon(
        [(cx + r * 0.18, cy - r - block * 0.14),
         (cx + r * 0.55, cy - r - block * 0.18),
         (cx + r * 0.38, cy - r - block * 0.02)],
        fill=APPLE_LEAF,
    )


def draw_rocks(img: Image.Image, rocks, block: int, origin: Tuple[int, int]) -> None:
    ox, oy = origin
    draw = ImageDraw.Draw(img)
    for r in rocks:
        rx, ry = r[0], r[1]
        x0, y0 = ox + rx * block + 1, oy + ry * block + 1
        w, h = block - 2, block - 2
        draw.rectangle([x0, y0, x0 + w, y0 + h], fill=MORTAR)
        rows = 2
        bh = h / rows
        bw = w / 2
        for row in range(rows):
            yy = y0 + row * bh
            offset = -bw / 2 if (row % 2) else 0
            for col in range(-1, 3):
                xx = x0 + col * bw + offset
                bx0 = max(x0, xx) + 0.5
                bx1 = min(xx + bw, x0 + w) - 0.5
                if bx1 - bx0 <= 0:
                    continue
                draw.rectangle([bx0, yy + 0.5, bx1, yy + bh - 0.5], fill=BRICK_RED)
                draw.rectangle([bx0, yy + 0.5, bx1, yy + bh * 0.35], fill=BRICK_RED_HI)
                draw.rectangle([bx0, yy + bh - 1.5, bx1, yy + bh - 0.5], fill=BRICK_RED_LO)
        draw.rectangle([x0, y0, x0 + w, y0 + h], outline=(0, 0, 0, 60))


# ---------- snake ----------

def dir_vec(d: int) -> Tuple[int, int]:
    return [(1, 0), (0, 1), (-1, 0), (0, -1)][d]


def draw_snakes(img: Image.Image, game: SnakeGame, block: int, origin: Tuple[int, int]) -> None:
    ox, oy = origin
    for idx, s in enumerate(game.snakes):
        if not s.body:
            continue
        is_a = idx == 0
        dead = not s.alive
        body_col = DEAD if dead else (SNAKE_A if is_a else SNAKE_B)
        head_col = DEAD_HEAD if dead else (SNAKE_A_HEAD if is_a else SNAKE_B_HEAD)
        edge_col = DEAD_EDGE if dead else (SNAKE_A_EDGE if is_a else SNAKE_B_EDGE)

        # convert body to pixel centers
        pts = [(ox + (c[0] + 0.5) * block, oy + (c[1] + 0.5) * block) for c in s.body]
        wrap_x = GRID_W / 2 * block
        wrap_y = GRID_H / 2 * block

        # split at wraps
        sub_paths = []
        sub = []
        for p in pts:
            if not sub:
                sub.append(p)
            else:
                q = sub[-1]
                if abs(p[0] - q[0]) > wrap_x or abs(p[1] - q[1]) > wrap_y:
                    sub_paths.append(sub); sub = [p]
                else:
                    sub.append(p)
        if sub: sub_paths.append(sub)

        d = ImageDraw.Draw(img, "RGBA")
        for sp in sub_paths:
            if len(sp) == 1:
                x, y = sp[0]
                # small dot
                rr = block * 0.41
                d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=edge_col)
                rr = block * 0.36
                d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=body_col)
                continue
            d.line(sp, fill=edge_col, width=int(block * 0.82), joint="curve")
        for sp in sub_paths:
            if len(sp) < 2:
                continue
            d.line(sp, fill=body_col, width=int(block * 0.72), joint="curve")

        # cute head
        draw_cute_head(img, pts[0], s.direction, block, head_col, edge_col, dead)


def draw_cute_head(img: Image.Image, head, direction: int, block: int,
                   fill, edge, dead: bool) -> None:
    hx, hy = head
    fv = dir_vec(direction)
    r = block * 0.50
    cx = hx + fv[0] * block * 0.05
    cy = hy + fv[1] * block * 0.05

    d = ImageDraw.Draw(img, "RGBA")
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill, outline=edge, width=max(1, int(block * 0.08)))

    if dead:
        _draw_dead_eyes(img, (cx, cy), direction, block)
        return

    # big cute eyes
    side_ax = (-fv[1], fv[0])
    fwd, side = 0.18, 0.22
    eye_r = max(3.2, block * 0.21)
    pupil_r = max(1.7, block * 0.11)
    e1 = (cx + fv[0] * block * fwd - side_ax[0] * block * side,
          cy + fv[1] * block * fwd - side_ax[1] * block * side)
    e2 = (cx + fv[0] * block * fwd + side_ax[0] * block * side,
          cy + fv[1] * block * fwd + side_ax[1] * block * side)
    for (ex, ey) in (e1, e2):
        d.ellipse([ex - eye_r, ey - eye_r, ex + eye_r, ey + eye_r], fill=EYE, outline=(10, 20, 15, 140), width=1)
    for (ex, ey) in (e1, e2):
        px = ex + fv[0] * eye_r * 0.35
        py = ey + fv[1] * eye_r * 0.35
        d.ellipse([px - pupil_r, py - pupil_r, px + pupil_r, py + pupil_r], fill=PUPIL)
    # catch-light
    for (ex, ey) in (e1, e2):
        r_cl = max(0.8, pupil_r * 0.55)
        cx_cl = ex - eye_r * 0.3
        cy_cl = ey - eye_r * 0.3
        d.ellipse([cx_cl - r_cl, cy_cl - r_cl, cx_cl + r_cl, cy_cl + r_cl], fill=(255, 255, 255, 240))


def _draw_dead_eyes(img: Image.Image, head, direction, block):
    hx, hy = head
    fv = dir_vec(direction)
    side_ax = (-fv[1], fv[0])
    fwd, side = 0.2, 0.22
    s = block * 0.12
    d = ImageDraw.Draw(img, "RGBA")
    eyes = [
        (hx + fv[0] * block * fwd - side_ax[0] * block * side,
         hy + fv[1] * block * fwd - side_ax[1] * block * side),
        (hx + fv[0] * block * fwd + side_ax[0] * block * side,
         hy + fv[1] * block * fwd + side_ax[1] * block * side),
    ]
    for (x, y) in eyes:
        d.line([(x - s, y - s), (x + s, y + s)], fill=(20, 20, 20, 220), width=2)
        d.line([(x - s, y + s), (x + s, y - s)], fill=(20, 20, 20, 220), width=2)


# ---------- orchestration ----------

def render_frame(game: SnakeGame) -> Image.Image:
    wall_pad = WALL_PAD_CELLS * BLOCK if game.mode == "walls" else 0
    W = GRID_W * BLOCK + 2 * wall_pad
    H = GRID_H * BLOCK + 2 * wall_pad
    img = Image.new("RGB", (W, H), (242, 236, 216))
    if wall_pad:
        draw_brick_frame(img, GRID_W, GRID_H, BLOCK, wall_pad)
    draw_garden(img, GRID_W, GRID_H, BLOCK, (wall_pad, wall_pad))
    draw_rocks(img, game.rocks, BLOCK, (wall_pad, wall_pad))
    draw_apple(img, game.food[0], game.food[1], BLOCK, (wall_pad, wall_pad))
    draw_snakes(img, game, BLOCK, (wall_pad, wall_pad))
    return img


def run_episode_frames(
    mode: str,
    model_path: Optional[Path],
    max_steps: int,
    seed: int = 0,
    device: str = "cpu",
) -> List[Image.Image]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    agent = DQNAgent(AgentConfig(), device=device)
    if model_path is not None and model_path.exists():
        agent.load(model_path)
        agent.online.eval()
        epsilon_override = 0.0
    else:
        # fresh random net => effectively random policy, still slightly biased
        agent.online.eval()
        # force some exploration so untrained snakes aren't stuck in a loop
        epsilon_override = 0.35

    game = SnakeGame(mode=mode, seed=seed)
    frames: List[Image.Image] = [render_frame(game)]
    for _ in range(max_steps):
        actions = []
        for i in range(game.num_snakes):
            if not game.snakes[i].alive:
                actions.append(0); continue
            if epsilon_override > 0 and np.random.rand() < epsilon_override:
                actions.append(int(np.random.randint(0, 3)))
            else:
                with torch.no_grad():
                    s = encode_state(game, i)
                    t = torch.from_numpy(s).float().unsqueeze(0).to(agent.device)
                    q = agent.online(t)
                actions.append(int(q.argmax(dim=-1).item()))
        result = game.step(actions)
        frames.append(render_frame(game))
        if result.done:
            # linger for a few frames on the dead state
            for _ in range(6):
                frames.append(frames[-1].copy())
            break
    return frames


def save_gif(frames: List[Image.Image], out: Path, fps: int = 10, max_frames: int = 220, target_w: int = 420) -> None:
    if not frames: return
    # if there are too many frames, sub-sample uniformly
    if len(frames) > max_frames:
        idxs = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
        frames = [frames[i] for i in idxs]
    scale = target_w / frames[0].width
    target_h = int(frames[0].height * scale)
    # quantize to a shared palette for much smaller file size
    resized = [f.resize((target_w, target_h), Image.LANCZOS) for f in frames]
    # use the first frame's palette for consistency
    palette_image = resized[0].convert("P", palette=Image.ADAPTIVE, colors=96)
    p_frames = [palette_image]
    for f in resized[1:]:
        p_frames.append(f.convert("RGB").quantize(palette=palette_image, dither=Image.FLOYDSTEINBERG))
    out.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(1000 / fps)
    p_frames[0].save(
        out, save_all=True, append_images=p_frames[1:],
        duration=duration_ms, loop=0, optimize=True,
    )


MODES = ["walls", "no_walls", "rocks", "duel"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="assets/gifs")
    parser.add_argument("--max-steps-untrained", type=int, default=90)
    parser.add_argument("--max-steps-trained", type=int, default=420)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    models_dir = ROOT / "backend" / "models"
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for mode in MODES:
        print(f"[{mode}] rendering BEFORE (untrained) …", flush=True)
        frames = run_episode_frames(mode, model_path=None, max_steps=args.max_steps_untrained,
                                    seed=args.seed)
        save_gif(frames, out_dir / f"{mode}_before.gif", fps=args.fps)

        ckpt = models_dir / f"{mode}.pth"
        print(f"[{mode}] rendering AFTER ({ckpt.name}) …", flush=True)
        frames = run_episode_frames(mode, model_path=ckpt, max_steps=args.max_steps_trained,
                                    seed=args.seed)
        save_gif(frames, out_dir / f"{mode}_after.gif", fps=args.fps)

    print("done. GIFs in", out_dir)


if __name__ == "__main__":
    main()
