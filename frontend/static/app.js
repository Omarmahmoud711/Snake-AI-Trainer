/* Snake-RL client: clean garden theme + solid-body snake rendering. */

const COLORS = {
    grassA:     "#c7e2a7",
    grassB:     "#b5d692",
    grassEdge:  "#9ac578",
    apple:      "#d13b3b",
    appleHi:    "#ff8a87",
    appleStem:  "#5b3a1f",
    appleLeaf:  "#4f9a3f",
    rock:       "#8e7a62",
    rockHi:     "#b5a388",
    rockEdge:   "#5c4d3a",
    snakeA:     "#3d7e49",
    snakeAHead: "#62a96e",
    snakeAEdge: "#204a2a",
    snakeB:     "#7957b5",
    snakeBHead: "#a285d6",
    snakeBEdge: "#432b82",
    tongue:     "#d13b3b",
    dead:       "#a2a8a4",
    deadHead:   "#c1c6c2",
    deadEdge:   "#6b706c",
    eye:        "#ffffff",
    pupil:      "#111111",
    cheek:      "#ff9cad",
    // walls mode (red brick border)
    brickRed:   "#a6362f",
    brickRedHi: "#c94a41",
    brickRedLo: "#782821",
    mortar:     "#d6c8b0",
};

const WALL_PAD_CELLS = 1;  // bricks add a 1-cell frame around the play area in "walls" mode

const TICK_MS = 150;          // how long the interpolation eases between server frames

const state = {
    socket: null,
    mode: null,
    intent: null,
    fps: 10,
    paused: false,
    prev: null,
    cur:  null,
    tRecv: 0,
    particles: [],
    prevScores: [0, 0],
    flashKeys: {},
    rafHandle: null,
    overlayText: "",
    // per-snake flags to disable interpolation (on reset / death)
    snapNext: [false, false],
};

const els = {};

function $(sel, root = document) { return root.querySelector(sel); }
function $$(sel, root = document) { return Array.from(root.querySelectorAll(sel)); }

document.addEventListener("DOMContentLoaded", () => {
    els.views = {};
    $$(".view").forEach(v => { els.views[v.dataset.view] = v; });
    els.canvas = $("#board");
    els.ctx = els.canvas.getContext("2d");
    els.overlay = $("#overlay");
    els.hudStats = $("#hud-stats");
    els.modeChip = $("#mode-chip");
    els.phaseChip = $("#phase-chip");
    els.speed = $("#speed");
    els.speedOut = $("#speed-out");
    els.pause = $("#pause");
    els.reset = $("#reset");
    els.back = $("#back");
    els.modesTitle = $("#modes-title");
    els.modeAvail = $("#mode-availability");

    bindMenu();
    bindControls();
    bindKeys();
    window.addEventListener("hashchange", route);
    route();
    startRenderLoop();
});

function bindMenu() {
    $$("[data-go]").forEach(btn => {
        btn.addEventListener("click", () => { window.location.hash = btn.dataset.go; });
    });
    $$(".mode").forEach(btn => {
        btn.addEventListener("click", () => onModePicked(btn.dataset.mode));
    });
}

function bindControls() {
    const onSpeed = () => {
        state.fps = parseInt(els.speed.value, 10);
        els.speedOut.textContent = state.fps;
        sendCtrl({ type: "speed", value: state.fps });
    };
    els.speed.addEventListener("input", onSpeed);
    els.pause.addEventListener("click", () => {
        state.paused = !state.paused;
        els.pause.innerHTML = state.paused
            ? `<svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor" stroke="none"><polygon points="6 3 20 12 6 21"/></svg> Resume`
            : `<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="6" y="5" width="4" height="14"/><rect x="14" y="5" width="4" height="14"/></svg> Pause`;
        sendCtrl({ type: "pause", value: state.paused });
    });
    els.reset.addEventListener("click", () => sendCtrl({ type: "reset" }));
    els.back.addEventListener("click", () => { window.location.hash = "#/"; });
    // initialize fps from slider
    state.fps = parseInt(els.speed.value, 10);
    els.speedOut.textContent = state.fps;
}

function bindKeys() {
    document.addEventListener("keydown", (e) => {
        if (state.intent !== "compete") return;
        const map = {
            ArrowRight: 0, d: 0, D: 0,
            ArrowDown:  1, s: 1, S: 1,
            ArrowLeft:  2, a: 2, A: 2,
            ArrowUp:    3, w: 3, W: 3,
        };
        if (e.key in map) {
            e.preventDefault();
            sendCtrl({ type: "input", action: map[e.key] });
        }
    });
}

/* ---------- routing ---------- */
function setView(name) {
    Object.entries(els.views).forEach(([k, el]) => {
        el.classList.toggle("active", k === name);
    });
}

async function route() {
    closeSocket();
    state.prev = state.cur = null;
    state.particles = [];
    state.snapNext = [true, true];
    const hash = window.location.hash || "#/";
    if (hash === "#/" || hash === "") { setView("menu"); return; }
    if (hash === "#/learn" || hash === "#/play") {
        state.intent = hash === "#/learn" ? "learn" : "play";
        els.modesTitle.textContent = state.intent === "learn" ? "Pick a mode to train" : "Pick a mode to watch";
        await refreshAvailability();
        setView("modes");
        return;
    }
    if (hash === "#/compete") {
        state.intent = "compete";
        state.mode = "duel";
        startGame();
        return;
    }
    if (hash.startsWith("#/game/")) {
        const [intent, mode] = hash.replace("#/game/", "").split("/");
        state.intent = intent;
        state.mode = mode;
        startGame();
        return;
    }
    window.location.hash = "#/";
}

async function refreshAvailability() {
    try {
        const r = await fetch("/api/status");
        const data = await r.json();
        const avail = data.available || {};
        $$(".mode").forEach(b => {
            const ok = state.intent === "learn" ? true : !!avail[b.dataset.mode];
            b.disabled = !ok;
        });
        if (state.intent === "play") {
            const missing = Object.entries(avail).filter(([_, v]) => !v).map(([k]) => k);
            els.modeAvail.textContent = missing.length
                ? `Missing weights for: ${missing.join(", ")}. Train with python -m backend.snake_rl.vec_train.`
                : "All four pre-trained models loaded.";
        } else {
            els.modeAvail.textContent = "Training runs inside your browser session. Close the tab to stop.";
        }
    } catch (e) {
        els.modeAvail.textContent = "Could not load status.";
    }
}

function onModePicked(mode) {
    state.mode = mode;
    window.location.hash = `#/game/${state.intent}/${mode}`;
}

/* ---------- websocket ---------- */
function wsUrl(path) {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${location.host}${path}`;
}
function closeSocket() {
    if (state.socket) { try { state.socket.close(); } catch (_) {} state.socket = null; }
}
function sendCtrl(payload) {
    if (state.socket && state.socket.readyState === WebSocket.OPEN) {
        state.socket.send(JSON.stringify(payload));
    }
}

function startGame() {
    setView("game");
    state.prev = state.cur = null;
    state.particles = [];
    state.prevScores = [0, 0];
    state.snapNext = [true, true];
    state.overlayText = "";
    els.overlay.classList.remove("show");

    els.modeChip.textContent = (state.mode || "").replace("_", " ").toUpperCase() || "—";
    const phaseLabel = state.intent === "learn" ? "Learning"
                     : state.intent === "play"  ? "Playing"
                     :                             "Vs AI";
    els.phaseChip.textContent = phaseLabel;

    let path;
    if (state.intent === "learn") path = `/ws/learn/${state.mode}`;
    else if (state.intent === "play") path = `/ws/play/${state.mode}`;
    else path = `/ws/compete`;

    state.paused = false;

    state.socket = new WebSocket(wsUrl(path));
    state.socket.onopen = () => { sendCtrl({ type: "speed", value: state.fps }); };
    state.socket.onmessage = (e) => {
        let msg;
        try { msg = JSON.parse(e.data); } catch (_) { return; }
        handleMessage(msg);
    };
    state.socket.onclose = () => renderHud([{ k: "status", v: "disconnected" }]);
    state.socket.onerror = () => renderHud([{ k: "status", v: "error" }]);
}

function handleMessage(msg) {
    if (msg.type === "frame") {
        const ns = msg.state.snakes || [];
        ns.forEach((s, i) => {
            const prev = state.prevScores[i] ?? 0;
            if (s.score > prev) {
                spawnFoodParticles(s.body[0]);
                state.flashKeys.score = Date.now();
            }
            state.prevScores[i] = s.score;
            // decide if we should snap this snake (no interpolation) on the next draw
            // - if snake was dead last frame, don't animate it now
            // - if snake body length jumped by > 1 (reset), snap
            // - if snake head distance is huge (reset), snap
            if (state.cur && state.cur.snakes && state.cur.snakes[i]) {
                const pcur = state.cur.snakes[i];
                const bigLenChange = Math.abs((pcur.body || []).length - (s.body || []).length) > 1;
                const pHead = (pcur.body && pcur.body[0]) || null;
                const cHead = (s.body && s.body[0]) || null;
                const bigHeadJump = pHead && cHead && (
                    (Math.abs(pHead.x - cHead.x) > 2 && Math.abs(pHead.x - cHead.x) < msg.state.grid.w - 2) ||
                    (Math.abs(pHead.y - cHead.y) > 2 && Math.abs(pHead.y - cHead.y) < msg.state.grid.h - 2)
                );
                if (!pcur.alive || bigLenChange || bigHeadJump) state.snapNext[i] = true;
            }
        });
        state.prev = state.cur || msg.state;
        state.cur = msg.state;
        state.tRecv = performance.now();
        renderStats(msg.stats);
        // overlay: only show when everything is dead and episode isn't about to reset
        // (server restarts automatically so this only shows briefly)
        if (!anySnakeAlive(msg.state)) {
            state.overlayText = msg.stats && msg.stats.winner
                ? (msg.stats.winner === "player" ? "You win!" : msg.stats.winner === "ai" ? "AI wins!" : "Draw")
                : "Game over";
            els.overlay.textContent = state.overlayText;
            els.overlay.classList.add("show");
        } else {
            els.overlay.classList.remove("show");
        }
    } else if (msg.type === "episode_end") {
        // next frame will be from a fresh reset -> force snap for both snakes
        state.snapNext = [true, true];
        renderStats({ ...((state.cur && {}) || {}), ...msg.stats, _ep_end: true });
    } else if (msg.type === "info") {
        renderHud([{ k: "info", v: msg.message }]);
    } else if (msg.type === "error") {
        renderHud([{ k: "error", v: msg.message }]);
    }
}

function anySnakeAlive(s) {
    return (s.snakes || []).some(x => x.alive);
}

/* ---------- render loop ---------- */
function startRenderLoop() {
    const tick = (ts) => {
        draw(ts);
        state.rafHandle = requestAnimationFrame(tick);
    };
    state.rafHandle = requestAnimationFrame(tick);
}

function draw(ts) {
    if (!state.cur) return;
    const st = state.cur;
    const block = st.grid.block;
    const wallPad = (st.mode === "walls") ? WALL_PAD_CELLS * block : 0;
    const wPx = st.grid.w * block + 2 * wallPad;
    const hPx = st.grid.h * block + 2 * wallPad;
    if (els.canvas.width !== wPx || els.canvas.height !== hPx) {
        els.canvas.width = wPx;
        els.canvas.height = hPx;
    }
    const ctx = els.ctx;
    ctx.clearRect(0, 0, wPx, hPx);

    const age = ts - state.tRecv;
    const tRaw = Math.max(0, Math.min(1, age / TICK_MS));
    // easeOutQuad for a bit of snap at the end of each tick
    const t = 1 - (1 - tRaw) * (1 - tRaw);

    // paint brick frame first (under the garden area)
    if (wallPad > 0) drawBrickFrame(ctx, st.grid.w, st.grid.h, block, wallPad);

    ctx.save();
    ctx.translate(wallPad, wallPad);
    drawGarden(ctx, st.grid.w, st.grid.h, block);
    drawRocks(ctx, st, block, ts);
    drawFood(ctx, st, block, ts);
    drawSnakes(ctx, state.prev, state.cur, block, t);
    updateAndDrawParticles(ctx, block, ts);
    ctx.restore();
}

/* ---------- garden / food / rocks ---------- */
function drawGarden(ctx, gw, gh, block) {
    for (let gx = 0; gx < gw; gx++) {
        for (let gy = 0; gy < gh; gy++) {
            const base = ((gx + gy) % 2) === 0 ? COLORS.grassA : COLORS.grassB;
            ctx.fillStyle = base;
            ctx.fillRect(gx * block, gy * block, block, block);
        }
    }
    ctx.strokeStyle = COLORS.grassEdge;
    ctx.lineWidth = 2;
    ctx.strokeRect(1, 1, gw * block - 2, gh * block - 2);
}

/* ---------- brick wall frame (walls mode) ---------- */
function drawBrickFrame(ctx, gw, gh, block, pad) {
    const totalW = gw * block + 2 * pad;
    const totalH = gh * block + 2 * pad;
    // brick size ~ half a cell wide, 0.45 cell tall; running-bond pattern
    const bw = block * 0.85;
    const bh = pad * 0.5;
    // paint mortar background for the frame, then cut out the inner play area
    ctx.save();
    ctx.fillStyle = COLORS.mortar;
    ctx.fillRect(0, 0, totalW, pad);                    // top
    ctx.fillRect(0, totalH - pad, totalW, pad);         // bottom
    ctx.fillRect(0, pad, pad, totalH - 2 * pad);        // left
    ctx.fillRect(totalW - pad, pad, pad, totalH - 2 * pad); // right

    // draw bricks in each of the 4 frame bands
    const paintStrip = (x, y, w, h) => {
        // clip to the strip so bricks overflowing get cut
        ctx.save();
        ctx.beginPath();
        ctx.rect(x, y, w, h);
        ctx.clip();
        const rows = Math.ceil(h / bh) + 1;
        for (let row = 0; row < rows; row++) {
            const yy = y + row * bh;
            const offset = (row % 2) ? -bw / 2 : 0;
            const cols = Math.ceil(w / bw) + 2;
            for (let col = -1; col < cols; col++) {
                const xx = x + col * bw + offset;
                paintBrick(ctx, xx + 1, yy + 1, bw - 2, bh - 2);
            }
        }
        ctx.restore();
    };
    paintStrip(0, 0, totalW, pad);
    paintStrip(0, totalH - pad, totalW, pad);
    paintStrip(0, pad, pad, totalH - 2 * pad);
    paintStrip(totalW - pad, pad, pad, totalH - 2 * pad);

    // subtle darker border around the whole thing
    ctx.strokeStyle = "rgba(0,0,0,0.25)";
    ctx.lineWidth = 1;
    ctx.strokeRect(0.5, 0.5, totalW - 1, totalH - 1);
    // inner edge shadow around the playable grass
    ctx.strokeStyle = "rgba(0,0,0,0.18)";
    ctx.lineWidth = 1;
    ctx.strokeRect(pad - 0.5, pad - 0.5, gw * block + 1, gh * block + 1);
    ctx.restore();
}

function paintBrick(ctx, x, y, w, h) {
    const grd = ctx.createLinearGradient(x, y, x, y + h);
    grd.addColorStop(0, COLORS.brickRedHi);
    grd.addColorStop(0.6, COLORS.brickRed);
    grd.addColorStop(1, COLORS.brickRedLo);
    ctx.fillStyle = grd;
    ctx.fillRect(x, y, w, h);
    // subtle darker line on the bottom for depth
    ctx.fillStyle = "rgba(0,0,0,0.15)";
    ctx.fillRect(x, y + h - 1, w, 1);
}

function drawFood(ctx, st, block, ts) {
    if (!st.food || st.food.x < 0) return;
    const x = st.food.x * block, y = st.food.y * block;
    const cx = x + block / 2, cy = y + block / 2;
    const bob = Math.sin(ts / 340) * 1.0;
    const r = block * 0.36;

    ctx.save();
    ctx.translate(cx, cy + bob);
    const grad = ctx.createRadialGradient(-r * 0.3, -r * 0.3, r * 0.1, 0, 0, r);
    grad.addColorStop(0, "#ff6b63");
    grad.addColorStop(1, COLORS.apple);
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(0, 0, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "rgba(255,255,255,0.65)";
    ctx.beginPath();
    ctx.arc(-r * 0.32, -r * 0.32, r * 0.22, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = COLORS.appleStem;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, -r + 1);
    ctx.lineTo(r * 0.15, -r - block * 0.18);
    ctx.stroke();
    ctx.fillStyle = COLORS.appleLeaf;
    ctx.beginPath();
    ctx.ellipse(r * 0.3, -r - block * 0.08, block * 0.14, block * 0.07, -0.6, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
}

function drawRocks(ctx, st, block, ts) {
    (st.rocks || []).forEach(r => {
        const x = r.x * block, y = r.y * block;
        const ttl = r.ttl ?? 120;
        const fadeIn  = Math.min(1, (220 - ttl) / 20);
        const fadeOut = Math.min(1, ttl / 40);
        const alpha = Math.max(0.25, Math.min(1, Math.min(fadeIn, fadeOut)));
        ctx.save();
        ctx.globalAlpha = alpha;
        // one rock cell = a small chunk of red brick wall
        const pad = 1;
        const x0 = x + pad, y0 = y + pad;
        const w  = block - 2 * pad, h = block - 2 * pad;
        // mortar base
        ctx.fillStyle = COLORS.mortar;
        ctx.fillRect(x0, y0, w, h);
        // two brick rows, running-bond
        const rows = 2;
        const bh = h / rows;
        const bw = w / 2;       // two bricks per row (offset on odd rows)
        for (let row = 0; row < rows; row++) {
            const yy = y0 + row * bh;
            const offset = (row % 2) ? -bw / 2 : 0;
            for (let col = -1; col <= 2; col++) {
                const xx = x0 + col * bw + offset;
                const bx = Math.max(x0, xx) + 0.5;
                const by = yy + 0.5;
                const bW = Math.min(xx + bw, x0 + w) - Math.max(xx, x0) - 1;
                const bH = bh - 1;
                if (bW <= 0) continue;
                paintBrick(ctx, bx, by, bW, bH);
            }
        }
        // subtle outer border
        ctx.strokeStyle = "rgba(0,0,0,0.25)";
        ctx.lineWidth = 1;
        ctx.strokeRect(x0 + 0.5, y0 + 0.5, w - 1, h - 1);
        ctx.restore();
    });
}

/* ---------- snakes: one continuous body + cute head ---------- */
function drawSnakes(ctx, prev, cur, block, t) {
    if (!cur || !cur.snakes) return;
    const prevSnakes = (prev && prev.snakes) || cur.snakes;
    const gw = cur.grid.w, gh = cur.grid.h;

    cur.snakes.forEach((s, idx) => {
        if (!s.body || s.body.length === 0) return;
        const ps = prevSnakes[idx] || s;
        const wasAlive = !!(ps && ps.alive);
        const snap = state.snapNext[idx] || !s.alive || !wasAlive;
        const isA = idx === 0;
        const dead = !s.alive;

        const body = dead ? COLORS.dead     : (isA ? COLORS.snakeA     : COLORS.snakeB);
        const head = dead ? COLORS.deadHead : (isA ? COLORS.snakeAHead : COLORS.snakeBHead);
        const edge = dead ? COLORS.deadEdge : (isA ? COLORS.snakeAEdge : COLORS.snakeBEdge);

        // Interpolate EVERY segment from its previous position (same index) to its
        // current position. This makes the whole snake slide forward smoothly —
        // fixing the prior bug where only the head appeared to animate.
        const pts = [];
        for (let i = 0; i < s.body.length; i++) {
            const c = s.body[i];
            let fromX, fromY;
            if (snap) {
                fromX = c.x;
                fromY = c.y;
            } else {
                const p = (ps.body && ps.body[i]) || c;
                fromX = p.x;
                fromY = p.y;
                if (Math.abs(c.x - fromX) > gw / 2) fromX = c.x;
                if (Math.abs(c.y - fromY) > gh / 2) fromY = c.y;
            }
            const lx = fromX + (c.x - fromX) * t;
            const ly = fromY + (c.y - fromY) * t;
            pts.push({ x: (lx + 0.5) * block, y: (ly + 0.5) * block });
        }

        // Split the polyline at wrap boundaries so we don't draw a line across the board
        const subPaths = [];
        const wrapThreshX = (gw / 2) * block;
        const wrapThreshY = (gh / 2) * block;
        let sub = [];
        for (let i = 0; i < pts.length; i++) {
            if (sub.length === 0) { sub.push(pts[i]); continue; }
            const p = pts[i], q = sub[sub.length - 1];
            if (Math.abs(p.x - q.x) > wrapThreshX || Math.abs(p.y - q.y) > wrapThreshY) {
                subPaths.push(sub); sub = [p];
            } else {
                sub.push(p);
            }
        }
        if (sub.length) subPaths.push(sub);

        // 1) thin soft outline
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.strokeStyle = edge;
        ctx.lineWidth = block * 0.82;
        subPaths.forEach(sp => strokePoly(ctx, sp));

        // 2) body fill (no harsh stripe)
        ctx.strokeStyle = body;
        ctx.lineWidth = block * 0.72;
        subPaths.forEach(sp => strokePoly(ctx, sp));

        // 3) cute head drawn on top of body
        drawCuteHead(ctx, pts[0], s.direction, block, head, edge, dead);
    });
}

function strokePoly(ctx, pts) {
    if (!pts || pts.length === 0) return;
    ctx.beginPath();
    if (pts.length === 1) {
        ctx.moveTo(pts[0].x, pts[0].y);
        ctx.lineTo(pts[0].x + 0.01, pts[0].y + 0.01);
    } else {
        ctx.moveTo(pts[0].x, pts[0].y);
        for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
    }
    ctx.stroke();
}

function drawCuteHead(ctx, head, dir, block, fill, edge, dead) {
    const { x: hx, y: hy } = head;
    const r = block * 0.50;  // larger head so bigger eyes fit
    // slight bump forward so the head feels distinct from the body
    const fv = dirVec(dir);
    const cx = hx + fv.x * block * 0.05;
    const cy = hy + fv.y * block * 0.05;

    // head blob
    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fillStyle = fill;
    ctx.fill();
    ctx.lineWidth = block * 0.08;
    ctx.strokeStyle = edge;
    ctx.stroke();
    ctx.restore();

    if (dead) {
        drawDeadEyes(ctx, { x: cx, y: cy }, dir, block);
        return;
    }

    // big cute eyes — placed forward, wide apart; fits inside the slightly larger head
    const sideAx = { x: -fv.y, y: fv.x };
    const fwd = 0.18;                            // forward offset toward facing direction
    const side = 0.22;                           // sideways offset
    const eyeR   = Math.max(3.2, block * 0.21);  // much bigger
    const pupilR = Math.max(1.7, block * 0.11);

    const e1x = cx + fv.x * block * fwd - sideAx.x * block * side;
    const e1y = cy + fv.y * block * fwd - sideAx.y * block * side;
    const e2x = cx + fv.x * block * fwd + sideAx.x * block * side;
    const e2y = cy + fv.y * block * fwd + sideAx.y * block * side;

    // eye whites with a thin dark outline
    ctx.save();
    ctx.fillStyle = COLORS.eye;
    ctx.strokeStyle = "rgba(10,20,15,0.55)";
    ctx.lineWidth = 1;
    [[e1x, e1y], [e2x, e2y]].forEach(([x, y]) => {
        ctx.beginPath();
        ctx.arc(x, y, eyeR, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
    });
    // pupils — look forward
    ctx.fillStyle = COLORS.pupil;
    [[e1x, e1y], [e2x, e2y]].forEach(([x, y]) => {
        ctx.beginPath();
        ctx.arc(x + fv.x * eyeR * 0.35, y + fv.y * eyeR * 0.35, pupilR, 0, Math.PI * 2);
        ctx.fill();
    });
    // little catch-light in each eye for cuteness
    ctx.fillStyle = "rgba(255,255,255,0.95)";
    [[e1x, e1y], [e2x, e2y]].forEach(([x, y]) => {
        ctx.beginPath();
        ctx.arc(x - eyeR * 0.3, y - eyeR * 0.3, Math.max(0.8, pupilR * 0.55), 0, Math.PI * 2);
        ctx.fill();
    });
    ctx.restore();
}

function drawDeadEyes(ctx, head, dir, block) {
    const { x: hx, y: hy } = head;
    const fv = dirVec(dir);
    const sideAx = { x: -fv.y, y: fv.x };
    const fwd = 0.2, side = 0.22;
    const s = block * 0.12;
    ctx.save();
    ctx.strokeStyle = "rgba(20,20,20,0.85)";
    ctx.lineWidth = 2;
    ctx.lineCap = "round";
    const eyes = [
        [hx + fv.x * block * fwd - sideAx.x * block * side, hy + fv.y * block * fwd - sideAx.y * block * side],
        [hx + fv.x * block * fwd + sideAx.x * block * side, hy + fv.y * block * fwd + sideAx.y * block * side],
    ];
    eyes.forEach(([x, y]) => {
        ctx.beginPath(); ctx.moveTo(x - s, y - s); ctx.lineTo(x + s, y + s);
        ctx.moveTo(x - s, y + s); ctx.lineTo(x + s, y - s);
        ctx.stroke();
    });
    ctx.restore();
}

function dirVec(dir) {
    // 0=R, 1=D, 2=L, 3=U
    if (dir === 0) return { x: 1, y: 0 };
    if (dir === 1) return { x: 0, y: 1 };
    if (dir === 2) return { x: -1, y: 0 };
    return { x: 0, y: -1 };
}

/* ---------- particles on food eaten ---------- */
function spawnFoodParticles(cell) {
    if (!cell) return;
    for (let i = 0; i < 14; i++) {
        const ang = (i / 14) * Math.PI * 2 + Math.random() * 0.3;
        const speed = 0.6 + Math.random() * 1.4;
        state.particles.push({
            x: cell.x + 0.5,
            y: cell.y + 0.5,
            vx: Math.cos(ang) * speed,
            vy: Math.sin(ang) * speed,
            life: 0,
            max: 360 + Math.random() * 220,
            color: Math.random() < 0.5 ? COLORS.apple : COLORS.appleHi,
        });
    }
}

let lastPtTs = 0;
function updateAndDrawParticles(ctx, block, ts) {
    const dt = lastPtTs ? ts - lastPtTs : 16;
    lastPtTs = ts;
    const alive = [];
    for (const p of state.particles) {
        p.life += dt;
        if (p.life > p.max) continue;
        p.x += p.vx * dt * 0.008;
        p.y += p.vy * dt * 0.008;
        p.vy += 0.02;
        const a = 1 - p.life / p.max;
        ctx.save();
        ctx.globalAlpha = a;
        ctx.fillStyle = p.color;
        ctx.beginPath();
        ctx.arc(p.x * block, p.y * block, Math.max(1.5, block * 0.08) * (0.6 + 0.4 * a), 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
        alive.push(p);
    }
    state.particles = alive;
}

/* ---------- HUD ---------- */
function renderStats(stats) {
    if (!stats) return;
    const rows = [];
    const push = (k, v, flashOn) => rows.push({ k, v, flash: flashOn });
    const fresh = state.flashKeys.score && Date.now() - state.flashKeys.score < 450;
    if ("episode" in stats) push("episode", stats.episode);
    if ("step" in stats) push("step", stats.step);
    if ("score" in stats) push("score", stats.score, fresh);
    if ("score_player" in stats) push("you", stats.score_player, fresh);
    if ("score_ai" in stats) push("AI", stats.score_ai);
    if ("best" in stats) push("best", stats.best);
    if ("mean" in stats) push("mean", stats.mean);
    if ("epsilon" in stats) push("epsilon", stats.epsilon);
    if ("buffer" in stats) push("buffer", stats.buffer);
    if (stats.winner) push("winner", stats.winner);
    renderHud(rows);
}

function renderHud(rows) {
    els.hudStats.innerHTML = rows
        .map(r => `<div class="k">${r.k}</div><div class="v${r.flash ? " flash" : ""}">${r.v}</div>`)
        .join("");
}

