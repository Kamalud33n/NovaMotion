(function () {
const G_HORIZON = 100, G_PLAYER_Y = 372, G_NEAR_HW = 300, G_FOCAL = 10;
const G_MAXD = 50, G_REACT_D = 30, G_DFAR = 2000;
const G_LANE_W = 2 * G_NEAR_HW / 3, G_BEAM_TOP = 125, G_BEAM_T = 35;
const G_SMOOTH_X = 10;
const G_JUDGE_GRACE_S = 0.25;   // an obstacle is judged on the player's last 0.25 s, not one instant (network/status jitter)
const G_TIME_SCALE = 3.0;   // game-time seconds per real second (= the old fixed dt 0.05 at 60 fps)
const G_STRIDE = 5, G_STRIDE_SPEED = 20, G_MAX_Q = 15;
const G_SPAWN_MIN = 32, G_SPAWN_MAX = 42;
const gScale = d => G_FOCAL / (G_FOCAL + d);
const gSy = d => G_HORIZON + (G_PLAYER_Y - G_HORIZON) * gScale(d);
const gLx = (lane, d) => GW / 2 + (lane - 1) * G_LANE_W * gScale(d);

let gameLastSteps = 0;
let GG = null;

const gBuildings = []; let gTOT = 0;
(function () {
    let x = 0;
    while (x < GW * 2) {
        const w = 30 + Math.random() * 40, h = 15 + Math.random() * 35;
        const windows = [];
        for (let wy = 4; wy < h - 4; wy += 7) {
            for (let wx = 4; wx < w - 4; wx += 8) {
                if (Math.random() < 0.35) windows.push({ x: wx, y: wy });
            }
        }
        gBuildings.push({ x, w, h, windows }); x += w + 6 + Math.random() * 12;
    }
    gTOT = x;
})();

const gStars = [];
for (let i = 0; i < 40; i++) {
    gStars.push({ x: Math.random() * GW, y: Math.random() * (G_HORIZON - 10), r: 0.6 + Math.random() * 1.4, tw: Math.random() * 6.28 });
}
const gMountains = []; let gMTOT = 0;
(function () {
    let x = 0;
    while (x < GW * 2) {
        const w = 80 + Math.random() * 100, h = 25 + Math.random() * 30;
        gMountains.push({ x, w, h }); x += w * 0.7;
    }
    gMTOT = x;
})();

function newGameCanvasState() {
    GG = {
        score: 0, lives: 3, obstacles: [], spawnD: 8, lastTypes: [],
        cleared: 0, hits: 0, rt: 0, hist: [],   // obstacle tally + the player's last few hundred ms
        moveQ: 0, roadOff: 0, bg: 0, shake: 0, popups: [], sparks: [], idleT: 0, t: 0,
        player: { lane: 1, x: gLx(1, 0), state: 'RUN', t: 0, flash: 0 },
    };
}

function spawnGameObstacle() {
    let type = Math.random() < 0.55 ? 'BLOCK' : 'CROUCH';
    const L = GG.lastTypes;
    if (L.length >= 2 && L[L.length - 1] === type && L[L.length - 2] === type)
        type = type === 'BLOCK' ? 'CROUCH' : 'BLOCK';
    L.push(type); if (L.length > 3) L.shift();
    let lane = 1;
    if (type === 'BLOCK') lane = Math.random() < 0.75 ? GG.player.lane : Math.floor(Math.random() * 3);
    GG.obstacles.push({ type, lane, d: G_MAXD, resolved: false });
}

function spawnSparks(x, y, color, n) {
    for (let i = 0; i < n; i++) {
        const ang = Math.random() * Math.PI * 2, spd = 40 + Math.random() * 90;
        GG.sparks.push({ x, y, vx: Math.cos(ang) * spd, vy: Math.sin(ang) * spd - 30, life: 0.5 + Math.random() * 0.3, color });
    }
}

function updateGamePlayer(realDt) {
    const p = GG.player, s = gameLiveStatus;
    const xn = Math.max(-1, Math.min(1, Number(s.x_norm) || 0));
    const tx = GW / 2 + xn * G_LANE_W;
    if (s.lane !== undefined) p.lane = s.lane;
    // time-based easing (== the old 0.5 per frame at 60 fps), so it feels the same at any refresh rate / frame time
    p.x += (tx - p.x) * (1 - Math.pow(1 - Math.min(1, 0.05 * G_SMOOTH_X), realDt * 60));
    p.state = s.crouching ? 'CROUCH' : 'RUN';
    if (p.flash > 0) p.flash -= 3 * realDt;
}

function updateGameCanvas(realDt) {
    // Real elapsed time (not a fixed step per animation frame), so the game runs at the same
    // speed on a 30 Hz / 60 Hz / 144 Hz screen and a slow frame doesn't slow the world down.
    const dt = realDt * G_TIME_SCALE;
    GG.t += dt;
    updateGamePlayer(realDt);
    // Remember where the player was / whether they were crouching for the last ~0.4 s
    // (real time), so an obstacle can be judged fairly - see below.
    GG.rt += realDt;
    GG.hist.push({ t: GG.rt, x: GG.player.x, crouch: GG.player.state === 'CROUCH' });
    while (GG.hist.length && GG.rt - GG.hist[0].t > 0.4) GG.hist.shift();
    GG.popups.forEach(o => { o.life -= dt; o.y -= 40 * dt; });
    GG.popups = GG.popups.filter(o => o.life > 0);
    GG.sparks.forEach(o => { o.life -= dt; o.x += o.vx * dt; o.y += o.vy * dt; o.vy += 160 * dt; });
    GG.sparks = GG.sparks.filter(o => o.life > 0);
    if (GG.shake > 0) GG.shake -= dt;

    // world advances by however many steps the server counted since the last poll
    const steps = gameLiveStatus.steps || 0;
    const stepDelta = Math.max(0, steps - gameLastSteps);
    gameLastSteps = steps;
    if (stepDelta > 0) GG.moveQ = Math.min(G_MAX_Q, GG.moveQ + stepDelta * G_STRIDE);

    let adv = 0;
    if (GG.moveQ > 0) { adv = Math.min(GG.moveQ, G_STRIDE_SPEED * dt); GG.moveQ -= adv; }
    const nearest = GG.obstacles.filter(o => !o.resolved).sort((a, b) => a.d - b.d)[0];
    if (nearest && nearest.type === 'CROUCH' && nearest.d <= G_REACT_D && GG.player.state === 'CROUCH')
        adv = Math.max(adv, G_STRIDE_SPEED * dt);
    GG.idleT = adv > 0 ? 0 : GG.idleT + dt;
    GG.roadOff += adv; GG.bg += adv * 0.6;

    GG.spawnD -= adv;
    if (GG.spawnD <= 0) { spawnGameObstacle(); GG.spawnD = G_SPAWN_MIN + Math.random() * (G_SPAWN_MAX - G_SPAWN_MIN); }

    const p = GG.player;
    for (const o of GG.obstacles) {
        o.d -= adv;
        if (!o.resolved && o.d <= 0.8) {
            o.resolved = true;
            // Judge the player's last G_JUDGE_GRACE_S seconds instead of a single instant.
            // The status reaches the page ~10-20x/s (more on a slow connection), so one
            // late/flickering sample used to turn a correct dodge/crouch into an OUCH.
            // BLOCK: a hit only if the player stayed in the obstacle's lane the whole window.
            // CROUCH: a hit only if the player was not crouching at any point in the window.
            const recent = GG.hist.filter(h => GG.rt - h.t <= G_JUDGE_GRACE_S);
            let hit = false;
            if (o.type === 'BLOCK') {
                const inLane = x => Math.abs(x - gLx(o.lane, 0)) < G_LANE_W * 0.55;
                hit = recent.length ? recent.every(h => inLane(h.x)) : inLane(p.x);
            } else {
                hit = recent.length ? !recent.some(h => h.crouch) : p.state !== 'CROUCH';
            }
            if (hit) GG.hits++; else GG.cleared++;
            if (hit) {
                GG.lives--; p.flash = 0.5; GG.shake = 0.25;
                GG.popups.push({ x: p.x - 10, y: G_PLAYER_Y - 120, text: 'OUCH', color: '#ff5a5a', life: 0.8 });
                spawnSparks(p.x, G_PLAYER_Y - 60, '#ff5a5a', 10);
                if (GG.lives <= 0) GG.lives = 3; // keep the session going — a miss never ends therapy
            } else {
                GG.score += 10;
                GG.popups.push({ x: p.x - 10, y: G_PLAYER_Y - 120, text: '+10', color: '#5fdc8a', life: 0.8 });
                spawnSparks(p.x, G_PLAYER_Y - 60, '#5fdc8a', 8);
            }
        }
    }
    GG.obstacles = GG.obstacles.filter(o => o.d > -6);
}

function gBox(x, y, w, h, c1, c2) {
    const grad = gctx.createLinearGradient(x, y, x, y + h);
    grad.addColorStop(0, c2); grad.addColorStop(0.3, c1); grad.addColorStop(1, c1);
    gctx.fillStyle = grad; gctx.fillRect(x, y, w, h);
    gctx.strokeStyle = 'rgba(0,0,0,.45)'; gctx.lineWidth = 2; gctx.strokeRect(x, y, w, h);
}
function gLabel(t, x, y, size) {
    gctx.fillStyle = '#fff'; gctx.font = 'bold ' + Math.max(7, size) + 'px Arial';
    gctx.textAlign = 'center'; gctx.fillText(t, x, y);
}
function drawGameObstacle(o) {
    const s = gScale(o.d), y = gSy(o.d), cx = GW / 2, fullW = 3 * G_LANE_W * s * 0.96;
    const near = o.d < G_REACT_D * 1.3;
    if (o.type === 'BLOCK') {
        const x = gLx(o.lane, o.d), w = 140 * s, h = 100 * s;
        if (near) {
            gctx.save(); gctx.shadowColor = '#ff6a6a'; gctx.shadowBlur = 18 * s;
            gBox(x - w / 2, y - h, w, h, '#dc4646', '#ff9a9a'); gctx.restore();
        } else {
            gBox(x - w / 2, y - h, w, h, '#dc4646', '#f08282');
        }
        gLabel('✕', x, y - h * 0.32, 40 * s);
    } else {
        const top = G_BEAM_TOP * s, th = G_BEAM_T * s, pw = Math.max(2, 10 * s);
        gctx.fillStyle = '#5a3480';
        gctx.fillRect(cx - fullW / 2, y - top, pw, top);
        gctx.fillRect(cx + fullW / 2 - pw, y - top, pw, top);
        if (near) { gctx.save(); gctx.shadowColor = '#c895f5'; gctx.shadowBlur = 16 * s; }
        gBox(cx - fullW / 2, y - top, fullW, th, '#aa64dc', '#e6c8ff');
        if (near) gctx.restore();
        gLabel('DUCK', cx, y - top + th * 0.72, 20 * s);
    }
}
function drawGamePlayer() {
    const p = GG.player, x = p.x, base = G_PLAYER_Y, crouch = p.state === 'CROUCH';
    const flashing = p.flash > 0 && Math.floor(p.flash * 20) % 2 === 0;
    const bodyGrad = gctx.createLinearGradient(x - 25, base - 90, x + 25, base);
    if (flashing) { bodyGrad.addColorStop(0, '#ff8a8a'); bodyGrad.addColorStop(1, '#dc4646'); }
    else if (crouch) { bodyGrad.addColorStop(0, '#ffb26b'); bodyGrad.addColorStop(1, '#e07a2c'); }
    else { bodyGrad.addColorStop(0, '#6fa2ff'); bodyGrad.addColorStop(1, '#3f6fd6'); }

    gctx.fillStyle = 'rgba(0,0,0,.35)'; gctx.beginPath();
    gctx.ellipse(x, G_PLAYER_Y + 2, 30, 8, 0, 0, 7); gctx.fill();

    const legH = crouch ? 14 : 26, torsoH = crouch ? 38 : 62;
    let fl = gameLiveStatus.step_foot || '';
    const liftL = (fl === 'L' && !crouch) ? 12 : 0, liftR = (fl === 'R' && !crouch) ? 12 : 0;

    if (!crouch && (liftL || liftR) && GG.moveQ > 0) {
        gctx.strokeStyle = 'rgba(255,255,255,.35)'; gctx.lineWidth = 3;
        for (let i = 0; i < 3; i++) {
            gctx.beginPath();
            gctx.moveTo(x - 30 - i * 10, base - 6 - i * 4);
            gctx.lineTo(x - 44 - i * 10, base - 6 - i * 4);
            gctx.stroke();
        }
    }

    gctx.fillStyle = '#151520';
    gctx.fillRect(x - 20, base - legH, 12, legH - liftL);
    gctx.fillRect(x + 8, base - legH, 12, legH - liftR);
    gctx.fillStyle = bodyGrad; gctx.beginPath(); gctx.roundRect(x - 25, base - legH - torsoH, 50, torsoH, 9); gctx.fill();
    gctx.strokeStyle = 'rgba(0,0,0,.3)'; gctx.lineWidth = 1.5; gctx.stroke();
    gctx.fillStyle = '#ebbe96'; gctx.beginPath();
    gctx.arc(x, base - legH - torsoH - 12, 14, 0, 7); gctx.fill();
    gctx.fillStyle = '#2a2a2a';
    gctx.beginPath(); gctx.arc(x + 5, base - legH - torsoH - 14, 2, 0, 7); gctx.fill();
}
function gameCueText() {
    let best = null;
    for (const o of GG.obstacles) {
        if (o.resolved || o.d <= 0) continue;
        if (o.type === 'BLOCK' && o.lane !== GG.player.lane) continue;
        if (!best || o.d < best.d) best = o;
    }
    if (!best) return '';
    if (best.type === 'CROUCH') return '⬇ CROUCH and HOLD (bend your knees)';
    return best.lane === 0 ? 'SIDE-STEP RIGHT ➡' : best.lane === 2 ? '⬅ SIDE-STEP LEFT' : '⬅ SIDE-STEP LEFT or RIGHT ➡';
}
function renderGameCanvas() {
    const s = gameLiveStatus;
    gctx.save();
    if (GG.shake > 0) gctx.translate((Math.random() - .5) * 14, (Math.random() - .5) * 14);

    const grad = gctx.createLinearGradient(0, 0, 0, G_HORIZON);
    grad.addColorStop(0, '#120c28'); grad.addColorStop(0.45, '#3a2160');
    grad.addColorStop(0.75, '#c2477a'); grad.addColorStop(1, '#ff9660');
    gctx.fillStyle = grad; gctx.fillRect(0, 0, GW, G_HORIZON);

    gStars.forEach(st => {
        const a = 0.35 + 0.35 * Math.sin(GG.t * 2 + st.tw);
        gctx.fillStyle = 'rgba(255,255,255,' + Math.max(0, a) + ')';
        gctx.beginPath(); gctx.arc(st.x, st.y, st.r, 0, 7); gctx.fill();
    });

    const sunY = G_HORIZON - 4, sunR = 26;
    const sunGlow = gctx.createRadialGradient(GW / 2, sunY, 2, GW / 2, sunY, sunR * 2.4);
    sunGlow.addColorStop(0, 'rgba(255,214,140,.85)'); sunGlow.addColorStop(1, 'rgba(255,214,140,0)');
    gctx.fillStyle = sunGlow; gctx.beginPath(); gctx.arc(GW / 2, sunY, sunR * 2.4, 0, 7); gctx.fill();
    gctx.fillStyle = '#ffe1a8'; gctx.beginPath(); gctx.arc(GW / 2, sunY, sunR, 0, 7); gctx.fill();
    gctx.strokeStyle = 'rgba(58,33,96,.5)'; gctx.lineWidth = 3;
    for (let i = -3; i <= 3; i++) { gctx.beginPath(); gctx.moveTo(GW / 2 - sunR, sunY + i * 7); gctx.lineTo(GW / 2 + sunR, sunY + i * 7); gctx.stroke(); }

    gctx.fillStyle = '#2c1f4a';
    for (const m of gMountains) {
        const x = (((m.x - GG.bg * 0.3) % gMTOT) + gMTOT) % gMTOT - 140;
        if (x < GW + 140) {
            gctx.beginPath(); gctx.moveTo(x, G_HORIZON);
            gctx.lineTo(x + m.w / 2, G_HORIZON - m.h); gctx.lineTo(x + m.w, G_HORIZON);
            gctx.closePath(); gctx.fill();
        }
    }

    for (const b of gBuildings) {
        const x = (((b.x - GG.bg) % gTOT) + gTOT) % gTOT - 100;
        if (x < GW) {
            gctx.fillStyle = '#20182f'; gctx.fillRect(x, G_HORIZON - b.h, b.w, b.h);
            gctx.fillStyle = 'rgba(255,214,140,.55)';
            b.windows.forEach(w => gctx.fillRect(x + w.x, G_HORIZON - b.h + w.y, 2.5, 3.5));
        }
    }
    gctx.fillStyle = '#14141c'; gctx.fillRect(-10, G_HORIZON, GW + 20, GH);

    const edge = (side, d) => [GW / 2 + side * 1.5 * G_LANE_W * gScale(d), gSy(d)];
    const roadGrad = gctx.createLinearGradient(0, G_HORIZON, 0, G_PLAYER_Y + 40);
    roadGrad.addColorStop(0, '#3a3550'); roadGrad.addColorStop(1, '#232333');
    gctx.fillStyle = roadGrad; gctx.beginPath();
    [edge(-1, G_DFAR), edge(1, G_DFAR), edge(1, -3), edge(-1, -3)].forEach((q, i) => i ? gctx.lineTo(q[0], q[1]) : gctx.moveTo(q[0], q[1]));
    gctx.closePath(); gctx.fill();

    [-1, 1].forEach(side => {
        gctx.strokeStyle = 'rgba(180,140,255,.55)'; gctx.lineWidth = 3;
        gctx.shadowColor = '#c895f5'; gctx.shadowBlur = 6;
        gctx.beginPath();
        gctx.moveTo(...edge(side, G_DFAR)); gctx.lineTo(...edge(side, -3));
        gctx.stroke(); gctx.shadowBlur = 0;
    });
    gctx.strokeStyle = '#6e7080'; gctx.lineWidth = 2;
    for (const fr of [-0.5, 0.5]) for (let k = 0; k < 9; k++) {
        const d0 = ((k * 8 - GG.roadOff) % 72 + 72) % 72 - 4, d1 = d0 + 3.5;
        gctx.beginPath();
        gctx.moveTo(GW / 2 + fr * G_LANE_W * gScale(d0), gSy(d0));
        gctx.lineTo(GW / 2 + fr * G_LANE_W * gScale(d1), gSy(d1));
        gctx.stroke();
    }

    [...GG.obstacles].sort((a, b) => b.d - a.d).forEach(drawGameObstacle);
    drawGamePlayer();

    GG.sparks.forEach(o => {
        gctx.globalAlpha = Math.max(0, o.life / 0.6); gctx.fillStyle = o.color;
        gctx.beginPath(); gctx.arc(o.x, o.y, 3, 0, 7); gctx.fill();
        gctx.globalAlpha = 1;
    });
    GG.popups.forEach(o => {
        gctx.globalAlpha = Math.max(0, o.life / 0.8); gctx.fillStyle = o.color;
        gctx.font = 'bold 24px Arial'; gctx.textAlign = 'left'; gctx.fillText(o.text, o.x, o.y);
        gctx.globalAlpha = 1;
    });

    // HUD bar
    gctx.fillStyle = 'rgba(10,8,20,.45)'; gctx.fillRect(0, 0, GW, 38);
    gctx.textAlign = 'left'; gctx.font = 'bold 20px Arial'; gctx.fillStyle = '#fff';
    gctx.fillText('STEPS: ' + (s.steps || 0), 14, 26);
    gctx.fillText('SCORE: ' + GG.score, 140, 26);
    gctx.textAlign = 'right';
    gctx.fillText('❤️'.repeat(Math.max(0, GG.lives)) + '💔'.repeat(3 - Math.max(0, GG.lives)), GW - 14, 26);
    gctx.textAlign = 'center';

    if (!s.calibrated) {
        gctx.fillStyle = 'rgba(0,0,0,.7)'; gctx.fillRect(0, 0, GW, GH);
        gctx.fillStyle = '#ffe14d'; gctx.font = 'bold 30px Arial';
        gctx.fillText(s.calibrating ? 'Calibrating…' : 'Waiting for camera…', GW / 2, GH / 2 - 10);
        gctx.font = '18px Arial'; gctx.fillStyle = '#fff';
        gctx.fillText(s.message || '', GW / 2, GH / 2 + 22);
    } else {
        let c = gameCueText();
        if (!c && GG.idleT > 1.5) c = '🦶 STEP to move forward';
        if (c) {
            gctx.font = 'bold 30px Arial'; gctx.lineWidth = 5; gctx.strokeStyle = '#000';
            gctx.strokeText(c, GW / 2, 60); gctx.fillStyle = '#ffe14d'; gctx.fillText(c, GW / 2, 60);
        }
        if (s.person !== 'ok') {
            gctx.fillStyle = 'rgba(0,0,0,.6)'; gctx.fillRect(0, 0, GW, GH);
            gctx.fillStyle = '#ffe14d'; gctx.font = 'bold 24px Arial';
            gctx.fillText(s.message || 'Step back into frame', GW / 2, GH / 2);
        }
    }
    gctx.restore();
}

registerGame({
    id: 'rehab_runner',
    canUpdate: s => !!s.calibrated && s.person === 'ok',
    create(s) {
        newGameCanvasState();
        gameLastSteps = s.steps || 0;
    },
    update(s, realDt) { updateGameCanvas(realDt); },
    render(s) { renderGameCanvas(); },
    cards: [
        { id: 'gameLaneDisplay', label: 'Lane Position', init: 'Center' },
        { id: 'gameCrouchDisplay', label: 'Crouch', init: 'No' },
    ],
    hud(data) {
        const laneNames = { 0: 'Left', 1: 'Center', 2: 'Right' };
        hudText('gameLaneDisplay', laneNames[data.lane] ?? '--');
        hudText('gameCrouchDisplay', data.crouching ? 'Yes' : 'No');
    },
    buildSession(summary) {
        const steps = summary.steps || 0;
        const targetSteps = summary.target_steps;
        const judged = GG ? GG.cleared + GG.hits : 0;
        const clearRate = judged > 0 ? (GG.cleared / judged) * 100 : null;
        const progress = targetSteps ? Math.min(100, (steps / targetSteps) * 100) : null;
        const accuracy = clearRate != null ? clearRate : (progress != null ? progress : (steps > 0 ? 100 : 0));
        return {
            total_reps: targetSteps || steps,
            completed_reps: steps,
            accuracy_percentage: +accuracy.toFixed(1),
            recovery_score: +accuracy.toFixed(1),
            session_data: {
                left_steps: summary.left_steps || 0,
                right_steps: summary.right_steps || 0,
                final_score: GG ? GG.score : 0,
                lives_remaining: GG ? GG.lives : 3,
                obstacles_cleared: GG ? GG.cleared : 0,
                obstacles_hit: GG ? GG.hits : 0,
                clear_rate: clearRate != null ? +clearRate.toFixed(1) : null,
                target_progress: progress != null ? +progress.toFixed(1) : null,
            },
        };
    },
    result(summary, reason) {
        const steps = summary.steps || 0;
        const targetSteps = summary.target_steps;
        const cleared = GG ? GG.cleared : 0;
        const judged = GG ? GG.cleared + GG.hits : 0;
        return {
            rows: [
                ['Steps', `${steps}${targetSteps ? ' / ' + targetSteps : ''}`],
                ['Target Met', reason === 'target' ? 'Yes' : (targetSteps ? 'No' : '—')],
                ['Finish Reason', finishReasonLabel(reason)],
            ],
            subtitle: GG ? ` · Score ${GG.score}` + (judged ? ` · Cleared ${cleared}/${judged}` : '') : '',
        };
    },
});
})();