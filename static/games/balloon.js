(function () {
// Balloon Pop canvas. The server engine (game/balloon_pop/worker.py) sends positions in
// "scene units": x in [0, aspect], y in [0, 1], already mirrored like the camera view.
const BP_COLORS = ['#eb4646', '#ffa53c', '#64cd5a', '#3c96eb', '#c85ad2', '#f0d73c'];
const BP_PLANE = { abd: 'Abduction', flex: 'Flexion' };
const C_GREEN = '#78e164', C_AMBER = '#ffbe2d', C_CYAN = '#46cdeb', C_WHITE = '#ffffff';
const REST_ANGLE = 25;

let BP = null;
// scene (engine) -> canvas: the camera frame fills the canvas ("cover", like the exercise view)
let bpOx = 0, bpOy = 0, bpS = 1;
const bpX = x => bpOx + x * bpS;
const bpY = y => bpOy + y * bpS;

function bpLayout(s) {
    const asp = s.aspect || 4 / 3;
    bpS = Math.max(GW / asp, GH);
    bpOx = (GW - asp * bpS) / 2;
    bpOy = (GH - bpS) / 2;
}

function bpReset(s) {
    BP = { t: 0, flash: 0, lastPop: (s && s.pop_seq) || 0, parts: [], rings: [], floats: [], arms: {}, balloons: {} };
}

function bpLerp(a, b, k) { return a + (b - a) * k; }

function bpSmooth(prev, next, k) {
    return prev ? [bpLerp(prev[0], next[0], k), bpLerp(prev[1], next[1], k)] : next.slice();
}

function bpUpdate(s, dt) {
    if (!BP) bpReset(s);
    BP.t += dt;
    BP.flash = Math.max(0, BP.flash - dt);
    bpLayout(s);

    if ((s.pop_seq || 0) > BP.lastPop && s.last_pop) {
        const p = s.last_pop, x = bpX(p.x), y = bpY(p.y), k = GH / 720;
        for (let i = 0; i < 36; i++) {
            const a = Math.random() * Math.PI * 2, sp = (140 + Math.random() * 380) * k;
            const col = [BP_COLORS[p.color % BP_COLORS.length], C_WHITE, C_AMBER][Math.floor(Math.random() * 3)];
            BP.parts.push({ x, y, vx: Math.cos(a) * sp, vy: Math.sin(a) * sp, life: 0.6 + Math.random() * 0.5, col, sz: 2 + Math.random() * 3 });
        }
        BP.rings.push({ x, y, age: 0, col: BP_COLORS[p.color % BP_COLORS.length] });
        BP.floats.push({ x, y: y - p.r * bpS, life: 1 });
        BP.flash = 0.08;
    }
    BP.lastPop = s.pop_seq || 0;

    BP.parts.forEach(p => { p.x += p.vx * dt; p.y += p.vy * dt; p.vy += 520 * GH / 720 * dt; p.life -= dt; });
    BP.parts = BP.parts.filter(p => p.life > 0);
    BP.rings.forEach(r => { r.age += dt; });
    BP.rings = BP.rings.filter(r => r.age < 0.5);
    BP.floats.forEach(f => { f.y -= 70 * GH / 720 * dt; f.life -= dt; });
    BP.floats = BP.floats.filter(f => f.life > 0);

    // the server status arrives ~15-20x/s: ease the drawn positions so motion looks smooth
    const k = 1 - Math.pow(1 - 0.45, dt * 60);
    const arms = s.arms || {};
    for (const side of Object.keys(BP.arms)) if (!arms[side]) delete BP.arms[side];
    for (const side of Object.keys(arms)) {
        const a = arms[side], o = BP.arms[side] || {};
        BP.arms[side] = {
            sh: bpSmooth(o.sh, a.sh, k), el: bpSmooth(o.el, a.el, k), wr: bpSmooth(o.wr, a.wr, k),
            hand: bpSmooth(o.hand, a.hand, k), osh: a.osh,
            elev: a.elev, lean: a.lean, target: a.target, hand_r: a.hand_r, arm_len: a.arm_len,
        };
    }
    const bl = {};
    (s.balloons || []).forEach(b => {
        const o = BP.balloons[b.side];
        bl[b.side] = Object.assign({}, b, { pos: o ? bpSmooth(o.pos, [b.x, b.y], k) : [b.x, b.y] });
    });
    BP.balloons = bl;
}

// ---- drawing helpers
function bpText(str, x, y, size, color, align, bold) {
    gctx.font = `${bold === false ? '' : 'bold '}${size}px Arial`;
    gctx.textAlign = align || 'left';
    gctx.lineWidth = Math.max(3, size / 6); gctx.strokeStyle = 'rgba(0,0,0,.75)'; gctx.lineJoin = 'round';
    gctx.strokeText(str, x, y);
    gctx.fillStyle = color || C_WHITE;
    gctx.fillText(str, x, y);
}

function bpPanel(x, y, w, h, fill, r) {
    gctx.fillStyle = fill || 'rgba(22,26,34,.72)';
    gctx.beginPath();
    if (gctx.roundRect) gctx.roundRect(x, y, w, h, r == null ? 12 : r); else gctx.rect(x, y, w, h);
    gctx.fill();
}

function bpBar(x, y, w, h, frac, color) {
    bpPanel(x, y, w, h, 'rgba(255,255,255,.18)', h / 2);
    if (frac > 0.02) bpPanel(x, y, Math.max(h, w * Math.min(frac, 1)), h, color, h / 2);
}

// Live camera (already mirrored like a mirror, same as the engine's coordinates) is the
// game background; the balloons and the arm are drawn on top of it.
function bpBackground(s) {
    const g = gctx.createLinearGradient(0, 0, 0, GH);
    g.addColorStop(0, '#1e2646'); g.addColorStop(1, '#0e0e16');
    gctx.fillStyle = g; gctx.fillRect(0, 0, GW, GH);
    const v = (typeof camVideo !== 'undefined') ? camVideo : null;
    if (!v || v.readyState < 2 || !v.videoWidth) return;
    const w = (s.aspect || 4 / 3) * bpS;
    gctx.save();
    gctx.translate(bpOx + w, bpOy); gctx.scale(-1, 1);
    gctx.drawImage(v, 0, 0, w, bpS);
    gctx.restore();
    const vg = gctx.createRadialGradient(GW / 2, GH / 2, GH * 0.35, GW / 2, GH / 2, GW * 0.7);
    vg.addColorStop(0, 'rgba(0,0,0,0.05)'); vg.addColorStop(1, 'rgba(0,0,0,0.5)');
    gctx.fillStyle = vg; gctx.fillRect(0, 0, GW, GH);
}

function bpEase(x) {
    x = Math.min(Math.max(x, 0), 1);
    return 1 + 2.70158 * Math.pow(x - 1, 3) + 1.70158 * Math.pow(x - 1, 2);
}

function bpBalloon(b) {
    const gone = b.gone;
    const sc = gone ? 1 : Math.max(0.05, bpEase(b.age / 0.45));
    const r = Math.max(6, b.r * bpS * sc);
    const ph = b.phase || 0;
    const x = bpX(b.pos[0]) + 5 * Math.sin(BP.t * 1.5 + ph);
    const y = bpY(b.pos[1]) + 7 * Math.sin(BP.t * 2 + ph);
    const col = BP_COLORS[b.color % BP_COLORS.length];

    gctx.strokeStyle = '#e1e1e1'; gctx.lineWidth = 2; gctx.beginPath();
    for (let i = 0; i <= 75; i += 5) {
        const px = x + 9 * Math.sin(i / 8 + BP.t * 2 + ph), py = y + r + i;
        i ? gctx.lineTo(px, py) : gctx.moveTo(px, py);
    }
    gctx.stroke();

    const grad = gctx.createRadialGradient(x - r * 0.3, y - r * 0.35, r * 0.1, x, y, r);
    grad.addColorStop(0, '#ffffffcc'); grad.addColorStop(0.25, col); grad.addColorStop(1, '#00000066');
    gctx.fillStyle = col; gctx.beginPath(); gctx.ellipse(x, y, r * 0.82, r, 0, 0, 7); gctx.fill();
    gctx.fillStyle = grad; gctx.beginPath(); gctx.ellipse(x, y, r * 0.82, r, 0, 0, 7); gctx.fill();
    gctx.fillStyle = col; gctx.beginPath();
    gctx.moveTo(x, y + r - 3); gctx.lineTo(x - 9, y + r + 12); gctx.lineTo(x + 9, y + r + 12); gctx.closePath(); gctx.fill();
    gctx.strokeStyle = 'rgba(240,240,240,.85)'; gctx.lineWidth = 2;
    gctx.beginPath(); gctx.ellipse(x, y, r * 0.82, r, 0, 0, 7); gctx.stroke();

    if (b.progress > 0 && !gone) {
        gctx.strokeStyle = C_GREEN; gctx.lineWidth = 6; gctx.lineCap = 'round';
        gctx.beginPath(); gctx.arc(x, y, r + 12, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * Math.min(b.progress, 1)); gctx.stroke();
        gctx.lineCap = 'butt';
    }
    if (!gone) {
        const tx = Math.min(Math.max(x, 150), GW - 150);
        bpText(`${BP_PLANE[b.plane] || b.plane}  ${Math.round(b.target)}°`, tx, y - r - 14, 15, C_WHITE, 'center');
    }
}

function bpArm(a) {
    const pts = [a.sh, a.el, a.wr].map(p => [bpX(p[0]), bpY(p[1])]);
    gctx.lineCap = 'round';
    [[11, '#1e1e1e'], [6, C_GREEN]].forEach(([w, c]) => {
        gctx.strokeStyle = c; gctx.lineWidth = w; gctx.beginPath();
        gctx.moveTo(pts[0][0], pts[0][1]); gctx.lineTo(pts[1][0], pts[1][1]); gctx.lineTo(pts[2][0], pts[2][1]); gctx.stroke();
    });
    gctx.lineCap = 'butt';
    pts.forEach(p => {
        gctx.fillStyle = C_WHITE; gctx.beginPath(); gctx.arc(p[0], p[1], 8, 0, 7); gctx.fill();
        gctx.strokeStyle = '#1e1e1e'; gctx.lineWidth = 2; gctx.stroke();
    });
    const hx = bpX(a.hand[0]), hy = bpY(a.hand[1]), hr = (a.hand_r || 0.025) * bpS;
    gctx.strokeStyle = 'rgba(70,205,235,.35)'; gctx.lineWidth = 2; gctx.beginPath(); gctx.arc(hx, hy, hr + 7, 0, 7); gctx.stroke();
    gctx.strokeStyle = C_CYAN; gctx.lineWidth = 3; gctx.beginPath(); gctx.arc(hx, hy, hr, 0, 7); gctx.stroke();
    gctx.fillStyle = C_WHITE; gctx.beginPath(); gctx.arc(hx, hy, 3.5, 0, 7); gctx.fill();
}

function bpArcGauge(a, color) {
    const sx = bpX(a.sh[0]), sy = bpY(a.sh[1]);
    const out = a.sh[0] > a.osh[0] ? 1 : -1;
    const rad = Math.max(50, 0.3 * (a.arm_len || 0.4) * bpS);
    const pt = (e, r) => {
        const ang = (out > 0 ? 90 - e : 90 + e) * Math.PI / 180;
        return [sx + r * Math.cos(ang), sy + r * Math.sin(ang)];
    };
    const path = (to, step) => {
        gctx.beginPath();
        for (let e = 0; e <= to + 0.01; e += step) { const p = pt(e, rad); e ? gctx.lineTo(p[0], p[1]) : gctx.moveTo(p[0], p[1]); }
        gctx.stroke();
    };
    gctx.strokeStyle = 'rgba(160,160,160,.6)'; gctx.lineWidth = 2; path(180, 6);
    if (a.elev > 2) { gctx.strokeStyle = color; gctx.lineWidth = 6; path(Math.min(a.elev, 180), 3); }
    if (a.target != null) {
        const p1 = pt(a.target, rad - 12), p2 = pt(a.target, rad + 14);
        gctx.strokeStyle = C_WHITE; gctx.lineWidth = 3; gctx.beginPath(); gctx.moveTo(p1[0], p1[1]); gctx.lineTo(p2[0], p2[1]); gctx.stroke();
    }
}

function bpMeter(x, label, elev, target, color) {
    const top = 84, bot = GH - 44, hh = bot - top;
    bpPanel(x - 26, top - 40, 52, hh + 82, 'rgba(22,26,34,.55)', 16);
    bpPanel(x - 8, top, 16, hh, 'rgba(255,255,255,.18)', 8);
    const restY = bot - hh * REST_ANGLE / 180;
    gctx.strokeStyle = 'rgba(160,160,160,.8)'; gctx.lineWidth = 1;
    gctx.beginPath(); gctx.moveTo(x - 12, restY); gctx.lineTo(x + 12, restY); gctx.stroke();
    const fill = hh * Math.min(Math.max(elev, 0), 180) / 180;
    if (fill > 6) bpPanel(x - 6, bot - fill, 12, fill, color, 6);
    if (target != null) {
        const ty = bot - hh * Math.min(target, 180) / 180;
        gctx.strokeStyle = C_WHITE; gctx.lineWidth = 3; gctx.beginPath(); gctx.moveTo(x - 20, ty); gctx.lineTo(x + 20, ty); gctx.stroke();
    }
    bpText(label, x, top - 16, 16, C_WHITE, 'center');
    bpText(String(Math.round(elev)), x, bot + 26, 16, C_WHITE, 'center');
}

function bpCalibration(s) {
    bpPanel(14, 10, GW - 28, 82, 'rgba(22,26,34,.78)', 14);
    bpText('CALIBRATION', 32, 40, 20, C_AMBER);
    bpText(s.message || '', 32, 74, 18, C_WHITE, 'left', false);
    let x = 190;
    (s.cal_chips || []).forEach(c => {
        const w = 28 + c.label.length * 8;
        const fill = c.state === 'done' ? '#50aa50' : c.state === 'now' ? '#c88728' : '#50463f';
        bpPanel(x, 22, w, 22, fill, 11);
        bpText(c.label, x + w / 2, 38, 12, C_WHITE, 'center');
        x += w + 8;
    });
    bpText('hold steady', GW / 2, GH - 46, 13, '#e6e6e6', 'center', false);
    bpBar(GW / 2 - 200, GH - 36, 400, 14, s.hold_progress || 0, C_GREEN);
}

function bpHud(s) {
    bpPanel(14, 10, 520, 44, 'rgba(22,26,34,.72)', 14);
    const goal = s.target_steps;
    bpText(`REPS ${s.reps || 0}${goal ? ' / ' + goal : ''}`, 30, 40, 22, C_GREEN);
    bpText(`MISSED ${s.misses || 0}`, 215, 40, 18, C_AMBER);
    const armName = { left: 'LEFT ARM', right: 'RIGHT ARM', both_alt: 'BOTH ARMS' }[s.arm] || '';
    bpText(armName, 355, 40, 18, C_CYAN);
}

function bpRender(s) {
    if (!BP) bpReset(s);
    bpLayout(s);
    bpBackground(s);

    const phase = s.phase;
    const arms = BP.arms;
    Object.values(arms).forEach(bpArm);

    if (phase === 'PLAY') {
        BP.rings.forEach(r => {
            const f = r.age / 0.5;
            gctx.strokeStyle = r.col; gctx.lineWidth = Math.max(1, 10 * (1 - f));
            gctx.beginPath(); gctx.arc(r.x, r.y, 22 + 230 * f, 0, 7); gctx.stroke();
        });
        Object.values(BP.balloons).forEach(bpBalloon);
        BP.parts.forEach(p => {
            gctx.globalAlpha = Math.min(1, p.life * 2); gctx.fillStyle = p.col;
            gctx.beginPath(); gctx.arc(p.x, p.y, p.sz, 0, 7); gctx.fill(); gctx.globalAlpha = 1;
        });
        BP.floats.forEach(f => { gctx.globalAlpha = Math.min(1, f.life * 2); bpText('+10', f.x, f.y, 26, C_AMBER, 'center'); gctx.globalAlpha = 1; });
        Object.entries(arms).forEach(([side, a]) => {
            const ok = a.target != null && a.elev >= a.target - (s.gate || 0);
            const col = ok ? C_GREEN : C_AMBER;
            bpArcGauge(a, col);
            bpMeter(side === 'left' ? 34 : GW - 34, side[0].toUpperCase(), a.elev, a.target, col);
        });
        bpHud(s);
        if (s.message && s.person === 'ok') {
            bpPanel(GW / 2 - 260, GH - 62, 520, 40, 'rgba(180,40,40,.82)', 18);
            bpText(s.message, GW / 2, GH - 35, 18, C_WHITE, 'center', false);
        }
    } else if (phase === 'COUNTDOWN') {
        const n = Math.max(1, Math.ceil(s.countdown || 0));
        bpText('Get ready', GW / 2, GH / 2 - 70, 44, C_WHITE, 'center');
        bpText(String(n), GW / 2, GH / 2 + 60, 130, C_AMBER, 'center');
    } else {
        bpCalibration(s);
    }

    if (s.person === 'none' || (phase === 'PLAY' && s.person !== 'ok')) {
        gctx.fillStyle = 'rgba(0,0,0,.55)'; gctx.fillRect(0, 0, GW, GH);
        bpText(s.message || 'Step back into frame', GW / 2, GH / 2, 24, C_AMBER, 'center');
    }
    if (BP.flash > 0) { gctx.fillStyle = `rgba(255,255,255,${Math.min(BP.flash, 0.4)})`; gctx.fillRect(0, 0, GW, GH); }
}

registerGame({
    id: 'balloon_pop',
    patientView: false,      // the camera is already the game background
    canvasSize: [960, 600],  // 16:10, same shape as the exercise view
    canUpdate: () => true,
    create(s) { bpReset(s); },
    update(s, dt) { bpUpdate(s, dt); },
    render(s) { bpRender(s); },
    cards: [
        { id: 'gameMissDisplay', label: 'Missed', init: '0' },
        { id: 'gameLeanDisplay', label: 'Trunk Lean Events', init: '0' },
    ],
    hud(data) {
        hudText('gameMissDisplay', String(data.misses || 0));
        hudText('gameLeanDisplay', String(data.lean_events || 0));
    },
    // Everything the engine reports at the end of a round is stored in
    // session_data so the PDF reports (services/report_builder.py) can show
    // it. summary.events (per-balloon log) is added by saveGameSessionPayload.
    buildSession(summary) {
        const reps   = summary.reps || 0;
        const misses = summary.misses || 0;
        const target = summary.target_steps;                 // target reps (or null)
        const judged = reps + misses;
        const popRate = judged > 0 ? +((reps / judged) * 100).toFixed(1) : 0;
        return {
            total_reps: target || reps,
            completed_reps: reps,
            accuracy_percentage: popRate,                    // pop rate = popped / (popped + missed)
            recovery_score: popRate,
            session_data: {
                reps,
                misses,
                lean_events:     summary.lean_events || 0,
                avg_time_to_pop: summary.avg_time_to_pop || 0,
                max_elevation:   summary.max_elevation || 0,
                left_reps:       summary.left_reps || 0,
                right_reps:      summary.right_reps || 0,
                by_plane:        summary.by_plane || {},
                by_arm_plane:    summary.by_arm_plane || {},
                calibration:     summary.calibration || {},
                settings:        summary.settings || {},
                calibrated:      !!summary.calibrated,
                elapsed_seconds: summary.elapsed_seconds || 0,
                pop_rate:        popRate,
            },
        };
    },

    result(summary, reason) {
        const reps = summary.reps || 0;
        const target = summary.target_steps;
        return {
            rows: [
                ['Balloons Popped', `${reps}${target ? ' / ' + target : ''}`],
                ['Missed', summary.misses != null ? summary.misses : '—'],
                ['Avg Pop Time', summary.avg_time_to_pop ? `${summary.avg_time_to_pop}s` : '—'],
            ],
            subtitle: ` · Max ${summary.max_elevation || 0}° · Lean ${summary.lean_events || 0} · ${finishReasonLabel(reason)}`,
        };
    },
});
})();