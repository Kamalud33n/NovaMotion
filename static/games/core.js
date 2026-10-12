const gcv = document.getElementById('gameCanvas');
const gctx = gcv.getContext('2d');
let GW = 900, GH = 460;      // canvas size; a game can ask for another one (canvasSize)

const GameModules = {};
function registerGame(mod) { GameModules[mod.id] = { ...FallbackGame, ...mod }; }

function finishReasonLabel(reason) {
    return reason === 'target' ? 'Target' : reason === 'time' ? 'Time up' : 'Manual stop';
}

const FallbackGame = {
    id: '_fallback',
    cards: [],
    canUpdate: () => false,
    create() {},
    update() {},
    render(s) {
        const grad = gctx.createLinearGradient(0, 0, 0, GH);
        grad.addColorStop(0, '#120c28'); grad.addColorStop(1, '#3a2160');
        gctx.fillStyle = grad; gctx.fillRect(0, 0, GW, GH);
        gctx.fillStyle = '#ffe14d'; gctx.font = 'bold 26px Arial'; gctx.textAlign = 'center';
        gctx.fillText(s.calibrating ? 'Calibrating…' : 'Game view not available', GW / 2, GH / 2 - 10);
        gctx.font = '18px Arial'; gctx.fillStyle = '#fff';
        gctx.fillText(s.message || '', GW / 2, GH / 2 + 22);
    },
    hud() {},
    buildSession(summary) {
        const n = summary.reps || summary.steps || 0;
        return { total_reps: n, completed_reps: n, accuracy_percentage: 0, recovery_score: 0, session_data: {} };
    },
    result(summary, reason) {
        return {
            rows: [
                ['Reps', summary.reps || 0],
                ['Missed', summary.misses != null ? summary.misses : '—'],
                ['Finish Reason', finishReasonLabel(reason)],
            ],
            subtitle: '',
        };
    },
};

function gameModuleFor(id) { return GameModules[id] || FallbackGame; }

function renderGameHud(game) {
    const grid = document.getElementById('gameHudGrid');
    if (!grid) return;
    grid.textContent = '';
    const add = (label, valueId, init, barId) => {
        const card = document.createElement('div');
        card.className = 'analytics-card';
        const l = document.createElement('span');
        l.className = 'analytics-label'; l.textContent = label;
        const v = document.createElement('span');
        v.className = 'analytics-value'; v.id = valueId; v.textContent = init;
        card.append(l, v);
        if (barId) {
            const bar = document.createElement('div');
            bar.className = 'progress-bar';
            const fill = document.createElement('div');
            fill.className = 'progress-fill'; fill.id = barId; fill.style.width = '0%';
            bar.appendChild(fill);
            card.appendChild(bar);
        }
        grid.appendChild(card);
    };
    add((game.target_label || 'Target').replace(/^Target\s+/, ''), 'gameTargetDisplay', '0 / 0', 'gameTargetProgress');
    gameModuleFor(game.id).cards.forEach(c => add(c.label, c.id, c.init, null));
    add('Calibration', 'gameCalibDisplay', 'Not started', 'gameCalibProgress');
}

function updateGameHud(data) {
    const progress = data.target_progress || 0;
    const target = data.target_steps;
    hudText('gameTargetDisplay', `${progress} / ${target != null ? target : '∞'}`);
    hudWidth('gameTargetProgress', target ? Math.min(100, (progress / target) * 100) : 0);
    currentGame().hud(data);
}

let gameLiveStatus = {};
let gameCanvasOn = false;
let gameRAF = null;
let gameLastTs = 0;
let activeGame = FallbackGame;

function currentGame() { return activeGame; }

// Some games (Balloon Pop) use the camera as their background, so the separate small
// "Patient Camera" panel is not needed: the stage is one single screen, like Exercise.
function syncGameStageView() {
    const noPanel = gameModuleFor(selectedGameId).patientView === false;
    document.getElementById('gameStage')?.classList.toggle('no-patient-view', noPanel);
}

function gameCanvasFrame(ts) {
    if (!gameCanvasOn) return;
    const realDt = gameLastTs ? Math.min(0.1, Math.max(0.001, (ts - gameLastTs) / 1000)) : 1 / 60;
    gameLastTs = ts;
    if (activeGame.canUpdate(gameLiveStatus)) activeGame.update(gameLiveStatus, realDt);
    activeGame.render(gameLiveStatus);
    gameRAF = requestAnimationFrame(gameCanvasFrame);
}

function showGameCanvas() {
    activeGame = gameModuleFor(selectedGameId);
    [GW, GH] = activeGame.canvasSize || [900, 460];
    gcv.width = GW; gcv.height = GH;
    gcv.style.display = 'block';
    activeGame.create(gameLiveStatus);
    gameLastTs = 0;
    if (!gameCanvasOn) { gameCanvasOn = true; gameRAF = requestAnimationFrame(gameCanvasFrame); }

    const panel = document.getElementById('patientVideoPanel');
    const stream = document.getElementById('poseStream');
    syncGameStageView();
    if (activeGame.patientView !== false && stream.parentElement !== panel) {
        panel.appendChild(stream);
        stream.style.display = cameraActive ? 'block' : 'none';
    }
}

function hideGameCanvas() {
    gameCanvasOn = false;
    if (gameRAF) { cancelAnimationFrame(gameRAF); gameRAF = null; }
    gcv.style.display = 'none';
    if (document.fullscreenElement || document.webkitFullscreenElement) {
        (document.exitFullscreen || document.webkitExitFullscreen).call(document);
    }

    const container = document.querySelector('.camera-container');
    const stream = document.getElementById('poseStream');
    if (stream.parentElement !== container) {
        container.insertBefore(stream, container.firstChild);
    }
}

function toggleGameFullscreen() {
    const el = document.querySelector('.camera-container.game-active') || document.querySelector('.camera-container');
    const isFs = document.fullscreenElement || document.webkitFullscreenElement;
    if (!isFs) {
        (el.requestFullscreen || el.webkitRequestFullscreen).call(el);
    } else {
        (document.exitFullscreen || document.webkitExitFullscreen).call(document);
    }
}

function updateGameFullscreenIcon() {
    const icon = document.getElementById('gameFullscreenIcon');
    if (!icon) return;
    const isFs = document.fullscreenElement || document.webkitFullscreenElement;
    icon.className = isFs ? 'fas fa-compress' : 'fas fa-expand';
}
document.addEventListener('fullscreenchange', updateGameFullscreenIcon);
document.addEventListener('webkitfullscreenchange', updateGameFullscreenIcon);