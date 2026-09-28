// State
let sessionActive    = false;
let cameraActive     = false;
let sessionStartTime = null;
let countdownTotal   = 60;       // seconds
let countdownLeft    = 60;
let countdownTimer   = null;
let elapsedTimer     = null;
let elapsedSec       = 0;

let currentAccuracy   = 0;
let currentRom        = 0;
let currentStability  = 0;
let currentSmoothness = 100;   //  real — from backend frame-jerk variance
let currentBalance    = 100;   //  real — from backend shoulder-sway variance
let currentFatigue    = 0;     //  real — from backend rep-quality decline trend
let repCount         = 0;
let repsSyncReady    = false;   // true once /api/session/reset has landed, so stale reps from the previous session can't auto-stop this one

let poseDetected     = false;
let selectedSide     = 'both';   // 'left' / 'right' / 'both' — which limb to track & draw

// Game mode state
let sessionMode        = 'exercise';   // 'exercise' | 'game'
let gameListCache      = [];
let selectedGameId     = null;
let gameDurationTotal  = 0;            // seconds, 0 = no limit
let gamePollInterval   = null;
let gameFinishHandled  = false;

// MJPEG + polling state
let poseInterval     = null;   // setInterval handle for /api/pose_data polling
let pollFailCount    = 0;
let lastPollOkTime   = 0;
let fpsCounter       = 0;
let fpsWindowStart   = Date.now();

// Time-series data for result chart
let _timeline = [];   // [{t, acc, rom, stab}]
let _timelineTimer = null;

let resultChartInst  = null;

// Boot 
document.addEventListener('DOMContentLoaded', () => {
    loadPatients();
    retryPendingSessions();
    document.getElementById('menuToggle')?.addEventListener('click', () =>
        document.querySelector('.sidebar').classList.toggle('active'));
});

// Patients
async function loadPatients() {
    try {
        const pts = await (await fetch('/api/patients')).json();
        const sel = document.getElementById('sessionPatientSelect');
        pts.forEach(p => {
            const o = document.createElement('option');
            o.value = p.id; o.textContent = p.name;
            sel.appendChild(o);
        });
    } catch(e) { console.error(e); }
}

// Duration selector
function selectDuration(btn, sec) {
    document.querySelectorAll('#durationSelector .dur-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const wrap = document.getElementById('customDurWrap');
    if (sec === 0) {
        wrap.classList.add('show');
        countdownTotal = parseInt(document.getElementById('customDurInput').value || 3) * 60;
    } else {
        wrap.classList.remove('show');
        countdownTotal = sec;
    }
}

// Side selector (left / right / both)
function selectSide(btn, side) {
    document.querySelectorAll('#sideSelector .dur-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    selectedSide = side;
    applyActiveJointUI();
    if (cameraActive) syncExerciseType();
}

document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('customDurInput')?.addEventListener('input', e => {
        countdownTotal = Math.max(1, parseInt(e.target.value || 1)) * 60;
    });
});

// Camera: the browser opens the patient's webcam (getUserMedia), streams frames
// to the server over a WebSocket (/ws/camera), and shows the annotated frame
// (skeleton drawn by MediaPipe on the server) in the same <img id="poseStream">.
// The server needs NO camera, so this works on any deployed (HTTPS) site.
function toggleCamera() {
    cameraActive ? stopCamera() : startCamera();
}

const CAM_MAX_W = 640;            // frames are downscaled to this width before sending
const CAM_JPEG_QUALITY = 0.7;
const CAM_MIN_INTERVAL_MS = 50;   // cap at ~20 fps
// Labels of built-in laptop cameras; anything else is treated as an external USB webcam.
const INTERNAL_CAM_RE = /integrated|built-?in|internal|facetime|laptop|front/i;

let camStream = null, camWs = null, camVideo = null, camCanvas = null, camCtx = null;
let camPrevUrl = null, camLastSend = 0, camSendTimer = null;

function cameraErrorMessage(e) {
    if (!window.isSecureContext) return 'Camera needs HTTPS. Open this site with https:// (or localhost).';
    switch (e && e.name) {
        case 'NotAllowedError':
        case 'SecurityError':      return 'Camera permission denied. Allow camera access for this site in the browser settings.';
        case 'NotFoundError':
        case 'OverconstrainedError': return 'No camera found on this device.';
        case 'NotReadableError':
        case 'AbortError':         return 'Camera is busy (used by another app/tab). Close it and try again.';
        default:                   return 'Could not start the camera: ' + ((e && e.message) || e);
    }
}

// USB webcam first (priority), built-in laptop camera as fallback.
async function openBestCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        const err = new Error('getUserMedia unavailable');
        err.name = window.isSecureContext ? 'NotFoundError' : 'SecurityError';
        throw err;
    }
    const size = { width: { ideal: 1280 }, height: { ideal: 720 } };
    // First call triggers the permission prompt; only after it are device labels visible.
    let stream = await navigator.mediaDevices.getUserMedia({ video: size, audio: false });
    try {
        const cams = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'videoinput');
        if (cams.length > 1) {
            const currentId = stream.getVideoTracks()[0].getSettings().deviceId;
            const external = cams.find(d => d.label && !INTERNAL_CAM_RE.test(d.label));
            if (external && external.deviceId !== currentId) {
                const better = await navigator.mediaDevices.getUserMedia({
                    video: { ...size, deviceId: { exact: external.deviceId } }, audio: false,
                });
                stream.getTracks().forEach(t => t.stop());
                stream = better;
            }
        }
    } catch (e) {
        console.warn('openBestCamera: could not switch to external camera, using default:', e);
    }
    return stream;
}

function camSendFrame() {
    camSendTimer = null;
    if (!cameraActive || !camWs || camWs.readyState !== WebSocket.OPEN) return;
    if (!camVideo || camVideo.readyState < 2 || !camVideo.videoWidth) {   // no frame decoded yet
        camSendTimer = setTimeout(camSendFrame, 30);
        return;
    }
    // Keep the camera's real aspect ratio (a stretched frame would distort joint angles).
    const w = Math.min(CAM_MAX_W, camVideo.videoWidth);
    const h = Math.round(w * camVideo.videoHeight / camVideo.videoWidth);
    if (camCanvas.width !== w || camCanvas.height !== h) { camCanvas.width = w; camCanvas.height = h; }
    camCtx.drawImage(camVideo, 0, 0, w, h);
    camCanvas.toBlob(blob => {
        if (blob && cameraActive && camWs && camWs.readyState === WebSocket.OPEN) {
            camLastSend = performance.now();
            camWs.send(blob);
        }
    }, 'image/jpeg', CAM_JPEG_QUALITY);
}

// Send the next frame only after the previous annotated one came back (no lag build-up).
function camScheduleNext() {
    if (camSendTimer) return;
    const wait = Math.max(0, CAM_MIN_INTERVAL_MS - (performance.now() - camLastSend));
    camSendTimer = setTimeout(camSendFrame, wait);
}

function camCleanup() {
    if (camSendTimer) { clearTimeout(camSendTimer); camSendTimer = null; }
    if (camWs) {
        camWs.onopen = camWs.onmessage = camWs.onerror = camWs.onclose = null;
        try { camWs.close(); } catch (e) {}
        camWs = null;
    }
    if (camStream) { camStream.getTracks().forEach(t => t.stop()); camStream = null; }
    if (camVideo) { camVideo.srcObject = null; camVideo = null; }
    camCanvas = camCtx = null;
    if (camPrevUrl) { URL.revokeObjectURL(camPrevUrl); camPrevUrl = null; }
}

async function startCamera() {
    const img = document.getElementById('poseStream');

    try {
        camStream = await openBestCamera();
    } catch (e) {
        console.error('startCamera:', e);
        showToast(cameraErrorMessage(e), 'error');
        camCleanup();
        return;
    }

    camVideo = document.createElement('video');
    camVideo.muted = true;
    camVideo.playsInline = true;
    camVideo.srcObject = camStream;
    try { await camVideo.play(); } catch (e) { console.warn('video.play():', e); }
    camCanvas = document.createElement('canvas');
    camCtx = camCanvas.getContext('2d');

    const wsProto = location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(`${wsProto}://${location.host}/ws/camera`);
    ws.binaryType = 'blob';
    camWs = ws;

    let gotFrame = false;
    ws.onmessage = (ev) => {
        if (typeof ev.data === 'string') {          // server-side error for one frame, keep going
            console.warn('camera ws:', ev.data);
            camScheduleNext();
            return;
        }
        const url = URL.createObjectURL(ev.data);
        img.src = url;
        if (!gotFrame) { gotFrame = true; img.style.display = 'block'; }
        if (camPrevUrl) URL.revokeObjectURL(camPrevUrl);
        camPrevUrl = url;
        camScheduleNext();
    };
    ws.onclose = (ev) => {
        if (!cameraActive && !camStream) return;    // we closed it ourselves
        console.error('camera ws closed', ev.code);
        showToast(ev.code === 4401 ? 'Session expired - please log in again.'
                                    : 'Camera connection lost. Check your internet and press Camera again.', 'error');
        if (cameraActive) stopCamera(); else camCleanup();
    };
    ws.onopen = () => {
        cameraActive = true;
        document.getElementById('cameraPlaceholder').style.display = 'none';
        document.getElementById('cameraStatusText').textContent    = 'Connected';
        document.getElementById('cameraStatus').classList.add('active');
        document.getElementById('poseBadges').style.display        = sessionMode === 'game' ? 'none' : 'flex';
        document.getElementById('statusText').textContent          = 'Connected';
        document.getElementById('statusDot').className             = 'badge-dot online';

        pollFailCount  = 0;
        fpsCounter     = 0;
        fpsWindowStart = Date.now();

        // Tell backend which exercise is active so it draws only relevant joints
        syncExerciseType();

        // Poll joint-angle/detection data independently of the video stream
        poseInterval = setInterval(pollPoseData, 300);

        camSendFrame();
    };
}

// Push current exercise selection + target ROM to backend (drives joint
// filtering AND server-side rep counting / threshold logic)
async function syncExerciseType() {
    applyActiveJointUI();
    try {
        await fetch('/api/exercise_type', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                exercise_type: document.getElementById('exerciseType').value,
                target_rom:    parseFloat(document.getElementById('targetRom').value) || 90,
                side:          selectedSide,
            }),
        });
    } catch(e) { console.error('syncExerciseType:', e); }
}

// Which BADGE/ANGLE-GRID group (shoulder/elbow/knee/all) lights up for the
// current exercise. session.html's UI only has shoulder/elbow/knee badges
// (no dedicated hand/ankle/hip badges), so those exercise types are mapped
// to the closest existing group instead of silently falling through to a
// wrong default.
function getActiveJointGroup() {
    const ex = (document.getElementById('exerciseType')?.value || '').toLowerCase();
    if (ex.includes('balance'))                                  return 'all';
    if (ex.includes('grip') || ex.includes('finger'))            return 'finger'; // Hand Grip Exercise — MediaPipe Hands, dedicated group
    if (ex.includes('hand'))                                     return 'elbow'; // old pose-only Hand Rehab — no dedicated hand badge, closest group
    if (ex.includes('ankle'))                                    return 'knee';  // no dedicated ankle badge — closest group
    if (ex.includes('hip'))                                      return 'all';   // spans shoulder+knee area — show all rather than mis-dim
    if (ex.includes('shoulder') || ex.includes('arm'))           return 'shoulder';
    if (ex.includes('elbow'))                                    return 'elbow';
    if (ex.includes('knee') || ex.includes('squat') || ex.includes('leg'))
        return 'knee';
    return 'elbow'; // fallback, matches backend default
}

// Dim every joint badge/angle-item except the one(s) relevant to the
// currently selected exercise AND currently selected side, and reset the
// dimmed ones to "--°" so old numbers from a previous exercise/side don't
// linger and look "live".
function applyActiveJointUI() {
    const active = getActiveJointGroup();
    document.querySelectorAll('[data-joint]').forEach(el => {
        const groupOk = active === 'all' || el.dataset.joint === active;
        const side    = el.dataset.side; // undefined for side-agnostic items (fingers)
        const sideOk  = !side || selectedSide === 'both' || side === selectedSide;
        const isActive = groupOk && sideOk;
        el.classList.toggle('joint-inactive', !isActive);
        el.classList.toggle('joint-active', isActive);
        if (!isActive) {
            const valSpan = el.querySelector('.angle-value');
            if (valSpan) valSpan.textContent = '--°';
        }
    });
}

document.addEventListener('DOMContentLoaded', () => {
    applyActiveJointUI();
    document.getElementById('exerciseType')?.addEventListener('change', () => {
        applyActiveJointUI();
        if (cameraActive) syncExerciseType();
    });
    document.getElementById('targetRom')?.addEventListener('change', () => {
        if (cameraActive) syncExerciseType();
    });
});

async function stopCamera() {
    const img = document.getElementById('poseStream');
    img.onerror = null;
    img.src = '';
    img.style.display = 'none';
    cameraActive = false;
    poseDetected = false;
    camCleanup();   // stop sending frames, close the WebSocket, release the browser camera

    clearInterval(poseInterval);
    poseInterval = null;

    document.getElementById('cameraPlaceholder').style.display = 'flex';
    document.getElementById('cameraStatusText').textContent    = 'Disconnected';
    document.getElementById('cameraStatus').classList.remove('active');
    document.getElementById('poseBadges').style.display        = 'none';
    document.getElementById('noPoseWarning').style.display     = 'none';
    document.getElementById('fpsDisplay').textContent          = '--';
    document.getElementById('statusText').textContent          = 'Ready';
    document.getElementById('statusDot').className             = 'badge-dot offline';

    try {
        await fetch('/api/camera/stop', { method: 'POST' });
    } catch(e) {
        console.error('stopCamera: /api/camera/stop failed:', e);
    }
}

// Poll latest pose data (replaces ws.onmessage)
async function pollPoseData() {
    try {
        const res = await fetch('/api/pose_data');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        pollFailCount = 0;
        lastPollOkTime = Date.now();

        // crude FPS readout based on poll cadence/responsiveness
        fpsCounter++;
        const now = Date.now();
        if (now - fpsWindowStart >= 1000) {
            document.getElementById('fpsDisplay').textContent = fpsCounter;
            fpsCounter = 0;
            fpsWindowStart = now;
        }

        handlePoseData(data);
    } catch(e) {
        pollFailCount++;
        console.error('pollPoseData:', e);
        if (pollFailCount >= 5) {
            document.getElementById('statusText').textContent = 'Connection lost';
            document.getElementById('statusDot').className    = 'badge-dot error';
        }
    }
}

// Pose data handler 
function handlePoseData(data) {
    poseDetected = !!data.detected;

    document.getElementById('noPoseWarning').style.display  = poseDetected ? 'none' : 'block';
    document.getElementById('detectionStatus').textContent  = poseDetected ? 'Yes' : 'No';
    document.getElementById('detectionStatus').style.color  = poseDetected ? '#1D8A6D' : '#D9503E';
    if (sessionActive && repsSyncReady && data.reps != null) {
        repCount = data.reps;
        const repGoal = parseInt(document.getElementById('targetReps').value);
        if (sessionMode === 'exercise' && repGoal > 0 && repCount >= repGoal) {
            updateAnalytics(data.primary_angle, parseInt(document.getElementById('targetRom').value) || 90);
            showToast('🎯 Target reps completed! Session ending...', 'success');
            stopSession();
            return;
        }
    }
    if (!poseDetected) return;

    const ang = data.angles || {};
    const set  = (id, v) => document.getElementById(id).textContent = v != null ? `${Math.round(v)}°` : '--°';
    const active = getActiveJointGroup();
    const targetRom = parseInt(document.getElementById('targetRom').value) || 90;

    // The camera frame is mirrored BEFORE MediaPipe runs, so MediaPipe's l_*/r_*
    // keys are swapped relative to the patient. The backend already compensates
    // for that when it DRAWS the skeleton and MEASURES ROM/reps for the selected
    // side (metrics.set_exercise_state), but the badges below still read the raw
    // keys - so "Left" showed one arm's angle while the other arm was being drawn
    // and counted. Read the swapped key so badge, skeleton and rep counter always
    // refer to the same limb.
    const A = (joint, uiSide) => ang[(uiSide === 'left' ? 'r_' : 'l_') + joint];

    if (active === 'all' || active === 'shoulder') {
        if (selectedSide === 'both' || selectedSide === 'left') {
            set('leftShoulderAngle', A('shoulder','left'));
            updateBadge('badge_l_shoulder','L.Shoulder', A('shoulder','left'), targetRom);
        }
        if (selectedSide === 'both' || selectedSide === 'right') {
            set('rightShoulderAngle',A('shoulder','right'));
            updateBadge('badge_r_shoulder','R.Shoulder', A('shoulder','right'), targetRom);
        }
    }
    if (active === 'all' || active === 'elbow') {
        if (selectedSide === 'both' || selectedSide === 'left') {
            set('leftElbowAngle',    A('elbow','left'));
            updateBadge('badge_l_elbow',   'L.Elbow',   A('elbow','left'), targetRom);
        }
        if (selectedSide === 'both' || selectedSide === 'right') {
            set('rightElbowAngle',   A('elbow','right'));
            updateBadge('badge_r_elbow',   'R.Elbow',   A('elbow','right'), targetRom);
        }
    }
    if (active === 'all' || active === 'knee') {
        if (selectedSide === 'both' || selectedSide === 'left') {
            set('leftKneeAngle',     A('knee','left'));
            updateBadge('badge_l_knee',    'L.Knee',    A('knee','left'),  targetRom);
        }
        if (selectedSide === 'both' || selectedSide === 'right') {
            set('rightKneeAngle',    A('knee','right'));
            updateBadge('badge_r_knee',    'R.Knee',    A('knee','right'),  targetRom);
        }
    }
    if (active === 'all' || active === 'finger') {
        // ang.index/middle/ring/pinky/thumb come from MediaPipe Hands
        // finger-curl calculation (see metrics.compute_finger_curl_angles) —
        // ~170-180° = extended, ~40-70° = curled into a fist.
        set('indexFingerAngle',  ang.index);
        set('middleFingerAngle', ang.middle);
        set('ringFingerAngle',   ang.ring);
        set('pinkyFingerAngle',  ang.pinky);
        set('thumbFingerAngle',  ang.thumb);
    }

    // Reps, stability, smoothness, balance, fatigue now ALL come straight
    // from the backend (real MediaPipe computation — no Math.random, no
    // metric copied from another metric)
    if (sessionActive) {
        try {
            if (data.stability  != null) currentStability  = data.stability;
            if (data.smoothness != null) currentSmoothness = data.smoothness;
            if (data.balance    != null) currentBalance    = data.balance;
            if (data.fatigue    != null) currentFatigue    = data.fatigue;
            updateAnalytics(data.primary_angle, targetRom);
        } catch (err) {
            console.error('Live Analytics update failed:', err, { data, targetRom, sessionActive });
        }
    }
}

// Joint angle -> how far the joint has MOVED from neutral (mirrors
// metrics.angle_to_rom on the backend). Shoulder angle grows as the arm rises;
// elbow/knee/hip/finger angles are ~180deg when straight and shrink as they
// flex, so movement = 180 - angle. Badge colours must use this, otherwise a
// resting straight leg (170deg) shows green against a 90deg target.
function romFromAngle(id, angle) {
    if (angle == null) return null;
    if (String(id).toLowerCase().includes('shoulder')) return Math.max(0, angle);
    return Math.max(0, 180 - angle);
}

function isBalanceExercise() {
    return (document.getElementById('exerciseType')?.value || '').toLowerCase().includes('balance');
}

function isFingerExercise() {
    const ex = (document.getElementById('exerciseType')?.value || '').toLowerCase();
    return ex.includes('grip') || ex.includes('finger');
}

function updateBadge(id, label, angle, targetRom) {
    const el  = document.getElementById(id);
    if (!el) return;
    const val = angle != null ? Math.round(angle) : null;
    el.innerHTML = `<i class="fas fa-circle" style="font-size:7px"></i> ${label}: ${val!=null?val+'°':'--°'}`;
    if (val == null) return;
    const pct = romFromAngle(id, val) / (targetRom > 0 ? targetRom : 90);
    el.className = 'pose-badge ' + (pct >= 0.85 ? 'green' : pct >= 0.6 ? 'orange' : 'red');
}

// Session start / stop 
function startSession() {
    if (sessionMode === 'game') { startGameSession(); return; }
    if (!document.getElementById('sessionPatientSelect').value) {
        showToast('Please select a patient', 'error'); return;
    }
    if (!cameraActive) {
        showToast('Please start the camera first', 'error'); return;
    }

    // Read custom duration if selected
    // Scoped to the EXERCISE duration selector. The unscoped selector also matched
    // the Game section's "No Limit" button (data-sec="0", active by default), so
    // every session silently used the Custom box (3 min) instead of 1/2/5 min.
    const customBtn = document.querySelector('#durationSelector .dur-btn[data-sec="0"].active');
    if (customBtn) {
        countdownTotal = Math.max(1, parseInt(document.getElementById('customDurInput').value||1)) * 60;
    }

    sessionActive    = true;
    sessionStartTime = Date.now();
    countdownLeft    = countdownTotal;
    elapsedSec       = 0;
    repCount = currentAccuracy = currentRom = 0;
    currentStability  = 100;
    currentSmoothness = 100;
    currentBalance    = 100;
    currentFatigue    = 0;
    _timeline = [];

    // Reset server-side rep counter + stability buffer for this new
    // session, and make sure backend has the latest exercise/ROM target
    repsSyncReady = false;
    syncExerciseType();
    fetch('/api/session/reset', { method: 'POST' })
        .catch(e => console.error('session reset failed:', e))
        // small delay > poll interval (300ms) so a poll already in flight before
        // the reset can't deliver an old rep count after we start trusting reps
        .finally(() => setTimeout(() => { repsSyncReady = true; }, 450));

    document.getElementById('startSessionBtn').style.display = 'none';
    document.getElementById('stopSessionBtn').style.display  = 'inline-block';
    document.getElementById('statusText').textContent        = 'Session Active';
    document.getElementById('statusDot').className           = 'badge-dot recording';
    document.getElementById('recIndicator').classList.add('active');
    document.getElementById('timerRing').classList.add('active');

    updateTimerRing();

    // Elapsed timer
    elapsedTimer = setInterval(() => {
        elapsedSec++;
        const m = String(Math.floor(elapsedSec/60)).padStart(2,'0');
        const s = String(elapsedSec%60).padStart(2,'0');
        document.getElementById('sessionTimer').textContent = `${m}:${s}`;
    }, 1000);

    // Countdown timer
    countdownTimer = setInterval(() => {
        countdownLeft--;
        updateTimerRing();

        const rm = String(Math.floor(countdownLeft/60)).padStart(2,'0');
        const rs = String(countdownLeft%60).padStart(2,'0');
        document.getElementById('remainingTimer').textContent = `${rm}:${rs}`;

        // Color warning when < 10s
        const arc = document.getElementById('timerArc');
        if (countdownLeft <= 10) arc.style.stroke = '#D9503E';
        else if (countdownLeft <= 30) arc.style.stroke = '#D68F35';
        else arc.style.stroke = '#1D8A6D';

        if (countdownLeft <= 0) {
            showToast('⏰ Time up! Session ending...', 'success');
            stopSession();
        }
    }, 1000);

    // Record timeline every 5s
    _timelineTimer = setInterval(() => {
        if (sessionActive) {
            _timeline.push({
                t:      elapsedSec,
                acc:    +currentAccuracy.toFixed(1),
                rom:    +currentRom.toFixed(1),
                stab:   +currentStability.toFixed(1),
                smooth: +currentSmoothness.toFixed(1),
                bal:    +currentBalance.toFixed(1),
                fat:    +currentFatigue.toFixed(1),
            });
        }
    }, 5000);

    showToast('Session started!', 'success');
}

function stopSession() {
    if (sessionMode === 'game') { stopGameSession(); return; }
    if (!sessionActive) return;
    sessionActive = false;
    repsSyncReady = false;

    clearInterval(countdownTimer);
    clearInterval(elapsedTimer);
    clearInterval(_timelineTimer);

    document.getElementById('startSessionBtn').style.display = 'inline-block';
    document.getElementById('stopSessionBtn').style.display  = 'none';
    const stillConnected = cameraActive;
    document.getElementById('statusText').textContent        = stillConnected ? 'Connected' : 'Ready';
    document.getElementById('statusDot').className           = stillConnected ? 'badge-dot online' : 'badge-dot offline';
    document.getElementById('recIndicator').classList.remove('active');
    document.getElementById('timerRing').classList.remove('active');
    document.getElementById('remainingTimer').textContent    = '--:--';

    saveSession();
    showResultModal();
}

function updateTimerRing() {
    const circ  = 188.5;
    const pct   = countdownLeft / countdownTotal;
    const offset = circ * (1 - pct);
    document.getElementById('timerArc').style.strokeDashoffset = offset;
    const m = String(Math.floor(countdownLeft/60)).padStart(2,'0');
    const s = String(countdownLeft%60).padStart(2,'0');
    document.getElementById('timerText').textContent = `${m}:${s}`;
}

// Analytics update (fully derived from real backend data, no Math.random) 
function updateAnalytics(primaryAngle, targetRom) {
    if (!(targetRom > 0)) targetRom = 90;   // empty/invalid target ROM used to give NaN% bars
    if (primaryAngle != null) {
        currentRom = Math.min(targetRom, Math.max(0, primaryAngle));
        const romPct = Math.min(100, (currentRom / targetRom) * 100);
        // Accuracy = form quality, not just "did the joint reach the target
        // angle". A jerky/unstable rep can still touch the target ROM, so
        // blend in the real stability + smoothness scores (same weighting
        // pattern as calculate_recovery_score in services/helpers.py) rather
        // than reporting ROM completion alone.
        currentAccuracy = Math.min(100,
            romPct * 0.6 + currentStability * 0.2 + currentSmoothness * 0.2);
    } else if (isBalanceExercise()) {
        // Balance has no joint angle / ROM, so it is scored on steadiness only
        // (accuracy used to sit at 0% for the whole session).
        currentAccuracy = Math.min(100, (currentStability + currentBalance) / 2);
    }

    const targetReps = parseInt(document.getElementById('targetReps').value) || 1;
    document.getElementById('accuracyDisplay').textContent   = `${Math.round(currentAccuracy)}%`;
    document.getElementById('accuracyProgress').style.width  = `${currentAccuracy}%`;
    document.getElementById('romDisplay').textContent        = `${Math.round(currentRom)}°`;
    document.getElementById('romProgress').style.width       = `${Math.min(100, (currentRom/targetRom)*100)}%`;
    document.getElementById('stabilityDisplay').textContent  = `${Math.round(currentStability)}%`;
    document.getElementById('stabilityProgress').style.width = `${currentStability}%`;
    document.getElementById('repCounter').textContent        = `${repCount} / ${targetReps}`;
    document.getElementById('repProgress').style.width       = `${Math.min(100, (repCount/targetReps)*100)}%`;

    updateFeedback(currentAccuracy, currentRom);
}

function updateFeedback(acc, rom) {
    const t = parseInt(document.getElementById('targetRom').value) || 90;
    let msg, icon, cls;
    if (acc>85&&rom>t*0.8)      { msg='Excellent form! Keep it up!';            icon='fa-check-circle';       cls='success'; }
    else if (acc>70&&rom>t*0.6) { msg='Good progress. Focus on range of motion.'; icon='fa-thumbs-up';         cls='warning'; }
    else if (acc>50)            { msg='Needs improvement. Adjust your posture.';  icon='fa-exclamation-triangle'; cls='warning'; }
    else                        { msg='Consult your therapist for guidance.';     icon='fa-exclamation-circle'; cls='error'; }
    document.getElementById('feedbackDisplay').innerHTML =
        `<div class="feedback-message ${cls}"><i class="fas ${icon}"></i><span>${msg}</span></div>`;
}

// Result modal 
function showResultModal() {
    const avgAcc  = _timeline.length ? (_timeline.reduce((s,r)=>s+r.acc, 0) / _timeline.length).toFixed(1) : currentAccuracy.toFixed(1);
    const avgRom  = _timeline.length ? (_timeline.reduce((s,r)=>s+r.rom, 0) / _timeline.length).toFixed(1) : currentRom.toFixed(1);
    const avgStab = _timeline.length ? (_timeline.reduce((s,r)=>s+r.stab,0) / _timeline.length).toFixed(1) : currentStability.toFixed(1);

    const durSec  = elapsedSec;
    const durStr  = `${String(Math.floor(durSec/60)).padStart(2,'0')}:${String(durSec%60).padStart(2,'0')}`;

    document.getElementById('res_duration').textContent = durStr;
    document.getElementById('res_accuracy').textContent  = `${avgAcc}%`;
    document.getElementById('res_rom').textContent       = `${avgRom}°`;
    document.getElementById('res_stability').textContent = `${avgStab}%`;

    const ex = document.getElementById('exerciseType').value;
    document.getElementById('resultSubtitle').textContent =
        `${ex} · ${durStr} · ${document.querySelector('#sessionPatientSelect option:checked')?.textContent || ''}`;

    // Build chart
    if (resultChartInst) { resultChartInst.destroy(); resultChartInst = null; }

    const labels = _timeline.length
        ? _timeline.map(r => `${Math.floor(r.t/60)}:${String(r.t%60).padStart(2,'0')}`)
        : ['Start', 'End'];
    const accData  = _timeline.length ? _timeline.map(r=>r.acc)  : [0, +avgAcc];
    const romData  = _timeline.length ? _timeline.map(r=>r.rom)  : [0, +avgRom];
    const stabData = _timeline.length ? _timeline.map(r=>r.stab) : [0, +avgStab];

    const ctx = document.getElementById('resultChart').getContext('2d');
    resultChartInst = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [
                { label:'Accuracy %',  data:accData,  borderColor:'#1D8A6D', backgroundColor:'rgba(29,138,109,0.1)',
                  tension:0.4, fill:true, borderWidth:2, pointRadius:3, pointBackgroundColor:'#1D8A6D' },
                { label:'ROM °',       data:romData,  borderColor:'#3E6FD9', backgroundColor:'rgba(59,91,165,0.1)',
                  tension:0.4, fill:true, borderWidth:2, pointRadius:3, pointBackgroundColor:'#3E6FD9' },
                { label:'Stability %', data:stabData, borderColor:'#8B5FC7', backgroundColor:'rgba(122,92,142,0.1)',
                  tension:0.4, fill:true, borderWidth:2, pointRadius:3, pointBackgroundColor:'#8B5FC7' },
            ]
        },
        options: {
            responsive:true, maintainAspectRatio:false,
            plugins:{
                legend:{ display:true, position:'top',
                    labels:{ usePointStyle:true, pointStyle:'circle', padding:16, font:{size:11} }},
                tooltip:{ backgroundColor:'rgba(255,255,255,0.95)',
                    titleColor:'#1a2332', bodyColor:'#5a6b7c',
                    borderColor:'rgba(0,0,0,0.08)', borderWidth:1, cornerRadius:10, padding:10 }
            },
            scales:{
                y:{ beginAtZero:true, max:Math.max(100, Math.ceil(+avgRom/10)*10+10),
                    grid:{color:'rgba(0,0,0,0.04)'}, ticks:{font:{size:10}} },
                x:{ grid:{display:false}, ticks:{font:{size:10}, maxRotation:45} }
            },
            interaction:{ intersect:false, mode:'index' }
        }
    });

    document.getElementById('resultModal').classList.add('active');
}

function closeResult() {
    document.getElementById('resultModal').classList.remove('active');
    resetResultModalLabels();
    if (resultChartInst) { resultChartInst.destroy(); resultChartInst=null; }
    // Reset UI
    currentAccuracy=currentRom=currentStability=repCount=elapsedSec=0;
    currentSmoothness=100; currentBalance=100; currentFatigue=0;
    document.getElementById('sessionTimer').textContent    = '00:00';
    document.getElementById('remainingTimer').textContent  = '--:--';
    document.getElementById('accuracyDisplay').textContent = '0%';
    document.getElementById('romDisplay').textContent      = '0°';
    document.getElementById('stabilityDisplay').textContent= '0%';
    document.getElementById('repCounter').textContent      = '0 / 0';
    ['accuracyProgress','romProgress','stabilityProgress','repProgress']
        .forEach(id => document.getElementById(id).style.width='0%');
    document.getElementById('feedbackDisplay').innerHTML =
        '<div class="feedback-message"><i class="fas fa-info-circle"></i><span>Start session to receive feedback</span></div>';
}

// Save session
async function saveSession() {
    const patientId    = document.getElementById('sessionPatientSelect').value;
    const exerciseType = document.getElementById('exerciseType').value;
    const targetReps   = parseInt(document.getElementById('targetReps').value);

    const jointAngles = [];
    document.querySelectorAll('.angle-item').forEach(item => {
        const label = item.querySelector('.angle-label').textContent;
        const val   = item.querySelector('.angle-value').textContent;
        if (val !== '--°') jointAngles.push({
            joint_name:  label.toLowerCase().replace(/\s+/g,'_'),
            angle_value: parseFloat(val),
        });
    });

    // All 6 metrics below are real independent averages pulled from the
    // session timeline, which itself is fed live from backend MediaPipe
    // calculations (angles, rep-cycle counter, hip jitter, shoulder sway,
    // frame-jerk variance, rep-quality trend). Nothing here is copied from
    // another metric or randomly generated.
    const avg = (key, fallback) =>
        _timeline.length ? _timeline.reduce((s,r)=>s+r[key], 0)/_timeline.length : fallback;

    const avgAcc    = avg('acc',    currentAccuracy);
    const avgRom    = avg('rom',    currentRom);
    const avgStab   = avg('stab',   currentStability);
    const avgSmooth = avg('smooth', currentSmoothness);
    const avgBal    = avg('bal',    currentBalance);
    const avgFat    = avg('fat',    currentFatigue);

    // Recovery score on a true 0-100 scale, same weights as
    // services/helpers.calculate_recovery_score (acc 30 / ROM 20 / stability 25 /
    // balance 25). ROM is converted to % of target first (it used to add raw
    // DEGREES to percentages and could exceed 100). Metrics that don't exist for
    // an exercise are left out and the weights renormalised: no ROM for Balance,
    // no balance signal for hand-grip (hands-only frame).
    const targetRomVal = parseInt(document.getElementById('targetRom').value) || 90;
    const romPctAvg    = Math.min(100, (avgRom / targetRomVal) * 100);
    const parts = [[avgAcc, 0.30], [avgStab, 0.25]];
    if (!isBalanceExercise()) parts.push([romPctAvg, 0.20]);
    if (!isFingerExercise())  parts.push([avgBal, 0.25]);
    const wSum = parts.reduce((t, [, w]) => t + w, 0);
    const recoveryScore = Math.max(0, Math.min(100, parts.reduce((t, [v, w]) => t + v * w, 0) / wSum));

    const payload = {
        patient_id:          patientId,
        exercise_type:       exerciseType,
        start_time:          new Date(sessionStartTime).toISOString(),
        end_time:            new Date().toISOString(),
        duration_seconds:    elapsedSec,
        total_reps:          targetReps,
        completed_reps:      repCount,
        accuracy_percentage: +avgAcc.toFixed(1),
        average_rom:         +avgRom.toFixed(1),
        stability_score:     +avgStab.toFixed(1),
        balance_score:       +avgBal.toFixed(1),
        movement_smoothness: +avgSmooth.toFixed(1),
        fatigue_estimation:  +avgFat.toFixed(1),
        recovery_score:      +recoveryScore.toFixed(1),
        incorrect_movements: Math.max(0, targetReps - repCount),
        joint_angles:        jointAngles,
        exercise_results:    [],
        session_data:        { timeline: _timeline },
    };

    return saveSessionPayload(payload);
}

// Actually performs the save + handles success/failure UI. Returns true/false.
async function saveSessionPayload(payload) {
    try {
        const res  = await fetch('/api/sessions', {
            method: 'POST',
            headers: { 'Content-Type':'application/json' },
            body: JSON.stringify(payload),
        });

        if (!res.ok) {
            let msg = `Session save failed (${res.status})`;
            try { msg = (await res.json()).detail || msg; } catch (_) {}
            console.error('saveSession failed:', msg);
            stashFailedSession(payload);
            showToast(`${msg} — saved locally, will retry`, 'error', 5000);
            return false;
        }

        // Success — make sure no stale local backup is replayed on top of it.
        clearStashedSession(payload);
        showToast('Session saved', 'success');
        return true;
    } catch (e) {
        console.error('saveSession network error:', e);
        stashFailedSession(payload);
        showToast('No connection — session saved locally, will retry', 'error', 5000);
        return false;
    }
}

// Local backup so a failed save isn't lost
const PENDING_SESSIONS_KEY = 'novamotion_pending_sessions';

function stashFailedSession(payload) {
    try {
        const pending = JSON.parse(localStorage.getItem(PENDING_SESSIONS_KEY) || '[]');
        pending.push(payload);
        localStorage.setItem(PENDING_SESSIONS_KEY, JSON.stringify(pending));
    } catch (e) { console.error('stashFailedSession:', e); }
}

function clearStashedSession(payload) {
    try {
        const pending = JSON.parse(localStorage.getItem(PENDING_SESSIONS_KEY) || '[]');
        const remaining = pending.filter(p => JSON.stringify(p) !== JSON.stringify(payload));
        localStorage.setItem(PENDING_SESSIONS_KEY, JSON.stringify(remaining));
    } catch (e) { console.error('clearStashedSession:', e); }
}

// Retry any sessions that failed to save earlier (e.g. network drop), called on page load.
async function retryPendingSessions() {
    let pending = [];
    try { pending = JSON.parse(localStorage.getItem(PENDING_SESSIONS_KEY) || '[]'); }
    catch (e) { return; }
    if (!pending.length) return;

    showToast(`Retrying ${pending.length} unsaved session(s)...`, 'success');
    for (const payload of pending) {
        await saveSessionPayload(payload);
    }
}

// Small local helper (session.html doesn't load /static/app.js)
function formatDuration(seconds) {
    if (!seconds || seconds < 0) return '00:00';
    const m = Math.floor(seconds/60), s = Math.floor(seconds%60);
    return `${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`;
}

// ══════════════════════════════════════════════════════════════════════
// GAME MODE
// ══════════════════════════════════════════════════════════════════════

// ── Canvas game engine ───────────────────────────────────────────────
// Renders the actual "runner" view (player, obstacles, road) inside
// #gameCanvas. Player position/lane/crouch/steps are NOT computed here —
// they come from the server engine (game/rehab_runner/worker.py) via the
// existing pollGameStatus() poll; this block only turns that status into
// pixels + a light client-side obstacle/score/lives layer for feedback.
// Ported from the standalone rehab_runner.py prototype's canvas game.
const gcv = document.getElementById('gameCanvas');
const gctx = gcv.getContext('2d');
const GW = 900, GH = 460, G_HORIZON = 100, G_PLAYER_Y = 372, G_NEAR_HW = 300, G_FOCAL = 10;
const G_MAXD = 50, G_REACT_D = 30, G_DFAR = 2000;
const G_LANE_W = 2 * G_NEAR_HW / 3, G_BEAM_TOP = 125, G_BEAM_T = 35;
const G_SMOOTH_X = 10;
const G_STRIDE = 5, G_STRIDE_SPEED = 20, G_MAX_Q = 15;
const G_SPAWN_MIN = 32, G_SPAWN_MAX = 42;
const gScale = d => G_FOCAL / (G_FOCAL + d);
const gSy = d => G_HORIZON + (G_PLAYER_Y - G_HORIZON) * gScale(d);
const gLx = (lane, d) => GW / 2 + (lane - 1) * G_LANE_W * gScale(d);

let gameLiveStatus = {};      // latest /api/game/status payload (set from pollGameStatus)
let gameLastSteps  = 0;       // steps count as of the previous poll, to derive step deltas
let gameCanvasOn   = false;   // is the render loop currently running
let gameRAF        = null;
let GG              = null;   // canvas-side game state (obstacles, score, lives, road position)

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

function updateGamePlayer() {
    const p = GG.player, s = gameLiveStatus;
    const xn = Math.max(-1, Math.min(1, Number(s.x_norm) || 0));
    const tx = GW / 2 + xn * G_LANE_W;
    if (s.lane !== undefined) p.lane = s.lane;
    p.x += (tx - p.x) * Math.min(1, 0.05 * G_SMOOTH_X);
    p.state = s.crouching ? 'CROUCH' : 'RUN';
    if (p.flash > 0) p.flash -= 0.05;
}

function updateGameCanvas() {
    const dt = 0.05; // fixed step (canvas polls status ~10x/sec; render runs at 60fps but game logic ticks per animation frame at this nominal rate)
    GG.t += dt;
    updateGamePlayer();
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
            let hit = false;
            if (o.type === 'BLOCK') hit = Math.abs(p.x - gLx(o.lane, 0)) < G_LANE_W * 0.55;
            else hit = p.state !== 'CROUCH';
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

function gameCanvasFrame() {
    if (!gameCanvasOn) return;
    if (gameLiveStatus.calibrated && gameLiveStatus.person === 'ok') updateGameCanvas();
    renderGameCanvas();
    gameRAF = requestAnimationFrame(gameCanvasFrame);
}
function showGameCanvas() {
    gcv.style.display = 'block';
    newGameCanvasState();
    gameLastSteps = gameLiveStatus.steps || 0;
    if (!gameCanvasOn) { gameCanvasOn = true; gameRAF = requestAnimationFrame(gameCanvasFrame); }

    // Move the live pose stream into the side "Patient Camera" panel so the
    // therapist can still see the raw feed next to the (bigger) game view —
    // same <img>/connection, just reparented, so nothing reloads.
    const panel = document.getElementById('patientVideoPanel');
    const stream = document.getElementById('poseStream');
    if (stream.parentElement !== panel) {
        panel.appendChild(stream);
        stream.style.display = cameraActive ? 'block' : 'none';
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
document.addEventListener('fullscreenchange', updateGameFullscreenIcon);
document.addEventListener('webkitfullscreenchange', updateGameFullscreenIcon);
function updateGameFullscreenIcon() {
    const icon = document.getElementById('gameFullscreenIcon');
    if (!icon) return;
    const isFs = document.fullscreenElement || document.webkitFullscreenElement;
    icon.className = isFs ? 'fas fa-compress' : 'fas fa-expand';
}

function hideGameCanvas() {
    gameCanvasOn = false;
    if (gameRAF) { cancelAnimationFrame(gameRAF); gameRAF = null; }
    gcv.style.display = 'none';
    if (document.fullscreenElement || document.webkitFullscreenElement) {
        (document.exitFullscreen || document.webkitExitFullscreen).call(document);
    }

    // Move the pose stream back into the main camera box for Exercise mode.
    const container = document.querySelector('.camera-container');
    const stream = document.getElementById('poseStream');
    if (stream.parentElement !== container) {
        container.insertBefore(stream, container.firstChild);
    }
}

// Session Type toggle (Exercise / Game) — now toggles the config/analytics
// field-groups inside the shared merged cards, instead of two separate cards.
function selectSessionMode(btn, mode) {
    if (sessionActive) { showToast('Stop the current session before switching mode', 'error'); return; }
    document.querySelectorAll('#modeSelector .dur-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    sessionMode = mode;

    document.getElementById('exerciseConfigFields').style.display     = mode === 'exercise' ? 'block' : 'none';
    document.getElementById('gameConfigFields').style.display         = mode === 'game'     ? 'block' : 'none';
    document.getElementById('exerciseAnalyticsFields').style.display  = mode === 'exercise' ? 'block' : 'none';
    document.getElementById('gameAnalyticsFields').style.display      = mode === 'game'     ? 'block' : 'none';

    document.querySelector('.session-container').classList.toggle('game-mode', mode === 'game');
    document.querySelector('.camera-container').classList.toggle('game-active', mode === 'game');
    document.getElementById('gameStage').classList.toggle('game-active', mode === 'game');
    if (cameraActive) document.getElementById('poseBadges').style.display = mode === 'game' ? 'none' : 'flex';
    if (mode !== 'game') hideGameCanvas();

    fetch('/api/session/mode', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode }),
    }).catch(e => console.error('session/mode failed:', e));

    if (mode === 'game' && gameListCache.length === 0) loadGames();
}

// Populate the game dropdown from the registry
async function loadGames() {
    try {
        const data = await (await fetch('/api/games')).json();
        gameListCache = data.games || [];
        const sel = document.getElementById('gameType');
        sel.innerHTML = '';
        if (!gameListCache.length) {
            sel.innerHTML = '<option value="">No games available</option>';
            return;
        }
        gameListCache.forEach(g => {
            const o = document.createElement('option');
            o.value = g.id; o.textContent = g.label;
            sel.appendChild(o);
        });
        sel.value = gameListCache[0].id;
        onGameTypeChange();
    } catch (e) {
        console.error('loadGames:', e);
        showToast('Could not load games list', 'error');
    }
}

function onGameTypeChange() {
    const sel = document.getElementById('gameType');
    const g = gameListCache.find(x => x.id === sel.value);
    selectedGameId = sel.value || null;
    const descWrap = document.getElementById('gameDescriptionWrap');
    if (g && g.description) {
        document.getElementById('gameDescription').textContent = g.description;
        descWrap.style.display = 'block';
    } else {
        descWrap.style.display = 'none';
    }
}

document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('gameType')?.addEventListener('change', onGameTypeChange);
});

// Duration selector for game mode (mirrors selectDuration())
function selectGameDuration(btn, sec) {
    document.querySelectorAll('#gameDurationSelector .dur-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    gameDurationTotal = sec; // 0 = no time limit
}

// Start a game session: select the engine, push target/duration, calibrate, poll status
async function startGameSession() {
    if (!document.getElementById('sessionPatientSelect').value) {
        showToast('Please select a patient', 'error'); return;
    }
    if (!cameraActive) {
        showToast('Please start the camera first', 'error'); return;
    }
    if (!selectedGameId) {
        showToast('Please select a game', 'error'); return;
    }

    try {
        const sel = await (await fetch('/api/game/select', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ game_id: selectedGameId }),
        })).json();
        if (!sel.success) { showToast('Could not select game', 'error'); return; }

        const targetStepsRaw = parseInt(document.getElementById('targetSteps').value);
        const targetSteps = (targetStepsRaw > 0) ? targetStepsRaw : null;
        const durationSeconds = gameDurationTotal > 0 ? gameDurationTotal : null;

        await fetch('/api/game/target', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ target_steps: targetSteps, duration_seconds: durationSeconds }),
        });

        await fetch('/api/game/calibrate', { method: 'POST' });
    } catch (e) {
        console.error('startGameSession:', e);
        showToast('Could not start game session', 'error');
        return;
    }

    sessionActive     = true;
    sessionStartTime  = Date.now();
    elapsedSec        = 0;
    gameFinishHandled = false;

    document.getElementById('startSessionBtn').style.display = 'none';
    document.getElementById('stopSessionBtn').style.display  = 'inline-block';
    document.getElementById('statusText').textContent        = 'Calibrating...';
    document.getElementById('statusDot').className           = 'badge-dot recording';
    document.getElementById('recIndicator').classList.add('active');
    document.getElementById('remainingTimer').textContent    = gameDurationTotal > 0 ? formatDuration(gameDurationTotal) : '--:--';
    document.getElementById('sessionTimer').textContent      = '00:00';

    document.getElementById('gameCalibDisplay').textContent  = 'Calibrating...';
    document.getElementById('gameCalibProgress').style.width = '0%';
    document.getElementById('gameStepsDisplay').textContent  = `0 / ${targetStepsLabel()}`;
    document.getElementById('gameStepsProgress').style.width = '0%';

    showGameCanvas();
    gamePollInterval = setInterval(pollGameStatus, 120);
    showToast('Calibrating — stand still in frame...', 'success');
}

function targetStepsLabel() {
    const v = parseInt(document.getElementById('targetSteps').value);
    return v > 0 ? v : '∞';
}

// Poll the active game engine's live status and drive the HUD
async function pollGameStatus() {
    let data;
    try {
        data = await (await fetch('/api/game/status')).json();
    } catch (e) {
        console.error('pollGameStatus:', e);
        return;
    }
    if (!data || data.active_game == null) return;
    gameLiveStatus = data;   // feeds the canvas render loop (showGameCanvas/gameCanvasFrame)

    // Calibration
    if (data.calibrating && !data.calibrated) {
        document.getElementById('gameCalibDisplay').textContent  = `Calibrating... ${Math.round((data.calib_progress||0)*100)}%`;
        document.getElementById('gameCalibProgress').style.width = `${(data.calib_progress||0)*100}%`;
        document.getElementById('statusText').textContent        = 'Calibrating...';
    } else if (data.calibrated) {
        document.getElementById('gameCalibDisplay').textContent  = 'Ready';
        document.getElementById('gameCalibProgress').style.width = '100%';
        if (sessionActive && document.getElementById('statusText').textContent === 'Calibrating...') {
            document.getElementById('statusText').textContent = 'Session Active';
        }
    }

    // Steps / target progress
    const steps = data.steps || 0;
    const targetSteps = data.target_steps;
    document.getElementById('gameStepsDisplay').textContent  = `${steps} / ${targetSteps != null ? targetSteps : '∞'}`;
    document.getElementById('gameStepsProgress').style.width = targetSteps ? `${Math.min(100, (steps/targetSteps)*100)}%` : '0%';

    // Lane + crouch
    const laneNames = { 0: 'Left', 1: 'Center', 2: 'Right' };
    document.getElementById('gameLaneDisplay').textContent   = laneNames[data.lane] ?? '--';
    document.getElementById('gameCrouchDisplay').textContent = data.crouching ? 'Yes' : 'No';

    // Timers (shared elements with exercise mode)
    if (data.calibrated) {
        elapsedSec = Math.round(data.elapsed_seconds || 0);
        const m = String(Math.floor(elapsedSec/60)).padStart(2,'0');
        const s = String(elapsedSec%60).padStart(2,'0');
        document.getElementById('sessionTimer').textContent = `${m}:${s}`;
        if (data.duration_seconds != null && data.remaining_seconds != null) {
            document.getElementById('remainingTimer').textContent = formatDuration(Math.ceil(data.remaining_seconds));
        }
    }

    // Feedback / guidance message
    if (data.message) {
        const cls = data.level === 'error' ? 'error' : data.level === 'ok' ? 'success' : 'warning';
        const icon = data.level === 'ok' ? 'fa-check-circle' : data.level === 'error' ? 'fa-exclamation-circle' : 'fa-info-circle';
        document.getElementById('gameFeedbackDisplay').innerHTML =
            `<div class="feedback-message ${cls}"><i class="fas ${icon}"></i><span>${data.message}</span></div>`;
    }

    // Auto-finish (target reached or time up)
    if (data.finished && sessionActive && !gameFinishHandled) {
        gameFinishHandled = true;
        showToast(data.finish_reason === 'target' ? '🎯 Target reached!' : '⏰ Time up!', 'success');
        finishGameSession(data.finish_reason);
    }
}

// Manual stop (before target/time reached)
async function stopGameSession() {
    if (!sessionActive) return;
    try { await fetch('/api/game/stop', { method: 'POST' }); }
    catch (e) { console.error('stopGameSession:', e); }
    if (!gameFinishHandled) {
        gameFinishHandled = true;
        finishGameSession('manual');
    }
}

// Common finish path: stop polling, fetch server-tracked summary, save + show result
async function finishGameSession(reason) {
    sessionActive = false;
    clearInterval(gamePollInterval);
    gamePollInterval = null;
    hideGameCanvas();

    document.getElementById('startSessionBtn').style.display = 'inline-block';
    document.getElementById('stopSessionBtn').style.display  = 'none';
    document.getElementById('statusText').textContent        = cameraActive ? 'Connected' : 'Ready';
    document.getElementById('statusDot').className           = cameraActive ? 'badge-dot online' : 'badge-dot offline';
    document.getElementById('recIndicator').classList.remove('active');

    let summary = {};
    try { summary = await (await fetch('/api/game/summary')).json(); }
    catch (e) { console.error('game/summary:', e); }

    await saveGameSessionPayload(summary, reason);
    showGameResultModal(summary, reason);
}

// Build + persist the /api/sessions payload for a finished game session
async function saveGameSessionPayload(summary, reason) {
    const patientId = document.getElementById('sessionPatientSelect').value;
    const g = gameListCache.find(x => x.id === selectedGameId);
    const steps       = summary.steps || 0;
    const leftSteps   = summary.left_steps || 0;
    const rightSteps  = summary.right_steps || 0;
    const targetSteps = summary.target_steps;
    const durationSec = summary.elapsed_seconds || elapsedSec || 0;
    const accuracy    = targetSteps ? Math.min(100, (steps/targetSteps)*100) : 100;
    // GG (canvas game state) is still holding its last values here — hideGameCanvas()
    // (called by finishGameSession() just before this) stops the render loop but does
    // not clear GG; that only happens on the next showGameCanvas().
    const finalScore  = GG ? GG.score : 0;
    const livesLeft   = GG ? GG.lives : 3;

    const payload = {
        patient_id:          patientId,
        exercise_type:       `Game: ${g ? g.label : selectedGameId}`,
        start_time:          new Date(sessionStartTime).toISOString(),
        end_time:            new Date().toISOString(),
        duration_seconds:    Math.round(durationSec),
        total_reps:          targetSteps || steps,
        completed_reps:      steps,
        accuracy_percentage: +accuracy.toFixed(1),
        average_rom:         0,
        incorrect_movements: 0,
        stability_score:     0,
        balance_score:       0,
        movement_smoothness: 0,
        fatigue_estimation:  0,
        recovery_score:      +accuracy.toFixed(1),
        joint_angles:        [],
        exercise_results:    [],
        session_data: {
            mode:             'game',
            game_id:          selectedGameId,
            target_metric:    summary.target_metric,
            target_steps:     targetSteps,
            duration_seconds: summary.duration_seconds,
            finish_reason:    reason,
            left_steps:       leftSteps,
            right_steps:      rightSteps,
            final_score:      finalScore,
            lives_remaining:  livesLeft,
            events:           summary.events || [],
        },
    };
    return saveSessionPayload(payload);
}

// Result modal, repurposed for game stats (reuses the exercise result-modal markup)
function showGameResultModal(summary, reason) {
    const steps       = summary.steps || 0;
    const targetSteps = summary.target_steps;
    const durSec       = Math.round(summary.elapsed_seconds || elapsedSec || 0);
    const durStr        = `${String(Math.floor(durSec/60)).padStart(2,'0')}:${String(durSec%60).padStart(2,'0')}`;
    const g = gameListCache.find(x => x.id === selectedGameId);

    document.getElementById('res_duration').textContent = durStr;
    document.querySelector('#resultModal .result-stat:nth-child(2) .lbl').textContent = 'Steps';
    document.getElementById('res_accuracy').textContent  = `${steps}${targetSteps ? ' / ' + targetSteps : ''}`;
    document.querySelector('#resultModal .result-stat:nth-child(3) .lbl').textContent = 'Target Met';
    document.getElementById('res_rom').textContent       = reason === 'target' ? 'Yes' : (targetSteps ? 'No' : '—');
    document.querySelector('#resultModal .result-stat:nth-child(4) .lbl').textContent = 'Finish Reason';
    document.getElementById('res_stability').textContent = reason === 'target' ? 'Target' : reason === 'time' ? 'Time up' : 'Manual stop';

    document.getElementById('resultSubtitle').textContent =
        `${g ? g.label : 'Game'} · ${durStr} · ${document.querySelector('#sessionPatientSelect option:checked')?.textContent || ''}`;

    // No timeline chart for game sessions — hide the chart canvas area if present
    const chartWrap = document.querySelector('.result-chart-wrap');
    if (chartWrap) chartWrap.style.display = 'none';

    document.getElementById('resultModal').classList.add('active');
}

// Restore the exercise-mode result-modal labels when it's reopened after a game session
function resetResultModalLabels() {
    const labels = ['Duration', 'Avg Accuracy', 'Avg ROM', 'Avg Stability'];
    document.querySelectorAll('#resultModal .result-stat .lbl').forEach((el, i) => el.textContent = labels[i]);
    const chartWrap = document.querySelector('.result-chart-wrap');
    if (chartWrap) chartWrap.style.display = 'block';
}

// Misc
function handleLogout() {
    if (sessionActive && !confirm('Session in progress. Stop and logout?')) return;
    if (sessionActive) stopSession();
    if (cameraActive)  stopCamera();
    fetch('/api/auth/logout', { method: 'POST', credentials: 'same-origin' })
        .catch(() => {})
        .finally(() => { window.location.href = '/login'; });
}

function showToast(message, type='success', duration=3500) {
    document.querySelectorAll('.toast').forEach(t=>t.remove());
    const t = document.createElement('div');
    t.className = `toast ${type}`;
    t.innerHTML = `<i class="fas fa-${type==='success'?'check-circle':'exclamation-circle'}"></i><span>${message}</span>`;
    document.body.appendChild(t);
    setTimeout(()=>t.classList.add('show'),100);
    setTimeout(()=>{ t.classList.remove('show'); setTimeout(()=>t.remove(),300); },duration);
}

// Stop camera + release device on page unload (no WS to close anymore)
window.addEventListener('beforeunload', () => {
    if (cameraActive) {
        // best-effort: fire-and-forget, can't reliably await in beforeunload
        navigator.sendBeacon
            ? navigator.sendBeacon('/api/camera/stop')
            : fetch('/api/camera/stop', { method: 'POST', keepalive: true });
    }
});