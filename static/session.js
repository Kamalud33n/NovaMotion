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
let currentRomPeak    = 0;     // best ROM of the last ~2.5 s (what the score uses, so resting between reps is not punished)
let romHist           = [];
let currentRom        = 0;
let currentStability  = 0;
let currentSmoothness = 100;   //  real — from backend frame-jerk variance
let currentBalance    = 100;   //  real — from backend shoulder-sway variance
let currentFatigue    = 0;     //  real — from backend rep-quality decline trend
let repCount         = 0;
let repDetails       = [];     // EVERY attempt in order, correct and wrong: {done, score, rom, speed, feedback}
let liveRepRom       = null;   // highest angle of the rep that is in progress right now
let repScores        = [];     // final score of every completed rep (scored per rep on the server)
let liveRepScore     = null;   // score of the rep in progress; stays at its final value once that rep ends
let repIncomplete    = 0;      // attempts that fell back to rest without reaching the target
let goalReachedAt    = 0;      // performance.now() when the target rep count was reached (the last rep gets a moment to finish)
let repsSyncReady    = false;   // true once /api/session/reset has landed, so stale reps from the previous session can't auto-stop this one
let resetDoneAt      = 0;       // Date.now() when /api/session/reset last completed; polls SENT before this carry old-session data
let lastPollApplied  = 0;       // send-time of the newest /api/pose_data response applied (drops out-of-order responses)
let currentRecordingId = null;   // keypoint recording of the running session (server returns it from reset / calibrate)
let resetSeq         = 0;       // server session counter returned by /api/session/reset; pose data with a lower seq is from an older session
let lastPoseWsAt     = 0;       // performance.now() of the newest pose data pushed over the camera WebSocket
let lastPoseApplyAt  = -1e9;    // when the page last ran handlePoseData for a pushed message (throttle)
let poseFallbackBusy = false;   // a fallback /api/pose_data request is already in flight

let poseDetected     = false;
let focusState       = 'off';   // patient-lock state pushed by the server: off | searching | locked | holding
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
let lastGameWsAt     = 0;      // performance.now() of the newest game status pushed over the camera WebSocket
let gameFallbackBusy = false;  // a fallback /api/game/status request is already in flight

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

// Camera: the browser opens the patient's webcam (getUserMedia) and shows it live
// on <canvas id="poseStream"> at full frame rate. Frames are also sent to the server
// over a WebSocket (/ws/camera?overlay=1); the server answers with the skeleton
// lines/dots only (tiny JSON) and the browser draws them over its own live video,
// so the picture stays smooth whatever the network latency is.
// The server needs NO camera, so this works on any deployed (HTTPS) site.
function toggleCamera() {
    cameraActive ? stopCamera() : startCamera();
}

const CAM_MAX_W = 640;            // frames are downscaled to this width before sending (exercise mode)
const CAM_JPEG_QUALITY = 0.7;
const CAM_EX_MAX_W = 384;         // exercise mode (body pose): MediaPipe resizes internally anyway, smaller = faster upload/decode
const CAM_EX_JPEG_QUALITY = 0.65; // (hand grip / finger exercises keep CAM_MAX_W / CAM_JPEG_QUALITY: small joints need the pixels)
const CAM_GAME_MAX_W = 384;       // game mode: lighter frames -> faster upload + less MediaPipe/decode work on the server
const CAM_GAME_JPEG_QUALITY = 0.6;
const CAM_MIN_INTERVAL_MS = 50;   // cap at ~20 fps
const CAM_MAX_INFLIGHT = 2;       // frames sent but not answered yet (overlaps upload + server work)
const CAM_DRAW_MAX_AGE_MS = 700;  // hide a skeleton older than this (no ghost skeleton if replies stop)
const CAM_STALL_MS = 4000;        // no reply for this long -> the frame/reply was lost, send a fresh one
// Skeleton smoothing + latency compensation. The skeleton comes back from the server a few
// hundred ms after the frame it describes, while the video is drawn live, so it trails fast
// movements. We (1) smooth it with a One-Euro filter (steady when still, low lag when moving)
// and (2) push it forward along its own velocity by roughly the age of the frame.
const SK_MIN_CUTOFF   = 0.7;    // Hz; lower = steadier when the patient is still
const SK_BETA         = 10.0;   // higher = less lag during fast movement
const SK_D_CUTOFF     = 5.0;    // Hz; smoothing of the velocity estimate
const SK_LEAD_FACTOR  = 0.85;   // share of the frame's age to compensate for
const SK_LEAD_MAX_S   = 0.35;   // never predict further ahead than this
const SK_MAX_OFFSET   = 0.15;   // max predicted shift per coordinate (fraction of the frame)
const SK_VEL_DEADZONE = 0.08;   // units/s below which we do not extrapolate (kills jitter)
const SK_SNAP_DIST    = 0.30;   // a joint jumping further than this = different skeleton, no blending
const SK_VIEW_TAU_S   = 0.035;  // on-screen easing between render frames
// Labels of built-in laptop cameras; anything else is treated as an external USB webcam.
const INTERNAL_CAM_RE = /integrated|built-?in|internal|facetime|laptop|front/i;

let camStream = null, camWs = null, camVideo = null, camCanvas = null, camCtx = null;
let camPrevUrl = null, camLastSend = 0, camSendTimer = null;
let camInFlight = 0, camWatchdog = null;   // frames sent but not answered yet / stall timer
let camEncoding = false, camRaf = null;
let camSentAt = [];                          // capture time of every frame still waiting for its reply
let camStats = { n: 0, sum: 0, max: 0, t0: 0 };   // round-trip time log (console, every 5 s)
let skX = null, skDX = null, skShown = null;       // filtered skeleton, its velocity, what is on screen
let skN = { l: 0, d: 0 }, skSentAt = 0, skArrive = 0, skLastRender = 0;

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
    // Game mode uses smaller, lighter frames: on the deployed site the upload + server
    // decode is what limits the frame rate.
    const gameMode = sessionMode === 'game';
    const exName   = (document.getElementById('exerciseType')?.value || '').toLowerCase();
    const handEx   = exName.includes('grip') || exName.includes('finger');
    const maxW     = gameMode ? CAM_GAME_MAX_W : handEx ? CAM_MAX_W : CAM_EX_MAX_W;
    const quality  = gameMode ? CAM_GAME_JPEG_QUALITY : handEx ? CAM_JPEG_QUALITY : CAM_EX_JPEG_QUALITY;
    // Keep the camera's real aspect ratio (a stretched frame would distort joint angles).
    const w = Math.min(maxW, camVideo.videoWidth);
    const h = Math.round(w * camVideo.videoHeight / camVideo.videoWidth);
    if (camCanvas.width !== w || camCanvas.height !== h) { camCanvas.width = w; camCanvas.height = h; }
    const capturedAt = performance.now();
    camCtx.drawImage(camVideo, 0, 0, w, h);
    camEncoding = true;
    camCanvas.toBlob(blob => {
        camEncoding = false;
        if (!cameraActive || !camWs || camWs.readyState !== WebSocket.OPEN) return;
        if (blob) {
            camLastSend = performance.now();
            camSentAt.push(capturedAt);
            camInFlight++;
            camArmWatchdog();
            camWs.send(blob);
            if (camInFlight < CAM_MAX_INFLIGHT) camScheduleNext();
        } else {
            camScheduleNext();   // encoder gave nothing (tab hidden / busy): retry instead of stalling for good
        }
    }, 'image/jpeg', quality);
}

// If a frame or its reply is lost (proxy hiccup, slow server), nothing would ever trigger
// the next send and the stream would freeze for good. This sends a fresh frame instead.
function camArmWatchdog() {
    clearTimeout(camWatchdog);
    camWatchdog = setTimeout(() => {
        camWatchdog = null;
        if (!cameraActive || !camWs || camWs.readyState !== WebSocket.OPEN) return;
        console.warn('camera stream stalled - sending a fresh frame');
        camInFlight = 0;
        camSentAt.length = 0;
        if (camSendTimer) { clearTimeout(camSendTimer); camSendTimer = null; }
        camSendFrame();
    }, CAM_STALL_MS);
}

// The server answered one frame (annotated image, or an error for it).
function camReplyReceived() {
    const sentAt = camSentAt.shift();
    if (sentAt !== undefined) {
        const now = performance.now(), rtt = now - sentAt;
        if (!camStats.t0) camStats.t0 = now;
        camStats.n++; camStats.sum += rtt; camStats.max = Math.max(camStats.max, rtt);
        if (now - camStats.t0 >= 5000) {
            console.info(`[cam] ${(camStats.n / ((now - camStats.t0) / 1000)).toFixed(1)} replies/s, ` +
                         `frame-to-skeleton delay avg ${Math.round(camStats.sum / camStats.n)} ms, max ${Math.round(camStats.max)} ms`);
            camStats = { n: 0, sum: 0, max: 0, t0: now };
        }
    }
    camInFlight = Math.max(0, camInFlight - 1);
    clearTimeout(camWatchdog);
    camWatchdog = null;
    if (camInFlight > 0) camArmWatchdog();
    if (camInFlight < CAM_MAX_INFLIGHT) camScheduleNext();
}

// Real frame rate = annotated frames received per second.
function countFrameForFps() {
    fpsCounter++;
    const now = Date.now();
    if (now - fpsWindowStart >= 1000) {
        document.getElementById('fpsDisplay').textContent = fpsCounter;
        fpsCounter = 0;
        fpsWindowStart = now;
    }
}

// Send the next frame only after the previous annotated one came back (no lag build-up).
function camScheduleNext() {
    if (camSendTimer || camEncoding) return;
    const wait = Math.max(0, CAM_MIN_INTERVAL_MS - (performance.now() - camLastSend));
    camSendTimer = setTimeout(camSendFrame, wait);
}

function skAlpha(cutoff, dt) {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
}

// A new skeleton arrived from the server for the frame captured at `capturedAt`.
function camSkelUpdate(msg, capturedAt) {
    const nl = msg.l.length, nd = msg.d.length, n = nl * 4 + nd * 2;
    const raw = new Float64Array(n);
    let i = 0;
    for (const l of msg.l) { raw[i++] = l[0]; raw[i++] = l[1]; raw[i++] = l[2]; raw[i++] = l[3]; }
    for (const d of msg.d) { raw[i++] = d[0]; raw[i++] = d[1]; }

    // Blend only if it is the same set of joints as last time and nothing jumped.
    let same = skX && skN.l === nl && skN.d === nd;
    if (same) {
        for (let k = 0; k < n; k++) {
            if (Math.abs(raw[k] - skX[k]) > SK_SNAP_DIST) { same = false; break; }
        }
    }
    if (!same) {
        skX = raw;
        skDX = new Float64Array(n);
        skShown = null;                               // snap on screen too
    } else {
        const dt = Math.min(0.5, Math.max(0.01, (capturedAt - skSentAt) / 1000));
        const aD = skAlpha(SK_D_CUTOFF, dt);
        for (let k = 0; k < n; k++) {
            const dx  = (raw[k] - skX[k]) / dt;
            const edx = skDX[k] + aD * (dx - skDX[k]);
            const cutoff = SK_MIN_CUTOFF + SK_BETA * Math.abs(edx);
            skX[k] += skAlpha(cutoff, dt) * (raw[k] - skX[k]);
            skDX[k] = edx;
        }
    }
    skN = { l: nl, d: nd };
    skSentAt = capturedAt;
    skArrive = performance.now();
}

// The skeleton to draw right now (filtered, moved forward by the frame's age, eased), or null.
function camSkelFrame(now) {
    if (!skX || now - skArrive > CAM_DRAW_MAX_AGE_MS) return null;
    const n = skX.length;
    let snap = false;
    if (!skShown || skShown.length !== n) { skShown = new Float64Array(n); snap = true; }
    const dtFrame = skLastRender ? Math.min(0.1, (now - skLastRender) / 1000) : 0.016;
    skLastRender = now;
    const k = snap ? 1 : 1 - Math.exp(-dtFrame / SK_VIEW_TAU_S);
    const lead = Math.min(SK_LEAD_FACTOR * Math.max(0, now - skSentAt) / 1000, SK_LEAD_MAX_S);
    for (let i = 0; i < n; i++) {
        let v = skDX[i];
        v = Math.abs(v) < SK_VEL_DEADZONE ? 0 : v - Math.sign(v) * SK_VEL_DEADZONE;
        const off = Math.max(-SK_MAX_OFFSET, Math.min(SK_MAX_OFFSET, v * lead));
        skShown[i] += (skX[i] + off - skShown[i]) * k;
    }
    return skShown;
}

// Draws the live local video (mirrored, like the server's frame) and the latest skeleton.
function camRenderLoop() {
    camRaf = requestAnimationFrame(camRenderLoop);
    const cv = document.getElementById('poseStream');
    if (!cv || !camVideo || !camVideo.videoWidth) return;
    const w = Math.min(960, camVideo.videoWidth);
    const h = Math.round(w * camVideo.videoHeight / camVideo.videoWidth);
    if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
    const c = cv.getContext('2d');
    c.save();
    c.translate(w, 0);
    c.scale(-1, 1);
    c.drawImage(camVideo, 0, 0, w, h);
    c.restore();
    const sk = camSkelFrame(performance.now());
    if (!sk) return;
    c.lineWidth = Math.max(2, w / 320);
    c.strokeStyle = '#00a0ff';
    c.lineCap = 'round';
    c.beginPath();
    let i = 0;
    for (let n = 0; n < skN.l; n++, i += 4) {
        c.moveTo(sk[i] * w, sk[i + 1] * h);
        c.lineTo(sk[i + 2] * w, sk[i + 3] * h);
    }
    c.stroke();
    const rad = Math.max(3, w / 200);
    c.fillStyle = '#78e600';
    c.strokeStyle = '#ffffff';
    c.lineWidth = 1;
    for (let n = 0; n < skN.d; n++, i += 2) {
        c.beginPath();
        c.arc(sk[i] * w, sk[i + 1] * h, rad, 0, 6.2832);
        c.fill();
        c.stroke();
    }
}

function camCleanup() {
    if (camRaf) { cancelAnimationFrame(camRaf); camRaf = null; }
    skX = skDX = skShown = null;
    skN = { l: 0, d: 0 };
    skLastRender = 0;
    camSentAt.length = 0;
    camStats = { n: 0, sum: 0, max: 0, t0: 0 };
    camEncoding = false;
    if (camSendTimer) { clearTimeout(camSendTimer); camSendTimer = null; }
    if (camWatchdog) { clearTimeout(camWatchdog); camWatchdog = null; }
    camInFlight = 0;
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
    const ws = new WebSocket(`${wsProto}://${location.host}/ws/camera?overlay=1`);
    camWs = ws;

    ws.onmessage = (ev) => {
        if (typeof ev.data === 'string') {
            let msg = null;
            try { msg = JSON.parse(ev.data); } catch (e) { /* not JSON */ }
            if (msg && msg.type === 'game') {        // live game status, pushed right before each annotated frame
                lastGameWsAt = performance.now();
                applyFocusState(msg.focus);
                applyGameStatus(msg.status);
                return;                              // the frame that follows triggers the next send
            }
            if (msg && msg.type === 'pose') {        // exercise-mode joint angles / reps / scores, same idea
                lastPoseWsAt = performance.now();
                applyPoseMessage(msg.data);
                return;
            }
            if (msg && msg.type === 'draw') {        // skeleton for one frame = the reply to that frame
                camSkelUpdate(msg, camSentAt.length ? camSentAt[0] : performance.now());
                countFrameForFps();
                camReplyReceived();
                return;
            }
            console.warn('camera ws:', ev.data);     // server-side error for one frame, keep going
            camReplyReceived();
            return;
        }
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
        img.style.display = 'block';
        if (!camRaf) camRaf = requestAnimationFrame(camRenderLoop);
        document.getElementById('cameraPlaceholder').style.display = 'none';
        document.getElementById('cameraStatusText').textContent    = 'Connected';
        document.getElementById('cameraStatus').classList.add('active');
        document.getElementById('poseBadges').style.display        = sessionMode === 'game' ? 'none' : 'flex';
        document.getElementById('statusText').textContent          = 'Connected';
        document.getElementById('statusDot').className             = 'badge-dot online';

        pollFailCount  = 0;
        fpsCounter     = 0;
        fpsWindowStart = Date.now();

        // The server keeps each user's Exercise/Game mode in memory for 30 min
        // after the last use, but this page always loads in "Exercise". If the
        // last visit used Game mode, the server still drew the FULL body skeleton
        // instead of the selected exercise. Push the page's mode first.
        fetch('/api/session/mode', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode: sessionMode }),
        }).catch(e => console.error('session/mode sync failed:', e));

        // Tell backend which exercise is active so it draws only relevant joints
        syncExerciseType();

        // Joint-angle/detection data normally arrives over this same WebSocket with every
        // frame (see ws.onmessage). This slow poll is only a fallback for when the socket
        // goes quiet - it also detects a lost connection.
        poseInterval = setInterval(pollPoseData, 1000);

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
                mode:          sessionMode,   // keeps server Exercise/Game mode in step with the page
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
    applyFocusState('off');
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
// Pose data pushed over the camera WebSocket (one message per annotated frame).
// The DOM only needs ~10 updates/s, so extra messages are skipped.
function applyPoseMessage(data) {
    if (!data || sessionMode === 'game') return;
    pollFailCount = 0;
    const now = performance.now();
    if (now - lastPoseApplyAt < 100) return;
    lastPoseApplyAt = now;
    // A message produced before the last /api/session/reset (still on its way) carries the
    // previous session's reps: it must never drive rep counting for this session.
    handlePoseData(data, (data.seq || 0) >= resetSeq);
}

// Fallback poll (every 1 s) - does nothing while the WebSocket is delivering pose data.
async function pollPoseData() {
    if (sessionMode === 'game') return;   // game mode has its own status feed
    if (performance.now() - lastPoseWsAt < 800) return;   // pushed data is fresh
    if (poseFallbackBusy) return;                         // never stack requests on a slow link
    poseFallbackBusy = true;
    try {
        const sentAt = Date.now();
        const res = await fetch('/api/pose_data');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        pollFailCount = 0;
        lastPollOkTime = Date.now();

        // On a slow/remote connection responses can arrive late or out of order.
        // Drop anything older than what we've already applied, and never let a
        // poll that was sent before the reset completed drive rep counting.
        if (sentAt < lastPollApplied) return;
        lastPollApplied = sentAt;
        handlePoseData(data, sentAt >= resetDoneAt && (data.seq || 0) >= resetSeq);
    } catch(e) {
        pollFailCount++;
        console.error('pollPoseData:', e);
        if (pollFailCount >= 5) {
            document.getElementById('statusText').textContent = 'Connection lost';
            document.getElementById('statusDot').className    = 'badge-dot error';
        }
    } finally {
        poseFallbackBusy = false;
    }
}

// Focus mode (patient lock) badge
const FOCUS_LABELS = {
    locked:    'Focus locked',
    holding:   'Someone in frame - holding',
    searching: 'Locking on patient...'
};

function ensureFocusBadge() {
    let badge = document.getElementById('focusBadge');
    if (badge) return badge;
    const host = document.getElementById('noPoseWarning');
    if (!host || !host.parentElement) return null;

    badge = document.createElement('div');
    badge.id = 'focusBadge';
    badge.className = 'focus-badge';
    const label = document.createElement('span');
    label.id = 'focusBadgeText';
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.textContent = 'Re-lock';
    btn.addEventListener('click', relockFocus);
    badge.append(label, btn);
    host.parentElement.appendChild(badge);
    return badge;
}

function applyFocusState(state) {
    focusState = FOCUS_LABELS[state] ? state : 'off';
    const badge = ensureFocusBadge();
    if (!badge) return;

    const visible = cameraActive && focusState !== 'off';
    badge.style.display = visible ? 'flex' : 'none';
    if (!visible) return;

    badge.className = `focus-badge ${focusState}`;
    const text = document.getElementById('focusBadgeText');
    if (text.textContent !== FOCUS_LABELS[focusState]) text.textContent = FOCUS_LABELS[focusState];
}

async function relockFocus() {
    try {
        await fetch('/api/focus/relock', { method: 'POST' });
        showToast('Re-locking on patient...', 'success');
    } catch (e) {
        console.error('focus relock failed:', e);
    }
}

// Pose data handler 
function handlePoseData(data, fresh = true) {
    poseDetected = !!data.detected;
    applyFocusState(data.focus);

    document.getElementById('noPoseWarning').style.display  = (poseDetected || focusState === 'holding') ? 'none' : 'block';
    document.getElementById('detectionStatus').textContent  = poseDetected ? 'Yes' : 'No';
    document.getElementById('detectionStatus').style.color  = poseDetected ? '#1D8A6D' : '#D9503E';
    if (sessionActive && repsSyncReady && fresh && data.reps != null) {
        repCount = data.reps;
        // Per-rep scores come straight from the server (see MetricsState.get_rep_scoring)
        if (Array.isArray(data.rep_scores)) repScores = data.rep_scores;
        if (Array.isArray(data.rep_details)) repDetails = data.rep_details;
        liveRepRom = data.rep_live_rom != null ? data.rep_live_rom : null;
        liveRepScore  = data.rep_live != null ? data.rep_live : null;
        repIncomplete = data.rep_incomplete || 0;
        renderRepScores();
        const repGoal = parseInt(document.getElementById('targetReps').value);
        if (sessionMode === 'exercise' && repGoal > 0 && repCount >= repGoal) {
            // A rep is counted the moment it reaches the target but is only scored once the
            // arm comes back down, so give the last rep a moment to finish before ending.
            if (!goalReachedAt) goalReachedAt = performance.now();
            const lastRepScored = repScores.length >= repGoal;
            if (lastRepScored || performance.now() - goalReachedAt > 4000) {
                updateAnalytics(data.primary_angle, parseInt(document.getElementById('targetRom').value) || 90);
                showToast('🎯 Target reps completed! Session ending...', 'success');
                stopSession();
                return;
            }
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
    repCount = currentAccuracy = currentRom = currentRomPeak = 0;
    romHist = [];
    currentStability  = 100;
    currentSmoothness = 100;
    currentBalance    = 100;
    currentFatigue    = 0;
    _timeline = [];
    repScores = []; repDetails = [];
    liveRepScore = null;
    repIncomplete = 0;
    goalReachedAt = 0; window._overtimeToastShown = false;
    renderRepScores();

    // Reset server-side rep counter + stability buffer for this new
    // session, and make sure backend has the latest exercise/ROM target
    repsSyncReady = false;
    currentRecordingId = null;
    syncExerciseType();
    fetch('/api/session/reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            patient_id:    document.getElementById('sessionPatientSelect').value,
            exercise_type: document.getElementById('exerciseType').value,
            target_rom:    parseFloat(document.getElementById('targetRom').value) || 90,
            side:          selectedSide,
        }),
    })
        .then(r => r.json())
        .then(j => {
            if (j && j.seq != null) resetSeq = j.seq;
            currentRecordingId = (j && j.recording_id) || null;
        })   // pose data older than this belongs to the previous session
        .catch(e => console.error('session reset failed:', e))
        // Polls are tagged with their send time (see pollPoseData); only polls
        // sent after this moment are trusted for rep counting, so a fixed delay
        // (which broke on slow production links) is no longer needed.
        .finally(() => { resetDoneAt = Date.now(); repsSyncReady = true; });

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
            const goal = parseInt(document.getElementById('targetReps').value);
            if (sessionMode === 'exercise' && goal > 0 && repCount < goal) {
                // Not enough correct reps yet: keep the session running (timer parked at 00:00)
                // until the target is reached, or the user presses Stop.
                countdownLeft = 0;
                if (!window._overtimeToastShown) {
                    window._overtimeToastShown = true;
                    showToast(`⏱ Time is over, but only ${repCount}/${goal} correct reps done — keep going!`, 'info', 5000);
                }
            } else {
                showToast('⏰ Time up! Session ending...', 'success');
                stopSession();
            }
        }
    }, 1000);

    // Record timeline every 5s
    _timelineTimer = setInterval(() => {
        if (sessionActive) {
            _timeline.push({
                t:      elapsedSec,
                acc:    +currentAccuracy.toFixed(1),
                rom:    +currentRomPeak.toFixed(1),
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
        const nowMs = performance.now();
        romHist.push({ t: nowMs, v: currentRom });
        while (romHist.length && nowMs - romHist[0].t > 2500) romHist.shift();
        currentRomPeak = romHist.reduce((m, r) => Math.max(m, r.v), 0);
        // Accuracy = score of the rep in progress (or of the last finished rep), scored per
        // rep on the server: 60% range reached in THAT rep + 20% stability + 20% smoothness
        // during that rep. Nothing is averaged over time any more.
        currentAccuracy = liveRepScore != null ? liveRepScore : 0;
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

function scoreColor(v) {
    return v >= 85 ? '#1D8A6D' : v >= 70 ? '#D68F35' : '#D9503E';
}

// Live list of finished reps ("Rep 1 87%"), filled from the server's per-rep scores
let _repListKey = null;
function renderRepScores() {
    const box = document.getElementById('repScoreList');
    if (!box) return;
    const key = repScores.join(',') + '|' + repIncomplete;
    if (key === _repListKey) return;          // runs ~10x/s: only touch the DOM when something changed
    _repListKey = key;
    box.replaceChildren();
    const chip = (text, bg) => {
        const el = document.createElement('span');
        el.textContent = text;
        el.style.cssText = `display:inline-block;margin:2px 4px 2px 0;padding:2px 10px;border-radius:12px;` +
                           `font-size:12px;font-weight:600;color:#fff;background:${bg}`;
        box.appendChild(el);
    };
    repScores.forEach((v, i) => chip(`Rep ${i + 1}: ${Math.round(v)}%`, scoreColor(v)));
    if (repIncomplete > 0) chip(`Incomplete: ${repIncomplete}`, '#8a96a3');
}

// Scores to report when the session ends. If it is stopped while the last counted rep is
// still in progress, that rep's live score is used for it.
function getFinalRepScores() {
    const list = repScores.slice();
    if (list.length < repCount && liveRepScore != null) list.push(liveRepScore);
    return list;
}

// Rows for the report: every attempt in order (correct AND wrong). A wrong attempt is saved
// with is_completed=false so the PDF shows it as "No". If the session is stopped while the last
// counted rep is still in progress, that rep is added from its live score / live angle.
function getFinalAttempts() {
    const list = repDetails.map(d => ({ done: !!d.done, score: d.score || 0, rom: d.rom || 0,
                                        speed: d.speed || 0, feedback: d.feedback || '',
                                        stability: d.stability ?? null, smoothness: d.smoothness ?? null,
                                        balance: d.balance ?? null, duration: d.duration || 0,
                                        joints: d.joints || {} }));
    const correct = list.filter(d => d.done).length;
    if (correct < repCount && liveRepScore != null)
        list.push({ done: true, score: liveRepScore, rom: liveRepRom || 0, speed: 0, feedback: 'Stopped during this rep',
                    stability: null, smoothness: null, balance: null, duration: 0, joints: {} });
    return list;
}

// Session accuracy = average of the per-rep scores. Balance has no reps, so it keeps the
// old steadiness timeline.
function sessionAccuracy() {
    const scores = getFinalRepScores();
    if (!isBalanceExercise() && scores.length) return scores.reduce((t, v) => t + v, 0) / scores.length;
    return _timeline.length ? _timeline.reduce((t, r) => t + r.acc, 0) / _timeline.length : currentAccuracy;
}

function buildRepChart(ctx, scores) {
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: scores.map((v, i) => [`Rep ${i + 1}`, `${Math.round(v)}%`]),
            datasets: [{ label: 'Rep score %', data: scores, backgroundColor: scores.map(scoreColor),
                         borderRadius: 6, maxBarThickness: 46 }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: { backgroundColor: 'rgba(255,255,255,0.95)', titleColor: '#1a2332', bodyColor: '#5a6b7c',
                           borderColor: 'rgba(0,0,0,0.08)', borderWidth: 1, cornerRadius: 10, padding: 10 }
            },
            scales: {
                y: { beginAtZero: true, max: 100, grid: { color: 'rgba(0,0,0,0.04)' }, ticks: { font: { size: 10 } } },
                x: { grid: { display: false }, ticks: { font: { size: 11 } } }
            }
        }
    });
}

function updateFeedback(acc, rom) {
    const t = parseInt(document.getElementById('targetRom').value) || 90;
    let msg, icon, cls;
    if (acc>85&&rom>t*0.8)      { msg='Excellent form! Keep it up!';            icon='fa-check-circle';       cls='success'; }
    else if (acc>70&&rom>t*0.6) { msg='Good progress. Focus on range of motion.'; icon='fa-thumbs-up';         cls='warning'; }
    else if (acc>50)            { msg='Needs improvement. Adjust your posture.';  icon='fa-exclamation-triangle'; cls='warning'; }
    else                        { msg='Consult your therapist for guidance.';     icon='fa-exclamation-circle'; cls='error'; }
    const fb = document.getElementById('feedbackDisplay');
    const key = cls + '|' + msg;
    if (fb.dataset.key === key) return;      // runs ~10x/s now - only rebuild the DOM when the message changes
    fb.dataset.key = key;
    fb.innerHTML =
        `<div class="feedback-message ${cls}"><i class="fas ${icon}"></i><span>${msg}</span></div>`;
}

// Result modal 
function showResultModal() {
    const finalScores = getFinalRepScores();
    const useReps = !isBalanceExercise() && finalScores.length > 0;
    const avgAcc  = sessionAccuracy().toFixed(1);
    const avgRom  = _timeline.length ? (_timeline.reduce((s,r)=>s+r.rom, 0) / _timeline.length).toFixed(1) : currentRom.toFixed(1);
    const avgStab = _timeline.length ? (_timeline.reduce((s,r)=>s+r.stab,0) / _timeline.length).toFixed(1) : currentStability.toFixed(1);

    const durSec  = elapsedSec;
    const durStr  = `${String(Math.floor(durSec/60)).padStart(2,'0')}:${String(durSec%60).padStart(2,'0')}`;

    document.getElementById('res_duration').textContent = durStr;
    document.getElementById('res_accuracy').textContent  = `${avgAcc}%`;
    document.querySelector('#resultModal .result-stat:nth-child(2) .lbl').textContent = useReps ? 'Avg Rep Score' : 'Avg Accuracy';
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
    resultChartInst = useReps ? buildRepChart(ctx, finalScores) : new Chart(ctx, {
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
    window.SessionRating?.reset();
    resetResultModalLabels();
    if (resultChartInst) { resultChartInst.destroy(); resultChartInst=null; }
    // Reset UI
    currentAccuracy=currentRom=currentStability=repCount=elapsedSec=0;
    currentSmoothness=100; currentBalance=100; currentFatigue=0;
    repScores = []; repDetails = []; liveRepRom = null; liveRepScore = null; repIncomplete = 0; goalReachedAt = 0;
    renderRepScores();
    document.getElementById('sessionTimer').textContent    = '00:00';
    document.getElementById('remainingTimer').textContent  = '--:--';
    document.getElementById('accuracyDisplay').textContent = '0%';
    document.getElementById('romDisplay').textContent      = '0°';
    document.getElementById('stabilityDisplay').textContent= '0%';
    document.getElementById('repCounter').textContent      = '0 / 0';
    ['accuracyProgress','romProgress','stabilityProgress','repProgress']
        .forEach(id => document.getElementById(id).style.width='0%');
    document.getElementById('feedbackDisplay').dataset.key = '';   // let updateFeedback() rebuild it next time
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

    // Every session number below is built from the PER-REP results (correct reps only),
    // not from the live frame-by-frame readings. The live timeline is only a fallback for
    // sessions that have no reps at all (Balance exercise, or stopped before the first rep).
    const attemptsForSave = isBalanceExercise() ? [] : getFinalAttempts();
    const okAtt = attemptsForSave.filter(a => a.done);
    const mean = (rows) => { const v = rows.filter(x => x != null); return v.length ? v.reduce((t, x) => t + x, 0) / v.length : null; };

    const avgAcc    = sessionAccuracy();
    const avgRom    = mean(okAtt.map(a => a.rom))        ?? avg('rom',    currentRom);
    const avgStab   = mean(okAtt.map(a => a.stability))  ?? avg('stab',   currentStability);
    const avgSmooth = mean(okAtt.map(a => a.smoothness)) ?? avg('smooth', currentSmoothness);
    // Hand grip / finger: hands-only frame, no balance signal -> 0 = "N/A" in the report
    const avgBal    = isFingerExercise() ? 0 : (mean(okAtt.map(a => a.balance)) ?? avg('bal', currentBalance));
    // Fatigue = how much rep quality dropped from the first half of the reps to the second half
    let avgFat = avg('fat', currentFatigue);
    if (okAtt.length >= 4) {
        const sc = okAtt.map(a => a.score), half = Math.floor(sc.length / 2);
        const early = sc.slice(0, half).reduce((t, x) => t + x, 0) / half;
        const late  = sc.slice(half).reduce((t, x) => t + x, 0) / (sc.length - half);
        avgFat = Math.min(100, Math.max(0, early - late) * 2.5);
    } else if (okAtt.length) {
        avgFat = 0;
    }

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

    const scoresForSave   = isBalanceExercise() ? [] : getFinalRepScores();

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
        incorrect_movements: repIncomplete,      // real wrong attempts (fell back without reaching the target)
        joint_angles:        jointAngles,
        exercise_results:    attemptsForSave.map((a, i) => ({
            exercise_name:     exerciseType,
            repetition_number: i + 1,
            accuracy:          +a.score.toFixed(1),
            rom_achieved:      +a.rom.toFixed(1),
            speed:             +a.speed.toFixed(1),
            feedback:          a.feedback,
            is_completed:      a.done,
        })),
        session_data:        { timeline: _timeline, rep_scores: scoresForSave, incomplete_attempts: repIncomplete,
                               rep_joints: attemptsForSave.map(a => a.joints),   // per rep: degrees each joint moved
                               rep_stats:  attemptsForSave.map(a => ({ stab: a.stability, smooth: a.smoothness,
                                                                       bal: a.balance, secs: a.duration })),
                               recording_id: currentRecordingId },
    };

    return saveSessionPayload(payload, true);
}

// Actually performs the save + handles success/failure UI. Returns true/false.
async function saveSessionPayload(payload, track = false) {
    if (track) window.SessionRating?.saveStarted();
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
            if (track) window.SessionRating?.saveFailed();
            return false;
        }

        // Success — make sure no stale local backup is replayed on top of it.
        clearStashedSession(payload);
        showToast('Session saved', 'success');
        if (track) {
            let saved = {};
            try { saved = await res.json(); } catch (_) { /* no id in the reply */ }
            window.SessionRating?.saved(saved.session_id);
        }
        return true;
    } catch (e) {
        console.error('saveSession network error:', e);
        stashFailedSession(payload);
        showToast('No connection — session saved locally, will retry', 'error', 5000);
        if (track) window.SessionRating?.saveFailed();
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
    syncGameStageView();
    const descWrap = document.getElementById('gameDescriptionWrap');
    if (g && g.description) {
        document.getElementById('gameDescription').textContent = g.description;
        descWrap.style.display = 'block';
    } else {
        descWrap.style.display = 'none';
    }
    if (!g) return;
    document.getElementById('gameTargetLabel').textContent = g.target_label;
    document.getElementById('targetSteps').value = g.target_default;
    loadGameCfg(g);
    renderGameOptions();
    renderGameHud(g);
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

        const cfg = gameConfigValues();
        if (Object.keys(cfg).length) {
            await fetch('/api/game/config', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(cfg),
            });
        }

        currentRecordingId = null;
        const cal = await (await fetch('/api/game/calibrate', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ patient_id: document.getElementById('sessionPatientSelect').value }),
        })).json();
        currentRecordingId = (cal && cal.recording_id) || null;
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
    document.getElementById('gameTargetDisplay').textContent = `0 / ${targetStepsLabel()}`;
    document.getElementById('gameTargetProgress').style.width = '0%';

    showGameCanvas();
    lastGameWsAt = 0;
    gamePollInterval = setInterval(pollGameStatus, 1000);   // fallback only - normal updates arrive over the camera WebSocket
    showToast('Calibrating — stand still in frame...', 'success');
}

function targetStepsLabel() {
    const v = parseInt(document.getElementById('targetSteps').value);
    return v > 0 ? v : '∞';
}

// Cheap DOM writers: the status now arrives with every camera frame (~15-20x/s), so
// only touch the DOM when a value actually changed.
function hudText(id, v) {
    const el = document.getElementById(id);
    if (el && el.textContent !== v) el.textContent = v;
}
function hudWidth(id, pct) {
    const el = document.getElementById(id);
    const v = `${pct}%`;
    if (el && el.style.width !== v) el.style.width = v;
}

// Drive the HUD / canvas from one game status object. Called for every status the
// server pushes over the camera WebSocket (and by the slow fallback poll below).
function applyGameStatus(data) {
    if (!sessionActive || sessionMode !== 'game') return;
    if (!data || data.active_game == null) return;
    gameLiveStatus = data;   // feeds the canvas render loop (showGameCanvas/gameCanvasFrame)

    // "No pose" overlay (this used to come from the /api/pose_data poll)
    poseDetected = data.person !== 'none';
    const warn = document.getElementById('noPoseWarning');
    if (warn) warn.style.display = (poseDetected || focusState === 'holding') ? 'none' : 'block';

    // Calibration
    if (data.calibrating && !data.calibrated) {
        const pct = Math.round((data.calib_progress || 0) * 100);
        hudText('gameCalibDisplay', `Calibrating... ${pct}%`);
        hudWidth('gameCalibProgress', pct);
        hudText('statusText', 'Calibrating...');
    } else if (data.calibrated) {
        hudText('gameCalibDisplay', 'Ready');
        hudWidth('gameCalibProgress', 100);
        if (sessionActive && document.getElementById('statusText').textContent === 'Calibrating...') {
            document.getElementById('statusText').textContent = 'Session Active';
        }
    }

    updateGameHud(data);

    // Timers (shared elements with exercise mode)
    if (data.calibrated) {
        elapsedSec = Math.round(data.elapsed_seconds || 0);
        const m = String(Math.floor(elapsedSec/60)).padStart(2,'0');
        const s = String(elapsedSec%60).padStart(2,'0');
        hudText('sessionTimer', `${m}:${s}`);
        if (data.duration_seconds != null && data.remaining_seconds != null) {
            hudText('remainingTimer', formatDuration(Math.ceil(data.remaining_seconds)));
        }
    }

    // Feedback / guidance message
    if (data.message) {
        const fb = document.getElementById('gameFeedbackDisplay');
        const key = `${data.level}|${data.message}`;
        if (fb && fb.dataset.key !== key) {
            fb.dataset.key = key;
            const cls = data.level === 'error' ? 'error' : data.level === 'ok' ? 'success' : 'warning';
            const icon = data.level === 'ok' ? 'fa-check-circle' : data.level === 'error' ? 'fa-exclamation-circle' : 'fa-info-circle';
            fb.innerHTML =
                `<div class="feedback-message ${cls}"><i class="fas ${icon}"></i><span>${data.message}</span></div>`;
        }
    }

    // Auto-finish (target reached or time up)
    if (data.finished && sessionActive && !gameFinishHandled) {
        gameFinishHandled = true;
        showToast(data.finish_reason === 'target' ? '🎯 Target reached!' : '⏰ Time up!', 'success');
        finishGameSession(data.finish_reason);
    }
}

// Fallback poll (every 1 s). Normally the server pushes the game status over the camera
// WebSocket with every frame, so this does nothing; it only kicks in if frames stall, so
// the time limit / finish state is still noticed.
async function pollGameStatus() {
    if (!sessionActive || sessionMode !== 'game') return;
    if (performance.now() - lastGameWsAt < 800) return;   // WebSocket status is fresh
    if (gameFallbackBusy) return;                         // never stack requests on a slow link
    gameFallbackBusy = true;
    let data;
    try {
        data = await (await fetch('/api/game/status')).json();
    } catch (e) {
        console.error('pollGameStatus:', e);
        return;
    } finally {
        gameFallbackBusy = false;
    }
    applyGameStatus(data);
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

    // The server-side summary IS the round's data. If it can't be read we must
    // not save: an empty summary would store a row full of zeros and still show
    // "Session saved". Retry once, and only accept a real, non-empty summary
    // (a 401/500 body is a JSON object too, so res.ok is checked first).
    let summary = null;
    for (let attempt = 0; attempt < 2 && !summary; attempt++) {
        try {
            const res = await fetch('/api/game/summary', { credentials: 'same-origin' });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();
            if (data && typeof data === 'object' && Object.keys(data).length) summary = data;
            else throw new Error('empty summary (no active game on the server)');
        } catch (e) {
            console.error(`game/summary (attempt ${attempt + 1}):`, e);
            if (attempt === 0) await new Promise(r => setTimeout(r, 500));
        }
    }
    if (!summary) {
        showToast('Could not read the game results from the server - this round was NOT saved. '
                + 'Check your connection / login and play the round again.', 'error', 8000);
        return;
    }

    await saveGameSessionPayload(summary, reason);
    showGameResultModal(summary, reason);
}

// Build + persist the /api/sessions payload for a finished game session
async function saveGameSessionPayload(summary, reason) {
    const patientId = document.getElementById('sessionPatientSelect').value;
    const g = gameListCache.find(x => x.id === selectedGameId);
    const durationSec = summary.elapsed_seconds || elapsedSec || 0;
    const part = currentGame().buildSession(summary, reason);

    const payload = {
        patient_id:          patientId,
        exercise_type:       `Game: ${g ? g.label : selectedGameId}`,
        start_time:          new Date(sessionStartTime).toISOString(),
        end_time:            new Date().toISOString(),
        duration_seconds:    Math.round(durationSec),
        total_reps:          part.total_reps,
        completed_reps:      part.completed_reps,
        accuracy_percentage: part.accuracy_percentage,
        average_rom:         0,
        incorrect_movements: 0,
        stability_score:     0,
        balance_score:       0,
        movement_smoothness: 0,
        fatigue_estimation:  0,
        recovery_score:      part.recovery_score,
        joint_angles:        [],
        exercise_results:    part.exercise_results || [],
        session_data: {
            mode:             'game',
            game_id:          selectedGameId,
            target_metric:    summary.target_metric,
            target_steps:     summary.target_steps,
            duration_seconds: summary.duration_seconds,
            finish_reason:    reason,
            events:           summary.events || [],
            recording_id:     currentRecordingId,
            ...part.session_data,
        },
    };
    return saveSessionPayload(payload, true);
}

// Result modal, repurposed for game stats (reuses the exercise result-modal markup)
function showGameResultModal(summary, reason) {
    const durSec = Math.round(summary.elapsed_seconds || elapsedSec || 0);
    const durStr = `${String(Math.floor(durSec/60)).padStart(2,'0')}:${String(durSec%60).padStart(2,'0')}`;
    const g = gameListCache.find(x => x.id === selectedGameId);
    const res = currentGame().result(summary, reason);

    document.getElementById('res_duration').textContent = durStr;
    const valueIds = ['res_accuracy', 'res_rom', 'res_stability'];
    res.rows.forEach(([label, value], i) => {
        document.querySelector(`#resultModal .result-stat:nth-child(${i + 2}) .lbl`).textContent = label;
        document.getElementById(valueIds[i]).textContent = value;
    });

    document.getElementById('resultSubtitle').textContent =
        `${g ? g.label : 'Game'} · ${durStr} · ${document.querySelector('#sessionPatientSelect option:checked')?.textContent || ''}` +
        res.subtitle;

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