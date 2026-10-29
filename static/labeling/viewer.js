/* Labeling viewer: loads a recording chosen in page.js, plays the skeleton, drives the
   timeline, and connects the timeline to the label editor (labels.js).

   The timeline gets the whole recording as a down-sampled overview (/series without landmarks).
   Landmarks are fetched in blocks of BLOCK frames, frame by frame (step 1), only around the
   frame being shown, so playback and labels always use real frame numbers. */
(function () {
    'use strict';

    const BLOCK = 300;
    const MAX_BLOCKS = 8;
    const DEFAULT_FPS = 15;

    const $ = (id) => document.getElementById(id);
    const els = {
        stageMsg: $('lvStageMsg'),
        canvas: $('lvSkeleton'),
        timeline: $('lvTimeline'),
        prev: $('lvPrev'),
        play: $('lvPlay'),
        next: $('lvNext'),
        scrub: $('lvScrub'),
        time: $('lvTimeReadout'),
        frame: $('lvFrameReadout'),
        speed: $('lvSpeed'),
    };

    const skeleton = window.LabelingSkeleton.create(els.canvas, { kind: 'pose', aspect: 16 / 9 });
    const timeline = window.LabelingTimeline.create(els.timeline, {
        onSeek: (frame) => goTo(frame),
        onSelect: (sel) => window.LabelingLabels.select(sel),
    });
    window.LabelingLabels.init({
        onLabels: (labels) => timeline.setLabels(labels),
        setSelection: (sel) => timeline.setSelection(sel),
        seek: (frame) => goTo(frame),
    });

    let rec = null;
    let overview = null;
    let blocks = new Map();
    let inflight = new Map();
    let total = 0;
    let fps = DEFAULT_FPS;
    let shown = 0;
    let pos = 0;
    let playing = false;
    let lastTs = 0;
    let raf = 0;
    let token = 0;

    async function getJSON(url) {
        const res = await fetch(url, { credentials: 'same-origin' });
        if (res.status === 401) {
            window.location.href = '/login';
            throw new Error('Not signed in');
        }
        if (!res.ok) {
            let message = `Request failed (${res.status})`;
            try { message = (await res.json()).detail || message; } catch (e) { /* keep default */ }
            throw new Error(message);
        }
        return res.json();
    }

    const seriesUrl = (query) => `/api/recordings/${encodeURIComponent(rec.patientId)}/${encodeURIComponent(rec.recordingId)}/series?${query}`;

    function setStage(text) {
        els.stageMsg.textContent = text || '';
        els.stageMsg.hidden = !text;
    }

    function setControls(enabled) {
        [els.prev, els.play, els.next, els.scrub, els.speed].forEach((el) => { el.disabled = !enabled; });
    }

    function nearest(values, target) {
        let lo = 0;
        let hi = values.length - 1;
        while (lo < hi) {
            const mid = (lo + hi) >> 1;
            if (values[mid] < target) lo = mid + 1; else hi = mid;
        }
        if (lo > 0 && Math.abs(values[lo - 1] - target) <= Math.abs(values[lo] - target)) return lo - 1;
        return lo;
    }

    function secondsAt(frame) {
        if (!overview || !overview.index.length) return frame / fps;
        const i = nearest(overview.index, frame);
        const j = overview.index[i] > frame ? i - 1 : i + 1;
        const t = overview.t[i];
        if (!Number.isFinite(t)) return frame / fps;
        if (j < 0 || j >= overview.index.length || !Number.isFinite(overview.t[j])) return t;
        const span = overview.index[j] - overview.index[i];
        return span ? t + ((frame - overview.index[i]) / span) * (overview.t[j] - t) : t;
    }

    const blockOf = (frame) => Math.floor(frame / BLOCK);

    async function loadBlock(b) {
        if (blocks.has(b)) return blocks.get(b);
        if (inflight.has(b)) return inflight.get(b);
        const mine = token;
        const start = b * BLOCK;
        const promise = getJSON(seriesUrl(`start=${start}&end=${start + BLOCK}&step=1&landmarks=true&sides=patient`))
            .then((data) => {
                inflight.delete(b);
                if (mine !== token) return null;
                const block = { start: data.start, landmarks: data.landmarks || [] };
                blocks.set(b, block);
                while (blocks.size > MAX_BLOCKS) blocks.delete(blocks.keys().next().value);
                return block;
            })
            .catch((err) => {
                inflight.delete(b);
                throw err;
            });
        inflight.set(b, promise);
        return promise;
    }

    function drawFrame(frame) {
        const block = blocks.get(blockOf(frame));
        if (!block) return;
        const drawn = skeleton.draw(block.landmarks[frame - block.start] || null);
        setStage(drawn ? '' : 'No skeleton in this frame');
    }

    function show(frame) {
        shown = frame;
        els.scrub.value = String(frame);
        els.time.textContent = `${secondsAt(frame).toFixed(1)} s`;
        els.frame.textContent = `frame ${frame} / ${Math.max(0, total - 1)}`;
        timeline.setPlayhead(frame);
        if (blocks.has(blockOf(frame))) {
            drawFrame(frame);
            const next = blockOf(frame) + 1;
            if (frame % BLOCK > BLOCK * 0.6 && next * BLOCK < total && !blocks.has(next)) loadBlock(next).catch(() => {});
            return;
        }
        const mine = token;
        loadBlock(blockOf(frame))
            .then(() => { if (mine === token && shown === frame) drawFrame(frame); })
            .catch((err) => { if (mine === token) setStage(`Could not load frames: ${err.message}`); });
    }

    function goTo(frame) {
        if (!total) return;
        const f = Math.max(0, Math.min(total - 1, Math.round(frame)));
        pos = f;
        show(f);
    }

    function setPlaying(on) {
        playing = on && total > 0;
        els.play.querySelector('i').className = `fas fa-${playing ? 'pause' : 'play'}`;
        cancelAnimationFrame(raf);
        if (playing) {
            if (shown >= total - 1) goTo(0);
            lastTs = performance.now();
            raf = requestAnimationFrame(tick);
        }
    }

    function tick(now) {
        if (!playing) return;
        const speed = parseFloat(els.speed.value) || 1;
        const dt = Math.min(0.25, (now - lastTs) / 1000);
        lastTs = now;
        const target = pos + dt * fps * speed;
        const frame = Math.floor(target);
        if (frame >= total - 1) {
            pos = total - 1;
            show(total - 1);
            setPlaying(false);
            return;
        }
        if (!blocks.has(blockOf(frame))) {
            loadBlock(blockOf(frame)).catch((err) => {
                setPlaying(false);
                setStage(`Could not load frames: ${err.message}`);
            });
        } else {
            pos = target;
            if (frame !== shown) show(frame);
        }
        raf = requestAnimationFrame(tick);
    }

    function reset() {
        token += 1;
        setPlaying(false);
        blocks = new Map();
        inflight = new Map();
        overview = null;
        total = 0;
        shown = 0;
        pos = 0;
        setControls(false);
        els.scrub.max = '0';
        els.scrub.value = '0';
        els.time.textContent = '0.0 s';
        els.frame.textContent = 'frame 0 / 0';
        skeleton.clear();
    }

    async function open(detail) {
        reset();
        if (!detail) {
            rec = null;
            return;
        }
        rec = { patientId: detail.patientId, recordingId: detail.recordingId };
        const meta = detail.meta || {};
        const mine = token;
        setStage('Preparing playback...');
        skeleton.setKind(meta.kind);
        const capture = meta.capture || {};
        if (capture.frame_width > 0 && capture.frame_height > 0) skeleton.setAspect(capture.frame_width / capture.frame_height);
        fps = meta.avg_fps > 0 ? meta.avg_fps : DEFAULT_FPS;
        try {
            const data = await getJSON(seriesUrl('sides=patient'));
            if (mine !== token) return;
            overview = data;
            total = data.frames_total || 0;
            if (!total) {
                setStage('This recording has no frames.');
                return;
            }
            timeline.setData({
                framesTotal: total,
                index: data.index,
                t: data.t,
                detected: data.detected,
                angle: data.angle,
                reps: data.reps,
                t0: data.t0,
                events: meta.events || [],
            });
            els.scrub.max = String(total - 1);
            setControls(true);
            goTo(0);
        } catch (err) {
            if (mine === token) setStage(`Could not load recording: ${err.message}`);
        }
    }

    els.play.addEventListener('click', () => setPlaying(!playing));
    els.prev.addEventListener('click', () => { setPlaying(false); goTo(shown - 1); });
    els.next.addEventListener('click', () => { setPlaying(false); goTo(shown + 1); });
    els.scrub.addEventListener('input', () => goTo(parseInt(els.scrub.value, 10) || 0));

    document.addEventListener('keydown', (event) => {
        if (!total || event.altKey || event.ctrlKey || event.metaKey) return;
        const tag = (event.target && event.target.tagName) || '';
        if (/^(INPUT|SELECT|TEXTAREA|BUTTON)$/.test(tag)) return;
        if (event.key === ' ') {
            event.preventDefault();
            setPlaying(!playing);
        } else if (event.key === 'ArrowLeft') {
            setPlaying(false);
            goTo(shown - 1);
        } else if (event.key === 'ArrowRight') {
            setPlaying(false);
            goTo(shown + 1);
        }
    });

    document.addEventListener('labeling:recording', (event) => open(event.detail));

    const current = window.LabelingPage && window.LabelingPage.current();
    if (current) open(current);
})();