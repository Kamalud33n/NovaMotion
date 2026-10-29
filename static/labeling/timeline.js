/* Timeline for the labeling viewer: labels, repetitions, tracking gaps, joint angle, events,
   playhead. Canvas only; the viewer owns the data and the current frame.

   const tl = LabelingTimeline.create(canvas, { onSeek: (frame) => {...} });
   tl.setData({ framesTotal, index, t, detected, angle, reps, t0, events, labels });
   tl.setLabels(labels);     // the user's own labels, as /labels returns them
   tl.setPlayhead(frame);
   tl.repSegments();         // [{ rep, start, end, counted }]  end is exclusive
   tl.setSelection(sel);     // sel = { kind: 'rep', start, end, rep } | { kind: 'label', label } | null
   tl.getSelection();

   A click (no drag) on a repetition or on a saved label selects it and calls
   onSelect(selection); a click on an empty part of those lanes clears it (onSelect(null)).

   index/t/detected/angle/reps are the arrays of /series (index = real frame number of each row,
   t = seconds from the first frame). t0 and the `t` of each event share one absolute clock.
   Colours are CSS variables (--lv-*, see viewer.css). */
(function () {
    'use strict';

    const PAD = { left: 40, right: 10, top: 6, bottom: 22 };
    const LANE = { labels: 22, rep: 20, track: 8, gap: 4 };
    const TICK_STEPS = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1800];

    function nearestIndex(values, target) {
        let lo = 0;
        let hi = values.length - 1;
        while (lo < hi) {
            const mid = (lo + hi) >> 1;
            if (values[mid] < target) lo = mid + 1; else hi = mid;
        }
        if (lo > 0 && Math.abs(values[lo - 1] - target) <= Math.abs(values[lo] - target)) return lo - 1;
        return lo;
    }

    /* A repetition is the run of frames during which the counter holds one value; it is
       `counted` when the counter then moved up. The run left open at the end (or ended by a
       counter reset) is a repetition that was never completed. */
    function repSegments(index, reps, framesTotal) {
        const out = [];
        let value = null;
        let start = 0;
        const close = (end, counted) => {
            if (value !== null) out.push({ rep: value + 1, start, end, counted });
            value = null;
        };
        for (let i = 0; i < reps.length; i++) {
            const v = reps[i];
            if (!(v >= 0)) {
                close(index[i], false);
                continue;
            }
            if (v === value) continue;
            if (value !== null) close(index[i], v > value);
            value = v;
            start = index[i];
        }
        close(framesTotal, false);
        return out;
    }

    function tickStep(duration, width) {
        const wanted = Math.max(2, Math.floor(width / 90));
        return TICK_STEPS.find((s) => duration / s <= wanted) || TICK_STEPS[TICK_STEPS.length - 1];
    }

    function clock(seconds, long) {
        const s = Math.round(seconds);
        return long ? `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}` : `${s}s`;
    }

    function create(canvas, options) {
        const opts = Object.assign({ onSeek: null, onSelect: null }, options);
        const ctx = canvas.getContext('2d');
        const layer = document.createElement('canvas');
        const lctx = layer.getContext('2d');
        canvas.style.touchAction = 'none';

        let data = null;
        let labels = [];
        let segments = [];
        let eventFrames = [];
        let playhead = 0;
        let hoverX = null;
        let dragging = false;
        let selection = null;
        let downPoint = null;
        let size = { w: 0, h: 0, dpr: 1 };
        let colors = {};
        let observer = null;

        function readColors() {
            const style = getComputedStyle(canvas);
            const pick = (name, fallback) => (style.getPropertyValue(name) || '').trim() || fallback;
            colors = {
                angle: pick('--lv-angle', '#1D8A6D'),
                rep: pick('--lv-rep', 'rgba(29, 138, 109, 0.12)'),
                lost: pick('--lv-lost', 'rgba(217, 80, 62, 0.20)'),
                event: pick('--lv-event', '#7a5bd1'),
                playhead: pick('--lv-playhead', '#191a19'),
                select: pick('--lv-select', '#2563eb'),
                text: pick('--text-secondary', '#656b68'),
                lane: 'rgba(20, 20, 18, 0.04)',
                grid: 'rgba(20, 20, 18, 0.07)',
                label: (id) => pick(`--lv-${id}`, '#9a9d97'),
            };
        }

        function geometry() {
            const l = PAD.left;
            const r = size.w - PAD.right;
            const labelsTop = PAD.top;
            const repTop = labelsTop + LANE.labels + LANE.gap;
            const trackTop = repTop + LANE.rep + LANE.gap;
            const plotTop = trackTop + LANE.track + LANE.gap * 2;
            const plotBottom = size.h - PAD.bottom;
            return { l, r, w: Math.max(1, r - l), labelsTop, repTop, trackTop, plotTop, plotBottom,
                     h: Math.max(1, plotBottom - plotTop) };
        }

        const total = () => (data ? Math.max(1, data.framesTotal) : 1);
        const xOf = (g, frame) => g.l + (frame / total()) * g.w;

        function frameAtX(x) {
            const g = geometry();
            const frame = Math.floor(((x - g.l) / g.w) * total());
            return Math.max(0, Math.min(total() - 1, frame));
        }

        function lane(g, top, height) {
            lctx.fillStyle = colors.lane;
            lctx.fillRect(g.l, top, g.w, height);
        }

        function drawLabels(g) {
            lane(g, g.labelsTop, LANE.labels);
            lctx.font = '600 11px Inter, sans-serif';
            lctx.textAlign = 'center';
            lctx.textBaseline = 'middle';
            labels.forEach((item) => {
                const x1 = xOf(g, item.start_frame);
                const x2 = xOf(g, item.end_frame);
                lctx.fillStyle = colors.label(item.label);
                lctx.fillRect(x1, g.labelsTop, Math.max(2, x2 - x1), LANE.labels);
                const name = item.label.charAt(0).toUpperCase() + item.label.slice(1);
                if (x2 - x1 > lctx.measureText(name).width + 10) {
                    lctx.fillStyle = '#ffffff';
                    lctx.fillText(name, (x1 + x2) / 2, g.labelsTop + LANE.labels / 2 + 1);
                }
            });
        }

        function drawReps(g) {
            lane(g, g.repTop, LANE.rep);
            lctx.font = '600 11px Inter, sans-serif';
            lctx.textAlign = 'center';
            lctx.textBaseline = 'middle';
            segments.forEach((seg) => {
                const x1 = xOf(g, seg.start);
                const x2 = xOf(g, seg.end);
                const width = Math.max(1, x2 - x1);
                lctx.fillStyle = colors.rep;
                lctx.fillRect(x1, g.repTop, width, LANE.rep);
                lctx.strokeStyle = colors.angle;
                lctx.lineWidth = 1;
                lctx.setLineDash(seg.counted ? [] : [3, 3]);
                lctx.strokeRect(x1 + 0.5, g.repTop + 0.5, Math.max(0, width - 1), LANE.rep - 1);
                lctx.setLineDash([]);
                if (width > 16) {
                    lctx.fillStyle = colors.text;
                    lctx.fillText(String(seg.rep), x1 + width / 2, g.repTop + LANE.rep / 2 + 1);
                }
            });
        }

        function drawTracking(g) {
            lane(g, g.trackTop, LANE.track);
            lctx.fillStyle = colors.lost;
            let runStart = null;
            const flush = (endFrame) => {
                if (runStart === null) return;
                const x1 = xOf(g, runStart);
                lctx.fillRect(x1, g.trackTop, Math.max(1.5, xOf(g, endFrame) - x1), LANE.track);
                runStart = null;
            };
            data.detected.forEach((flag, i) => {
                if (!flag) {
                    if (runStart === null) runStart = data.index[i];
                } else {
                    flush(data.index[i]);
                }
            });
            flush(data.framesTotal);
        }

        function drawEvents(g) {
            lctx.strokeStyle = colors.event;
            lctx.lineWidth = 1;
            lctx.globalAlpha = 0.5;
            eventFrames.forEach((frame) => {
                const x = Math.round(xOf(g, frame + 0.5)) + 0.5;
                lctx.beginPath();
                lctx.moveTo(x, g.plotTop);
                lctx.lineTo(x, g.plotBottom);
                lctx.stroke();
            });
            lctx.globalAlpha = 1;
        }

        function drawAngle(g) {
            const finite = data.angle.filter((v) => Number.isFinite(v));
            lctx.strokeStyle = colors.grid;
            lctx.lineWidth = 1;
            lctx.strokeRect(g.l + 0.5, g.plotTop + 0.5, g.w - 1, g.h - 1);
            if (!finite.length) {
                lctx.fillStyle = colors.text;
                lctx.font = '12px Inter, sans-serif';
                lctx.textAlign = 'center';
                lctx.textBaseline = 'middle';
                lctx.fillText('No angle data in this recording', g.l + g.w / 2, g.plotTop + g.h / 2);
                return;
            }
            let min = Math.min(...finite);
            let max = Math.max(...finite);
            if (max - min < 1) { min -= 1; max += 1; }
            const margin = (max - min) * 0.08;
            min -= margin;
            max += margin;
            const yOf = (v) => g.plotBottom - ((v - min) / (max - min)) * g.h;

            lctx.fillStyle = colors.text;
            lctx.font = '11px Inter, sans-serif';
            lctx.textAlign = 'right';
            lctx.textBaseline = 'top';
            lctx.fillText(`${Math.round(max)}\u00b0`, g.l - 6, g.plotTop);
            lctx.textBaseline = 'bottom';
            lctx.fillText(`${Math.round(min)}\u00b0`, g.l - 6, g.plotBottom);

            lctx.strokeStyle = colors.angle;
            lctx.lineWidth = 1.6;
            lctx.lineJoin = 'round';
            lctx.beginPath();
            let pen = false;
            data.angle.forEach((v, i) => {
                if (!Number.isFinite(v)) { pen = false; return; }
                const x = xOf(g, data.index[i] + 0.5);
                if (pen) lctx.lineTo(x, yOf(v)); else lctx.moveTo(x, yOf(v));
                pen = true;
            });
            lctx.stroke();
        }

        function drawAxis(g) {
            const duration = data.t.length ? data.t[data.t.length - 1] : 0;
            if (!(duration > 0)) return;
            const step = tickStep(duration, g.w);
            lctx.fillStyle = colors.text;
            lctx.strokeStyle = colors.grid;
            lctx.font = '11px Inter, sans-serif';
            lctx.textAlign = 'center';
            lctx.textBaseline = 'top';
            for (let sec = 0; sec <= duration; sec += step) {
                const frame = data.index[nearestIndex(data.t, sec)];
                const x = Math.round(xOf(g, frame + 0.5)) + 0.5;
                lctx.beginPath();
                lctx.moveTo(x, g.plotBottom);
                lctx.lineTo(x, g.plotBottom + 4);
                lctx.stroke();
                lctx.fillText(clock(sec, duration >= 60), x, g.plotBottom + 6);
            }
        }

        function renderLayer() {
            layer.width = canvas.width;
            layer.height = canvas.height;
            lctx.setTransform(size.dpr, 0, 0, size.dpr, 0, 0);
            lctx.clearRect(0, 0, size.w, size.h);
            if (!data) return;
            const g = geometry();
            drawLabels(g);
            drawReps(g);
            drawTracking(g);
            drawEvents(g);
            drawAngle(g);
            drawAxis(g);
        }

        function describe(frame, x) {
            const i = nearestIndex(data.index, frame);
            const parts = [`frame ${frame}`];
            if (Number.isFinite(data.t[i])) parts.push(`${data.t[i].toFixed(1)} s`);
            if (Number.isFinite(data.angle[i])) parts.push(`${data.angle[i].toFixed(0)}\u00b0`);
            const seg = segments.find((s) => frame >= s.start && frame < s.end);
            if (seg) parts.push(`rep ${seg.rep}`);
            const labelled = labels.find((s) => frame >= s.start_frame && frame < s.end_frame);
            if (labelled) parts.push(labelled.label);
            const g = geometry();
            const near = (data.events || []).findIndex((ev, k) => Math.abs(xOf(g, eventFrames[k] + 0.5) - x) <= 5);
            if (near >= 0) {
                const ev = data.events[near];
                parts.push(ev.type === 'phase' && ev.phase ? `phase ${ev.phase}` : ev.type);
            }
            return parts.join('  \u00b7  ');
        }

        function paint() {
            ctx.setTransform(1, 0, 0, 1, 0, 0);
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            if (layer.width && layer.height) ctx.drawImage(layer, 0, 0);
            if (!data) return;
            ctx.setTransform(size.dpr, 0, 0, size.dpr, 0, 0);
            const g = geometry();

            if (hoverX !== null) {
                ctx.strokeStyle = colors.grid;
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(Math.round(hoverX) + 0.5, g.labelsTop);
                ctx.lineTo(Math.round(hoverX) + 0.5, g.plotBottom);
                ctx.stroke();
                const text = describe(frameAtX(hoverX), hoverX);
                ctx.font = '11px Inter, sans-serif';
                const width = ctx.measureText(text).width + 14;
                ctx.textAlign = 'left';
                ctx.textBaseline = 'middle';
                const left = Math.min(g.r - width - 4, Math.max(g.l + 4, hoverX - width / 2));
                const top = g.plotBottom - 24;
                ctx.fillStyle = 'rgba(25, 26, 25, 0.82)';
                ctx.fillRect(left, top, width, 18);
                ctx.fillStyle = '#ffffff';
                ctx.fillText(text, left + 7, top + 10);
            }

            if (selection) {
                const x1 = xOf(g, selection.start);
                const x2 = xOf(g, selection.end);
                ctx.fillStyle = colors.select;
                ctx.globalAlpha = 0.12;
                ctx.fillRect(x1, g.labelsTop, Math.max(2, x2 - x1), g.plotBottom - g.labelsTop);
                ctx.globalAlpha = 1;
                ctx.strokeStyle = colors.select;
                ctx.lineWidth = 2;
                const laneTop = selection.kind === 'label' ? g.labelsTop : g.repTop;
                const laneHeight = selection.kind === 'label' ? LANE.labels : LANE.rep;
                ctx.strokeRect(x1 + 1, laneTop + 1, Math.max(1, x2 - x1 - 2), laneHeight - 2);
            }

            const x = Math.round(xOf(g, playhead + 0.5)) + 0.5;
            ctx.strokeStyle = colors.playhead;
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(x, g.labelsTop - 2);
            ctx.lineTo(x, g.plotBottom);
            ctx.stroke();
            ctx.fillStyle = colors.playhead;
            ctx.beginPath();
            ctx.moveTo(x - 4, g.labelsTop - 4);
            ctx.lineTo(x + 4, g.labelsTop - 4);
            ctx.lineTo(x, g.labelsTop + 2);
            ctx.closePath();
            ctx.fill();
        }

        function resize() {
            const rect = canvas.getBoundingClientRect();
            const dpr = (typeof window !== 'undefined' && window.devicePixelRatio) || 1;
            size = { w: Math.max(1, rect.width), h: Math.max(1, rect.height), dpr };
            canvas.width = Math.round(size.w * dpr);
            canvas.height = Math.round(size.h * dpr);
            readColors();
            renderLayer();
            paint();
        }

        function pointerX(event) {
            return event.clientX - canvas.getBoundingClientRect().left;
        }

        function pointerY(event) {
            return event.clientY - canvas.getBoundingClientRect().top;
        }

        function sameSelection(a, b) {
            if (!a || !b) return a === b;
            if (a.kind !== b.kind) return false;
            return a.kind === 'label' ? a.label.id === b.label.id : a.start === b.start && a.end === b.end;
        }

        function applySelection(next, notify) {
            if (sameSelection(selection, next)) return;
            selection = next;
            paint();
            if (notify && opts.onSelect) opts.onSelect(selection ? Object.assign({}, selection) : null);
        }

        function selectAt(x, y) {
            if (!data) return;
            const g = geometry();
            const frame = frameAtX(x);
            if (y >= g.labelsTop && y <= g.labelsTop + LANE.labels) {
                const hit = labels.find((l) => frame >= l.start_frame && frame < l.end_frame);
                applySelection(hit ? { kind: 'label', label: hit, start: hit.start_frame, end: hit.end_frame } : null, true);
            } else if (y >= g.repTop && y <= g.repTop + LANE.rep) {
                const seg = segments.find((s) => frame >= s.start && frame < s.end);
                applySelection(seg ? { kind: 'rep', start: seg.start, end: seg.end, rep: seg.rep, counted: seg.counted } : null, true);
            }
        }

        function seekTo(event) {
            if (!data || !opts.onSeek) return;
            opts.onSeek(frameAtX(pointerX(event)));
        }

        canvas.addEventListener('pointerdown', (event) => {
            dragging = true;
            downPoint = { x: pointerX(event), y: pointerY(event) };
            if (canvas.setPointerCapture) canvas.setPointerCapture(event.pointerId);
            seekTo(event);
        });
        canvas.addEventListener('pointermove', (event) => {
            hoverX = pointerX(event);
            if (dragging) seekTo(event);
            paint();
        });
        canvas.addEventListener('pointerup', (event) => {
            dragging = false;
            if (downPoint && Math.abs(pointerX(event) - downPoint.x) < 4) selectAt(downPoint.x, downPoint.y);
            downPoint = null;
        });
        canvas.addEventListener('pointercancel', () => { dragging = false; downPoint = null; });
        canvas.addEventListener('pointerleave', () => { hoverX = null; paint(); });

        if (typeof ResizeObserver !== 'undefined') {
            observer = new ResizeObserver(() => resize());
            observer.observe(canvas);
        }
        resize();

        return {
            setData(next) {
                const index = next.index || [];
                data = {
                    framesTotal: next.framesTotal || (index.length ? index[index.length - 1] + 1 : 0),
                    index,
                    t: next.t || [],
                    detected: next.detected || [],
                    angle: next.angle || [],
                    reps: next.reps || [],
                    events: next.events || [],
                };
                segments = repSegments(index, data.reps, data.framesTotal);
                eventFrames = Number.isFinite(next.t0) && data.t.length
                    ? data.events.map((ev) => index[nearestIndex(data.t, ev.t - next.t0)])
                    : [];
                if (eventFrames.length !== data.events.length) data.events = [];
                labels = next.labels || labels;
                selection = null;
                playhead = 0;
                renderLayer();
                paint();
            },
            setLabels(next) {
                labels = next || [];
                if (selection && selection.kind === 'label') {
                    const fresh = labels.find((l) => l.id === selection.label.id);
                    selection = fresh ? { kind: 'label', label: fresh, start: fresh.start_frame, end: fresh.end_frame } : null;
                }
                renderLayer();
                paint();
            },
            setSelection(next) {
                applySelection(next || null, false);
            },
            getSelection() {
                return selection ? Object.assign({}, selection) : null;
            },
            setPlayhead(frame) {
                playhead = frame;
                paint();
            },
            repSegments() {
                return segments.map((s) => Object.assign({}, s));
            },
            destroy() {
                if (observer) observer.disconnect();
                observer = null;
                data = null;
            },
        };
    }

    window.LabelingTimeline = { create, repSegments };
})();