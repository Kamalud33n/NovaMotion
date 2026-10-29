/* Skeleton drawing for the labeling viewer (canvas only, no network, no page state).

   const view = LabelingSkeleton.create(canvas, { kind: 'pose', aspect: 384 / 216 });
   view.draw(frame);   // one frame exactly as /series returns it; true when something was drawn

   pose frame  : 33 x [x, y, z, visibility]
   hands frame : 2 x 21 x [x, y, z]   (slot 0 = left hand, slot 1 = right hand)
   x, y are 0..1 of the camera frame; missing values arrive as null.
   Fetch /series with sides=patient so "left" and "right" here are the patient's own sides. */
(function () {
    'use strict';

    const POSE_LINES = [
        [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8], [9, 10],
        [11, 12], [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19],
        [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
        [11, 23], [12, 24], [23, 24], [23, 25], [24, 26], [25, 27], [26, 28],
        [27, 29], [28, 30], [29, 31], [30, 32], [27, 31], [28, 32],
    ];

    const HAND_LINES = [
        [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10],
        [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17],
        [17, 18], [18, 19], [19, 20],
    ];

    const POSE_WRISTS = { l: 15, r: 16 };

    const COLORS = {
        l: '#3b82a6',
        r: '#1D8A6D',
        c: '#656b68',
        joint: '#ffffff',
        text: '#191a19',
        field: 'rgba(255, 255, 255, 0.55)',
        fieldEdge: 'rgba(20, 20, 18, 0.12)',
    };

    function poseSide(i) {
        return i === 0 ? 'c' : (i % 2 === 1 ? 'l' : 'r');
    }

    function lineSide(sideOf, a, b) {
        const sa = sideOf(a);
        return sa === sideOf(b) ? sa : 'c';
    }

    function create(canvas, options) {
        const opts = Object.assign({ kind: 'pose', aspect: 16 / 9, minVisibility: 0.3, sideLabels: true }, options);
        const ctx = canvas.getContext('2d');
        let frame = null;
        let observer = null;

        function dpr() {
            return (typeof window !== 'undefined' && window.devicePixelRatio) || 1;
        }

        function fit() {
            const rect = canvas.getBoundingClientRect();
            const w = Math.max(1, Math.round(rect.width * dpr()));
            const h = Math.max(1, Math.round(rect.height * dpr()));
            if (canvas.width !== w || canvas.height !== h) {
                canvas.width = w;
                canvas.height = h;
            }
        }

        function field() {
            const cw = canvas.width;
            const ch = canvas.height;
            const pad = Math.round(Math.min(cw, ch) * 0.03);
            let w = cw - 2 * pad;
            let h = w / opts.aspect;
            if (h > ch - 2 * pad) {
                h = ch - 2 * pad;
                w = h * opts.aspect;
            }
            return { x: (cw - w) / 2, y: (ch - h) / 2, w, h };
        }

        function point(lm, box, minVisibility) {
            if (!Array.isArray(lm) || !Number.isFinite(lm[0]) || !Number.isFinite(lm[1])) return null;
            if (minVisibility != null && Number.isFinite(lm[3]) && lm[3] < minVisibility) return null;
            return { x: box.x + lm[0] * box.w, y: box.y + lm[1] * box.h };
        }

        function drawSet(points, lines, sideOf, box) {
            const unit = box.w;
            ctx.lineCap = 'round';
            ctx.lineWidth = Math.max(2, unit * 0.005);
            lines.forEach(([a, b]) => {
                if (!points[a] || !points[b]) return;
                ctx.strokeStyle = COLORS[lineSide(sideOf, a, b)];
                ctx.beginPath();
                ctx.moveTo(points[a].x, points[a].y);
                ctx.lineTo(points[b].x, points[b].y);
                ctx.stroke();
            });
            const radius = Math.max(2.5, unit * 0.006);
            ctx.lineWidth = Math.max(1, unit * 0.002);
            points.forEach((p, i) => {
                if (!p) return;
                ctx.fillStyle = COLORS[sideOf(i)];
                ctx.strokeStyle = COLORS.joint;
                ctx.beginPath();
                ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
                ctx.fill();
                ctx.stroke();
            });
        }

        function sideLabel(text, p, box, below) {
            if (!p || !opts.sideLabels) return;
            const gap = Math.max(6, box.w * 0.012);
            ctx.font = `600 ${Math.max(11, Math.round(box.w * 0.03))}px Inter, sans-serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = below ? 'top' : 'bottom';
            ctx.fillStyle = COLORS.text;
            ctx.fillText(text, p.x, below ? p.y + gap : p.y - gap);
        }

        function drawPose(box) {
            if (!Array.isArray(frame)) return false;
            const points = frame.map((lm) => point(lm, box, opts.minVisibility));
            if (!points.some(Boolean)) return false;
            drawSet(points, POSE_LINES, poseSide, box);
            sideLabel('L', points[POSE_WRISTS.l], box);
            sideLabel('R', points[POSE_WRISTS.r], box);
            return true;
        }

        function drawHands(box) {
            if (!Array.isArray(frame)) return false;
            let drawn = false;
            ['l', 'r'].forEach((side, slot) => {
                const hand = frame[slot];
                if (!Array.isArray(hand)) return;
                const points = hand.map((lm) => point(lm, box, null));
                if (!points.some(Boolean)) return;
                drawSet(points, HAND_LINES, () => side, box);
                sideLabel(side.toUpperCase(), points[0], box, true);
                drawn = true;
            });
            return drawn;
        }

        function redraw() {
            fit();
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            const box = field();
            ctx.fillStyle = COLORS.field;
            ctx.strokeStyle = COLORS.fieldEdge;
            ctx.lineWidth = 1;
            ctx.fillRect(box.x, box.y, box.w, box.h);
            ctx.strokeRect(box.x + 0.5, box.y + 0.5, box.w - 1, box.h - 1);
            return opts.kind === 'hands' ? drawHands(box) : drawPose(box);
        }

        if (typeof ResizeObserver !== 'undefined') {
            observer = new ResizeObserver(() => redraw());
            observer.observe(canvas);
        }

        return {
            draw(next) {
                frame = next || null;
                return redraw();
            },
            clear() {
                frame = null;
                redraw();
            },
            setKind(kind) {
                opts.kind = kind === 'hands' ? 'hands' : 'pose';
                frame = null;
            },
            setAspect(aspect) {
                if (aspect > 0) opts.aspect = aspect;
                redraw();
            },
            destroy() {
                if (observer) observer.disconnect();
                observer = null;
                frame = null;
            },
        };
    }

    window.LabelingSkeleton = { create, POSE_LINES, HAND_LINES };
})();