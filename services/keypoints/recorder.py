"""In-memory keypoint recorder, one instance per live user session.

Frames are buffered in preallocated numpy arrays and written once, as a
compressed .npz, when the session ends. Never raises into the frame pipeline.
"""
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .settings import KeypointConfig
from .storage import delete_partial, save_partial, save_recording

KIND_POSE = "pose"
KIND_HANDS = "hands"
POSE_POINTS = 33
HAND_POINTS = 21
HAND_SLOTS = ("left", "right")
FOCUS_CODES = {"off": 0, "searching": 1, "locked": 2, "holding": 3}
SCHEMA_VERSION = 2   # v2: adds meta.capture (mirror/model/frame size) and meta.timing
_CHUNK = 1024


def new_recording_id() -> str:
    return f"REC-{uuid.uuid4().hex[:12].upper()}"


def _utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _spec(kind: str) -> Dict[str, tuple]:
    spec = {
        "t": (np.float64, (), np.nan),
        "detected": (np.uint8, (), 0),
        "focus": (np.uint8, (), 0),
        "reps": (np.int16, (), -1),
        "angle": (np.float32, (), np.nan),
    }
    if kind == KIND_POSE:
        spec["pose"] = (np.float16, (POSE_POINTS, 4), np.nan)
    else:
        spec["hands"] = (np.float16, (len(HAND_SLOTS), HAND_POINTS, 3), np.nan)
    return spec


class _Buffers:
    def __init__(self, kind: str, max_frames: int):
        self.spec = _spec(kind)
        self.max_frames = max_frames
        self.size = 0
        self.capacity = 0
        self.arrays: Dict[str, np.ndarray] = {}
        self._grow()

    def _grow(self) -> None:
        new_cap = min(self.capacity + _CHUNK, self.max_frames)
        grown = {}
        for name, (dtype, shape, fill) in self.spec.items():
            arr = np.full((new_cap,) + shape, fill, dtype=dtype)
            if name in self.arrays:
                arr[: self.capacity] = self.arrays[name]
            grown[name] = arr
        self.arrays, self.capacity = grown, new_cap

    def next_row(self) -> Optional[int]:
        if self.size >= self.max_frames:
            return None
        if self.size == self.capacity:
            self._grow()
        row = self.size
        self.size += 1
        return row

    def trimmed(self) -> Dict[str, np.ndarray]:
        return {name: arr[: self.size].copy() for name, arr in self.arrays.items()}


def _pose_array(landmarks: Sequence) -> np.ndarray:
    pts = landmarks[:POSE_POINTS]
    return np.array([(p.x, p.y, p.z, p.visibility) for p in pts], dtype=np.float32)


def _hand_array(landmarks: Sequence) -> np.ndarray:
    pts = list(landmarks)[:HAND_POINTS]
    return np.array([(p.x, p.y, p.z) for p in pts], dtype=np.float32)


def _timing_stats(t: np.ndarray) -> Dict[str, Any]:
    """Frame-interval stats: tells you whether fps was steady or the stream stalled."""
    dt = np.diff(t[np.isfinite(t)])
    dt = dt[dt > 0]
    if dt.size == 0:
        return {}
    median = float(np.median(dt))
    big = np.nonzero(dt > 0.5)[0]
    return {
        "median_fps": round(1.0 / median, 2) if median > 0 else None,
        "p95_interval_ms": round(float(np.percentile(dt, 95)) * 1000, 1),
        "max_gap_ms": round(float(dt.max()) * 1000, 1),
        "gaps_over_250ms": int((dt > 0.25).sum()),
        "gaps_over_500ms": int(big.size),
        # first 20 stalls over 0.5 s: frame index where the stall ended + its length
        "long_gaps": [{"frame": int(i) + 1, "seconds": round(float(dt[i]), 2)} for i in big[:20]],
    }


class KeypointRecorder:
    def __init__(self, config: Optional[KeypointConfig] = None):
        self._cfg = config or KeypointConfig.from_env()
        self._lock = threading.Lock()
        self._buf: Optional[_Buffers] = None
        self._meta: Dict[str, Any] = {}
        self._events: List[Dict[str, Any]] = []
        self._truncated = False
        self._errors = 0
        self._frame_sizes: Dict[Tuple[int, int], int] = {}
        self._last_info: Optional[Dict[str, Any]] = None
        self._next_checkpoint = 0.0            # time.monotonic() of the next partial save
        self._ckpt_busy = False
        self._ckpt_io = threading.Lock()       # serialises partial-file writes and deletes

    @property
    def enabled(self) -> bool:
        return self._cfg.enabled

    @property
    def active(self) -> bool:
        return self._buf is not None

    @property
    def recording_id(self) -> Optional[str]:
        return self._meta.get("recording_id") if self._buf is not None else None

    @property
    def patient_id(self) -> Optional[str]:
        return self._meta.get("patient_id") if self._buf is not None else None

    @property
    def frame_count(self) -> int:
        buf = self._buf
        return buf.size if buf is not None else 0

    def start(self, kind: str, *, patient_id: Optional[str] = None, owner_id: Optional[str] = None,
              label: Optional[str] = None, side: Optional[str] = None,
              extra: Optional[Dict[str, Any]] = None,
              capture: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Begin a recording and return its id, or None when recording is disabled.
        A recording that was never stopped is saved first (reason "restarted")."""
        if not self._cfg.enabled:
            return None
        if kind not in (KIND_POSE, KIND_HANDS):
            raise ValueError(f"unknown recording kind: {kind!r}")
        with self._lock:
            if self._buf is not None:
                self._finish_locked("restarted")
            self._buf = _Buffers(kind, self._cfg.max_frames)
            self._events = []
            self._truncated = False
            self._errors = 0
            self._frame_sizes = {}
            interval = self._cfg.checkpoint_seconds
            self._next_checkpoint = time.monotonic() + interval if interval > 0 else 0.0
            self._meta = {
                "schema_version": SCHEMA_VERSION,
                "recording_id": new_recording_id(),
                "kind": kind,
                "patient_id": patient_id,
                "owner_id": owner_id,
                "label": label,
                "side": side,
                "extra": dict(extra or {}),
                "capture": dict(capture or {}),
            }
            return self._meta["recording_id"]

    def add_frame(self, ts: float, *, pose: Optional[Sequence] = None,
                  hands: Optional[Dict[str, Optional[Sequence]]] = None,
                  focus: Optional[str] = None, detected: bool = False,
                  angle: Optional[float] = None, reps: Optional[int] = None,
                  frame_size: Optional[Tuple[int, int]] = None) -> bool:
        """Record one frame. `pose` is the 33 pose landmarks, `hands` maps
        "left"/"right" to 21 hand landmarks; leave them None when nobody was found."""
        if self._buf is None:
            return False
        try:
            pose_arr = _pose_array(pose) if pose is not None else None
            hand_arrs = {
                slot: _hand_array(hands[name])
                for slot, name in enumerate(HAND_SLOTS)
                if hands and hands.get(name) is not None
            }
            with self._lock:
                buf = self._buf
                if buf is None:
                    return False
                row = buf.next_row()
                if row is None:
                    self._truncated = True
                    return False
                if frame_size is not None:
                    key = (int(frame_size[0]), int(frame_size[1]))
                    if key in self._frame_sizes or len(self._frame_sizes) < 8:
                        self._frame_sizes[key] = self._frame_sizes.get(key, 0) + 1
                arrays = buf.arrays
                arrays["t"][row] = ts
                arrays["detected"][row] = 1 if detected else 0
                arrays["focus"][row] = FOCUS_CODES.get(focus or "off", 0)
                if reps is not None:
                    arrays["reps"][row] = reps
                if angle is not None:
                    arrays["angle"][row] = angle
                if pose_arr is not None and "pose" in arrays:
                    arrays["pose"][row, : len(pose_arr)] = pose_arr
                if "hands" in arrays:
                    for slot, arr in hand_arrs.items():
                        arrays["hands"][row, slot, : len(arr)] = arr
                snapshot = self._checkpoint_due_locked()
            if snapshot is not None:
                self._start_checkpoint_thread(snapshot)
            return True
        except Exception as exc:
            self._errors += 1
            if self._errors <= 3:
                print(f"KeypointRecorder: frame skipped: {exc}")
            return False

    def add_event(self, ts: float, kind: str, **data: Any) -> None:
        """Game / rep events on the same clock as the frames."""
        if self._buf is None:
            return
        with self._lock:
            if self._buf is not None and len(self._events) < self._cfg.max_events:
                self._events.append({"t": round(float(ts), 3), "type": kind, **data})

    def stop(self, reason: str = "stopped") -> Optional[Dict[str, Any]]:
        """Finish and write the recording. Returns file info, or None when there
        was nothing worth keeping (too few frames) or no recording was active."""
        with self._lock:
            return self._finish_locked(reason)

    def finish(self, recording_id: str, reason: str = "saved") -> Optional[Dict[str, Any]]:
        """Stop the active recording if it is `recording_id`; if that recording was
        already stopped (camera stop, restart), return its saved file info instead."""
        with self._lock:
            if self._buf is not None and self._meta.get("recording_id") == recording_id:
                return self._finish_locked(reason)
            last = self._last_info
            return last if last and last["recording_id"] == recording_id else None

    def discard(self) -> None:
        with self._lock:
            pid, rid = self._meta.get("patient_id"), self._meta.get("recording_id")
            self._buf = None
            self._events = []
            self._meta = {}
            if rid:
                self._remove_partial(pid, rid)

    def discard_for_patient(self, patient_id: str) -> bool:
        """Drop (never save) the active recording if it belongs to `patient_id`.
        Used when consent is withdrawn mid-session. Returns True if one was dropped."""
        with self._lock:
            if self._buf is not None and self._meta.get("patient_id") == patient_id:
                rid = self._meta.get("recording_id")
                self._buf, self._events, self._meta = None, [], {}
                self._remove_partial(patient_id, rid)
                return True
            if self._last_info and self._last_info.get("patient_id") == patient_id:
                self._last_info = None
            return False

    # -- checkpoints -----------------------------------------------------------
    def _compose_meta(self, arrays: Dict[str, np.ndarray], meta: Dict[str, Any],
                      events: List[Dict[str, Any]], reason: str, partial: bool) -> Dict[str, Any]:
        t = arrays["t"]
        n = len(t)
        duration = float(t[-1] - t[0]) if n > 1 else 0.0
        out = {
            **meta,
            "partial": partial,
            "end_reason": reason,
            "started_at": _utc_iso(float(t[0])),
            "ended_at": _utc_iso(float(t[-1])),
            "frame_count": int(n),
            "duration_seconds": round(duration, 2),
            "avg_fps": round((n - 1) / duration, 2) if duration > 0 else None,
            "timing": _timing_stats(t),
            "truncated": self._truncated,
            "dropped_frames": self._errors,
            "arrays": {name: {"dtype": str(a.dtype), "shape": list(a.shape)} for name, a in arrays.items()},
            "focus_codes": FOCUS_CODES,
            "events": list(events),
        }
        out["capture"] = {**meta.get("capture", {}), **self._frame_size_meta()}
        return out

    def _checkpoint_due_locked(self):
        """Caller holds self._lock. A (arrays, meta) copy when a checkpoint is due, else None."""
        buf = self._buf
        if (self._cfg.checkpoint_seconds <= 0 or buf is None or self._ckpt_busy
                or buf.size < self._cfg.min_frames or time.monotonic() < self._next_checkpoint):
            return None
        self._ckpt_busy = True
        self._next_checkpoint = time.monotonic() + self._cfg.checkpoint_seconds
        try:
            arrays = buf.trimmed()
            meta = self._compose_meta(arrays, self._meta, self._events, "checkpoint", partial=True)
        except Exception as exc:
            self._ckpt_busy = False
            print(f"KeypointRecorder: checkpoint snapshot failed: {exc}")
            return None
        return arrays, meta

    def _start_checkpoint_thread(self, snapshot) -> None:
        threading.Thread(target=self._write_checkpoint, args=snapshot, daemon=True,
                         name="keypoint-checkpoint").start()

    def _active_recording_id(self) -> Optional[str]:
        # read without self._lock on purpose: the checkpoint thread must never wait on the
        # frame lock while holding _ckpt_io (stop()/discard() take them in the other order)
        buf, meta = self._buf, self._meta
        return meta.get("recording_id") if buf is not None else None

    def _write_checkpoint(self, arrays: Dict[str, np.ndarray], meta: Dict[str, Any]) -> None:
        rid = meta.get("recording_id")
        try:
            with self._ckpt_io:
                if self._active_recording_id() != rid:    # stopped/discarded while we were queued
                    return
                save_partial(self._cfg, meta.get("patient_id"), rid, arrays, meta)
        except Exception as exc:
            print(f"KeypointRecorder: checkpoint failed for {rid}: {exc}")
        finally:
            self._ckpt_busy = False

    def _remove_partial(self, patient_id, recording_id) -> None:
        if not recording_id:
            return
        with self._ckpt_io:       # waits for a checkpoint write that is already in progress
            delete_partial(self._cfg, patient_id, recording_id)

    def checkpoint_now(self) -> bool:
        """Write a checkpoint immediately (synchronously). Used by tests and shutdown paths."""
        with self._lock:
            if self._buf is None or self._buf.size < self._cfg.min_frames:
                return False
            self._ckpt_busy = True
            arrays = self._buf.trimmed()
            meta = self._compose_meta(arrays, self._meta, self._events, "checkpoint", partial=True)
        self._write_checkpoint(arrays, meta)
        return True

    def _frame_size_meta(self) -> Dict[str, Any]:
        if not self._frame_sizes:
            return {}
        (w, h), _ = max(self._frame_sizes.items(), key=lambda kv: kv[1])
        out: Dict[str, Any] = {"frame_width": w, "frame_height": h}
        if len(self._frame_sizes) > 1:   # resolution changed mid-session (rare, but ML must know)
            out["frame_sizes_seen"] = [{"width": a, "height": b, "frames": n}
                                       for (a, b), n in self._frame_sizes.items()]
        return out

    def _finish_locked(self, reason: str) -> Optional[Dict[str, Any]]:
        buf, meta, events = self._buf, self._meta, self._events
        self._buf, self._meta, self._events = None, {}, []
        if buf is not None and buf.size < self._cfg.min_frames:
            self._remove_partial(meta.get("patient_id"), meta.get("recording_id"))
        if buf is None or buf.size < self._cfg.min_frames:
            return None

        arrays = buf.trimmed()
        meta = self._compose_meta(arrays, meta, events, reason, partial=False)
        try:
            rel_path = save_recording(self._cfg, meta.get("patient_id"), meta["recording_id"], arrays, meta)
        except Exception as exc:
            # The last checkpoint (if any) is deliberately kept: it can still be recovered.
            print(f"KeypointRecorder: could not save {meta.get('recording_id')}: {exc}")
            return None
        self._remove_partial(meta.get("patient_id"), meta["recording_id"])
        self._last_info = {
            "recording_id": meta["recording_id"],
            "patient_id": meta.get("patient_id"),
            "file": rel_path,
            "frames": meta["frame_count"],
            "duration_seconds": meta["duration_seconds"],
            "truncated": meta["truncated"],
        }
        return dict(self._last_info)