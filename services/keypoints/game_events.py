"""Copies game engine events onto the recording clock."""
from typing import Any, Dict, Optional

_DEFAULT_TYPES = {"balloon_pop": "pop"}
_COUNTERS = {"misses": "miss", "lean_events": "lean"}


def _plain(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


class GameEventTap:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._errors = 0
        self._last_id = 0
        self._counters: Dict[str, int] = {}
        self._calibrated: Optional[bool] = None
        self._phase: Optional[str] = None

    def poll(self, engine, recorder, now: float, game_id: Optional[str] = None) -> None:
        """Call once per frame while recording; adds whatever the engine logged since the last call.
        A game without get_events_since()/get_status(), or one that raises, never breaks the frame."""
        if engine is None:
            return
        try:
            self._poll(engine, recorder, now, game_id)
        except Exception as exc:
            self._errors += 1
            if self._errors <= 3:
                print(f"GameEventTap: skipped: {exc}")

    def _poll(self, engine, recorder, now: float, game_id: Optional[str]) -> None:
        default_type = _DEFAULT_TYPES.get(game_id or "", "event")
        get_events = getattr(engine, "get_events_since", None)
        for event in (get_events(self._last_id) if callable(get_events) else ()):
            self._last_id = max(self._last_id, event.get("id", 0))
            data = {k: _plain(v) for k, v in event.items() if k not in ("id", "t", "type")}
            recorder.add_event(float(event.get("t", now)), event.get("type") or default_type,
                               event_id=event.get("id"), **data)

        get_status = getattr(engine, "get_status", None)
        status = get_status() if callable(get_status) else {}
        calibrated = status.get("calibrated")
        if calibrated is not None and calibrated != self._calibrated:
            if self._calibrated is not None or calibrated:
                recorder.add_event(now, "calibrated" if calibrated else "calibration_lost")
            self._calibrated = calibrated
        phase = status.get("phase")
        if phase is not None and phase != self._phase:
            recorder.add_event(now, "phase", phase=phase)
            self._phase = phase
        for key, name in _COUNTERS.items():
            value = status.get(key)
            if not isinstance(value, int):
                continue
            last = self._counters.get(key, 0)
            if value > last:
                recorder.add_event(now, name, count=value - last)
            self._counters[key] = value