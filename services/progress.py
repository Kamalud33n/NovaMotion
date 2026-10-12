"""
Per-patient progress over time, grouped per exercise and per game.
"""
from typing import Callable, Optional

from game.registry import get_game
from services.helpers import (
    split_sessions, game_id_of, is_balloon_session, balloon_stats,
    game_steps, game_final_score,
)

Metric = tuple  # (key, label, unit, higher_is_better, getter(session) -> number|None)


def _num(v) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _col(attr: str) -> Callable:
    return lambda s: _num(getattr(s, attr, None))


def _balloon_pop_rate(s):
    b = balloon_stats(s)
    if b["misses"] is None:                       # old session, misses unknown
        return None
    judged = b["reps"] + b["misses"]
    return round(b["reps"] / judged * 100, 1) if judged else None


def _balloon(key):
    return lambda s: _num(balloon_stats(s).get(key))


EXERCISE_METRICS: list = [
    ("accuracy",   "Accuracy",   "%", True, _col("accuracy_percentage")),
    ("rom",        "ROM",        "°", True, _col("average_rom")),
    ("stability",  "Stability",  "%", True, _col("stability_score")),
    ("balance",    "Balance",    "%", True, _col("balance_score")),
    ("smoothness", "Smoothness", "%", True, _col("movement_smoothness")),
    ("reps",       "Reps",       "",  True, _col("completed_reps")),
]

BALLOON_METRICS: list = [
    ("pop_rate",        "Pop rate",         "%", True,  _balloon_pop_rate),
    ("reps",            "Balloons popped",  "",  True,  _balloon("reps")),
    ("misses",          "Missed",           "",  False, _balloon("misses")),
    ("avg_time_to_pop", "Avg pop time",     "s", False, _balloon("avg_time_to_pop")),
    ("max_elevation",   "Max elevation",    "°", True,  _balloon("max_elevation")),
    ("lean_events",     "Trunk lean events", "", False, _balloon("lean_events")),
]

RUNNER_METRICS: list = [
    ("steps",       "Steps", "", True, lambda s: _num(game_steps(s))),
    ("final_score", "Score", "", True, lambda s: _num(game_final_score(s))),
]

GENERIC_GAME_METRICS: list = [
    ("reps", "Reps", "", True, _col("completed_reps")),
]


def _metrics_for_game(game_id: Optional[str]) -> list:
    if game_id == "balloon_pop":
        return BALLOON_METRICS
    if game_id == "rehab_runner":
        return RUNNER_METRICS
    return GENERIC_GAME_METRICS


def _round(v):
    return None if v is None else round(v, 2)


def _group(name: str, label: str, kind: str, game_id: Optional[str],
           sessions: list, metrics: list) -> dict:
    sessions = sorted(sessions, key=lambda s: s.start_time)

    # Exercise columns the app never filled in (e.g. balance / smoothness are
    # stored as 0 when not measured) would show a fake "0 -> 0" row, so drop
    # any exercise metric that is zero or missing in every session. Game
    # metrics are kept: 0 misses / 0 lean events is a real, good result.
    if kind == "exercise":
        metrics = [m for m in metrics
                   if any((m[4](s) or 0) != 0 for s in sessions)]

    series = []
    for s in sessions:
        series.append({
            "session_id": s.id,
            "date":       s.start_time.isoformat(),
            "metrics":    {k: _round(get(s)) for k, _l, _u, _h, get in metrics},
        })

    change = {}
    for key, mlabel, unit, higher_better, _get in metrics:
        vals = [p["metrics"][key] for p in series if p["metrics"][key] is not None]
        if len(vals) < 2:
            continue
        first, latest = vals[0], vals[-1]
        delta = round(latest - first, 2)
        pct = round(delta / first * 100, 1) if first else None
        improved = None if delta == 0 else ((delta > 0) == higher_better)
        change[key] = {
            "label": mlabel, "unit": unit, "higher_is_better": higher_better,
            "first": first, "latest": latest, "delta": delta, "pct": pct,
            "improved": improved,
        }

    return {
        "kind":          kind,                 # "exercise" | "game"
        "name":          name,                 # exercise_type or game_id
        "label":         label,
        "game_id":       game_id,
        "sessions":      len(series),
        "first_date":    series[0]["date"] if series else None,
        "latest_date":   series[-1]["date"] if series else None,
        "enough_data":   len(series) >= 2,     # need 2+ sessions to show improvement
        "metrics":       [{"key": k, "label": l, "unit": u, "higher_is_better": h}
                          for k, l, u, h, _g in metrics],
        "series":        series,
        "change":        change,
    }


def build_progress(sessions: list) -> dict:
    """{"exercises": [...], "games": [...]} — one entry per exercise type and
    per game the patient has sessions for, most-practised first."""
    exercise_sessions, game_sessions = split_sessions(sessions)

    by_ex: dict = {}
    for s in exercise_sessions:
        by_ex.setdefault(s.exercise_type or "Unknown", []).append(s)

    by_game: dict = {}
    for s in game_sessions:
        by_game.setdefault(game_id_of(s) or "unknown", []).append(s)

    exercises = [
        _group(name, name, "exercise", None, ss, EXERCISE_METRICS)
        for name, ss in by_ex.items()
    ]
    games = []
    for gid, ss in by_game.items():
        entry = get_game(gid)
        label = entry["label"] if entry else gid
        games.append(_group(gid, label, "game", gid, ss, _metrics_for_game(gid)))

    exercises.sort(key=lambda g: -g["sessions"])
    games.sort(key=lambda g: -g["sessions"])
    return {"exercises": exercises, "games": games}