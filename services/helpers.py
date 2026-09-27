import cv2
import numpy as np

from models import SessionModel


def compress_photo(raw_bytes: bytes, max_dim: int = 800, quality: int = 80) -> bytes:
    try:
        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return raw_bytes
        h, w = img.shape[:2]
        scale = max_dim / max(h, w)
        if scale < 1:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ok else raw_bytes
    except Exception as exc:
        print(f"Photo compression failed, storing original bytes: {exc}")
        return raw_bytes


def is_game_session(s: SessionModel) -> bool:
    """True for a session saved from the session page's \"Game\" mode
    (services/game_state.py / game/registry.py). Game sessions don't have
    meaningful ROM/stability/balance values (those columns are stored as 0
    for them), so anything averaging exercise-quality metrics across
    sessions should exclude these — see split_sessions() below."""
    return bool((s.session_data or {}).get("mode") == "game")


def game_steps(s: SessionModel) -> int:
    """Steps recorded for a game session (0 for a non-game session)."""
    if not is_game_session(s):
        return 0
    return (s.session_data or {}).get("steps") or s.completed_reps or 0


def game_left_steps(s: SessionModel) -> int:
    """Left-foot steps recorded for a game session (0 for a non-game session)."""
    if not is_game_session(s):
        return 0
    return (s.session_data or {}).get("left_steps") or 0


def game_right_steps(s: SessionModel) -> int:
    """Right-foot steps recorded for a game session (0 for a non-game session)."""
    if not is_game_session(s):
        return 0
    return (s.session_data or {}).get("right_steps") or 0


def game_final_score(s: SessionModel) -> int:
    """Canvas-game score recorded for a game session (0 for a non-game session)."""
    if not is_game_session(s):
        return 0
    return (s.session_data or {}).get("final_score") or 0


def split_sessions(sessions: list) -> tuple:
    """(exercise_sessions, game_sessions) — the standard split before
    computing any accuracy/ROM/stability/balance aggregate, so a Rehab
    Runner session (ROM=0, stability=0 by construction) doesn't silently
    drag down a patient's exercise-quality averages."""
    exercise = [s for s in sessions if not is_game_session(s)]
    game     = [s for s in sessions if is_game_session(s)]
    return exercise, game


def calculate_recovery_score(sessions: list) -> float:
    exercise_sessions, _ = split_sessions(sessions)
    # Fall back to the raw list if a patient has ONLY game sessions so far —
    # better to show a rough number than a hard 0.
    ss = exercise_sessions or sessions
    if not ss:
        return 0.0
    n = len(ss)
    avg = lambda attr: sum(getattr(s, attr) or 0 for s in ss) / n
    score = (
        avg("accuracy_percentage") * 0.30
        + avg("average_rom")       * 0.20
        + avg("stability_score")   * 0.25
        + avg("balance_score")     * 0.25
    )
    if n >= 3:
        score += min(n * 2, 10)
    return min(score, 100.0)


def calculate_improvement(sessions: list) -> float:
    exercise_sessions, _ = split_sessions(sessions)
    ss = exercise_sessions or sessions
    if len(ss) < 2:
        return 0.0
    ss = sorted(ss, key=lambda s: s.start_time)
    def _avg(s):
        return (s.accuracy_percentage + s.average_rom + s.stability_score + s.balance_score) / 4
    first, last = _avg(ss[0]), _avg(ss[-1])
    if first == 0:
        return 0.0
    return max(-100.0, min(100.0, ((last - first) / first) * 100))


def session_summary(s: SessionModel) -> dict:
    return {
        "session_id":          s.id,
        "exercise_type":       s.exercise_type,
        "duration":            s.duration_seconds,
        "total_reps":          s.total_reps,
        "completed_reps":      s.completed_reps,
        "accuracy":            s.accuracy_percentage,
        "rom":                 s.average_rom,
        "stability":           s.stability_score,
        "balance":             s.balance_score,
        "smoothness":          s.movement_smoothness,
        "fatigue":             s.fatigue_estimation,
        "recovery_score":      s.recovery_score,
        "incorrect_movements": s.incorrect_movements,
        "date":                s.start_time.strftime("%Y-%m-%d %H:%M"),
        "is_game_session":     is_game_session(s),
        "steps":               game_steps(s) if is_game_session(s) else None,
    }