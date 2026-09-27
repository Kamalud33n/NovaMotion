"""
Registry of available games for the session page's "Game" mode dropdown.

To add a new game later:
  1. Create game/<your_game>/ with an engine class (see rehab_runner/worker.py
     for the shape: reset(), request_calibration(), process_frame(lm, world,
     now, frame_shape), get_status(), get_summary()).
  2. Add one entry to GAMES below, including "target_metric": the key inside
     that engine's get_status() dict which counts toward the therapist's
     "target steps/reps" goal (services/game_state.py reads this generically
     to decide when a session should auto-finish — no other file changes).
  3. Add one entry to EXTRA_VIS_JOINTS below only if the calibration/status
     screen should insist on extra body parts beyond the default (head,
     shoulders, hip, knees, feet) that every engine already checks.
That's it — the session-page dropdown and the camera-loop wiring both read
this dict, so no other file needs to change to add a game.
"""
from game.rehab_runner.worker import RehabRunnerEngine

GAMES = {
    "rehab_runner": {
        "label": "Rehab Runner",
        "description": "Step-driven runner — step in place to advance, side-step to dodge, crouch under obstacles.",
        "engine_cls": RehabRunnerEngine,
        # eng.get_status()["steps"] is what "Target Steps" / auto-finish
        # is measured against for this game.
        "target_metric": "steps",
    },
}


def get_game(game_id: str):
    """Return the registry entry for game_id, or None if not registered."""
    return GAMES.get(game_id)


def list_games():
    """[{id, label, description}, ...] — for the session page's game dropdown."""
    return [
        {"id": gid, "label": g["label"], "description": g["description"]}
        for gid, g in GAMES.items()
    ]


def get_target_metric(game_id: str) -> str:
    """Which eng.get_status() key counts toward the configured target for
    game_id. Defaults to "steps" if a game entry doesn't declare one."""
    entry = GAMES.get(game_id) or {}
    return entry.get("target_metric", "steps")