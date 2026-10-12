"""
Registry of available games for the session page's "Game" mode dropdown.
"""
from game.rehab_runner.worker import RehabRunnerEngine
from game.balloon_pop.worker import BalloonPopEngine

GAMES = {
    "rehab_runner": {
        "label": "Rehab Runner",
        "description": "Step-driven runner — step in place to advance, side-step to dodge, crouch under obstacles.",
        "engine_cls": RehabRunnerEngine,
        # eng.get_status()["steps"] is what "Target Steps" / auto-finish
        # is measured against for this game.
        "target_metric": "steps",
        "target_label": "Target Steps",
        "target_default": 30,
    },
    "balloon_pop": {
        "label": "Balloon Pop",
        "description": "Shoulder rehab - raise your arm to the side or forward to touch and pop the balloons.",
        "engine_cls": BalloonPopEngine,
        # eng.get_status()["reps"] = balloons popped so far.
        "target_metric": "reps",
        "target_label": "Target Reps",
        "target_default": 10,
    },
}


def get_game(game_id: str):
    """Return the registry entry for game_id, or None if not registered."""
    return GAMES.get(game_id)


def list_games():
    """[{id, label, description, target_label, target_default, config_schema}, ...]
    for the session page's game dropdown. config_schema comes from the engine
    class (config_schema() staticmethod); games without options report []."""
    return [
        {
            "id": gid,
            "label": g["label"],
            "description": g["description"],
            "target_label": g.get("target_label", "Target Steps"),
            "target_default": g.get("target_default", 0),
            "config_schema": getattr(g["engine_cls"], "config_schema", lambda: [])(),
        }
        for gid, g in GAMES.items()
    ]


def get_target_metric(game_id: str) -> str:
    """Which eng.get_status() key counts toward the configured target for
    game_id. Defaults to "steps" if a game entry doesn't declare one."""
    entry = GAMES.get(game_id) or {}
    return entry.get("target_metric", "steps")