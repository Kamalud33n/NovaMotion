"""Recording settings, all read from .env."""
import os
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int((os.getenv(name) or "").strip())
    except ValueError:
        return default


@dataclass(frozen=True)
class KeypointConfig:
    enabled: bool = False
    directory: str = "data/keypoints"
    max_frames: int = 20000
    min_frames: int = 10
    max_events: int = 5000
    checkpoint_seconds: int = 90      # 0 = no partial checkpoints
    unlinked_ttl_hours: int = 72      # finished recording with no saved session: delete after this long (0 = keep forever)

    @classmethod
    def from_env(cls) -> "KeypointConfig":
        d = cls()
        return cls(
            enabled=_env_bool("KEYPOINT_RECORDING_ENABLED", d.enabled),
            directory=(os.getenv("KEYPOINT_DIR") or d.directory).strip() or d.directory,
            max_frames=max(100, _env_int("KEYPOINT_MAX_FRAMES", d.max_frames)),
            min_frames=max(1, _env_int("KEYPOINT_MIN_FRAMES", d.min_frames)),
            max_events=max(100, _env_int("KEYPOINT_MAX_EVENTS", d.max_events)),
            checkpoint_seconds=max(0, _env_int("KEYPOINT_CHECKPOINT_SECONDS", d.checkpoint_seconds)),
            unlinked_ttl_hours=max(0, _env_int("KEYPOINT_UNLINKED_TTL_HOURS", d.unlinked_ttl_hours)),
        )

    @property
    def base_path(self) -> Path:
        path = Path(self.directory)
        return path if path.is_absolute() else _PROJECT_ROOT / path