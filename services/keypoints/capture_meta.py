"""Facts about HOW the keypoints were captured, saved in every recording's meta.

Without these, the same arrays are ambiguous for ML later: a flipped image swaps
left/right, a different model complexity changes landmark noise, and a different
frame size changes any pixel-space angle. Everything here is read from the live
config at recording start, so it cannot drift from what the pipeline really used.
"""
import re
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def _app_version() -> str:
    try:
        text = (_PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
        match = re.search(r'version\s*=\s*"([^"]+)"', text)
        return match.group(1) if match else "unknown"
    except Exception:
        return "unknown"


def build_capture_meta(hand_mode: bool = False) -> Dict[str, Any]:
    """Never raises: a failure here must not stop a recording from starting."""
    import config
    from services import mjpeg_camera
    from services.focus_lock import FocusConfig

    conf = config.MEDIAPIPE_MIN_CONFIDENCE
    meta: Dict[str, Any] = {
        "app_version": _app_version(),
        # The server flips every frame horizontally BEFORE inference (mjpeg_camera.py),
        # so all x values are in the flipped (mirror/selfie) image.
        "mirrored": True,
        "mirror_note": "x is normalised to the horizontally flipped frame; x_original_camera = 1 - x. "
                       "Left/right names are exactly as MediaPipe returned them on the flipped image.",
        "coordinates": "x,y: 0..1 of frame width/height (origin top-left); z: MediaPipe relative depth; "
                       "pose[...,3] = visibility 0..1",
        "timestamp_source": "server_unix_time_when_frame_was_processed",
        "timestamp_note": "includes network + decode delay, not the camera capture time",
        "pose_model": {
            "engine": "mediapipe.solutions.pose",
            "mediapipe_version": getattr(config.mp, "__version__", "unknown"),
            "model_complexity": config._pose_complexity(),
            "smooth_landmarks": config.POSE_SMOOTH_LANDMARKS,
            "min_detection_confidence": conf,
            "min_tracking_confidence": conf,
        },
        # Pose names vs patient side (see services/keypoints/laterality.py). Confirmed from a real
        # balloon_pop recording on 2026-10-21: left-hand pops showed the stored RIGHT wrist raised.
        "pose_names_swapped": True,
        "side_note": "meta.side, game-event side and the UI use the patient's REAL side. Pose landmark "
                     "names left_*/right_* are the OPPOSITE (frame is flipped before inference). "
                     "Hands slot 0/1 are NOT swapped.",
        "layout": {
            "pose": "33 MediaPipe pose landmarks, order = mediapipe PoseLandmark enum, values (x, y, z, visibility)",
            "hands": "slot 0 = left, slot 1 = right, 21 MediaPipe hand landmarks each, values (x, y, z)",
        },
    }
    if hand_mode:
        meta["hands_model"] = {
            "engine": "mediapipe.solutions.hands",
            "model_complexity": config.HANDS_MODEL_COMPLEXITY,
            "max_num_hands": 2,
            "min_detection_confidence": conf,
            "min_tracking_confidence": conf,
            "hand_label_swapped": bool(mjpeg_camera._HAND_LABEL_SWAPPED),
        }
    try:
        meta["focus_lock"] = asdict(FocusConfig.from_env())
    except Exception:
        pass
    return meta