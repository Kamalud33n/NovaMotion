"""Left/right handling for stored keypoints.

The server flips every camera frame BEFORE MediaPipe runs (mjpeg_camera.py). For Pose this
makes the landmark NAMES (left_wrist, right_wrist, ...) the opposite of the patient's real
side (metrics.py applies the same correction to the UI side). Everything else that carries a
side -- recording meta `side`, game event `side`, the UI -- uses the PATIENT's real side.
Hands are not affected (MediaPipe Hands already assumes a mirrored image).

The recordings keep the raw MediaPipe order, so nothing is lost; use `to_patient_sides()`
when a model or a labelling screen needs "left" to mean the patient's left.
"""
from typing import Any, Dict

import numpy as np

# (left, right) index pairs of the 33 MediaPipe pose landmarks
POSE_LR_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
                 (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]


def pose_names_swapped(meta: Dict[str, Any]) -> bool:
    """True when this recording's pose left_*/right_* names are the opposite of the patient's side.
    Recordings made before the flag existed come from the same pipeline, so a missing flag on a
    pose recording counts as swapped."""
    if (meta or {}).get("kind") != "pose":
        return False
    flag = ((meta or {}).get("capture") or {}).get("pose_names_swapped")
    return True if flag is None else bool(flag)


def swap_pose_sides(pose: np.ndarray) -> np.ndarray:
    """Return a copy of a (..., 33, C) pose array with every left/right landmark pair exchanged."""
    arr = np.array(pose, copy=True)
    for left, right in POSE_LR_PAIRS:
        arr[..., [left, right], :] = arr[..., [right, left], :]
    return arr


def to_patient_sides(pose: np.ndarray, meta: Dict[str, Any]) -> np.ndarray:
    """Pose array whose 'left' landmarks are the patient's real left (no-op when already so)."""
    return swap_pose_sides(pose) if pose_names_swapped(meta) else np.array(pose, copy=True)