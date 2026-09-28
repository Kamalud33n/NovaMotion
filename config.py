"""
Shared configuration / singletons used across the app:
- PDF report color tokens
- required directories
- MediaPipe Pose model (initialized once)
- key landmark index map + joint-angle helper
- shared Jinja2Templates instance (with the `tojson` filter registered)
"""
import os
import sys
import threading
import types
import json as _json
import warnings

warnings.filterwarnings("ignore")

# ── Block real `sounddevice` import before MediaPipe pulls it in ──────────
# `import mediapipe` unconditionally imports mediapipe.tasks.python, which
# imports its audio submodule, which imports `sounddevice` and calls
# PortAudio's Pa_Initialize() at import time. On some Windows machines
# (bad/absent audio drivers, sleeping Bluetooth audio devices, etc.) that
# call hangs forever — even though we never use MediaPipe's audio features,
# only Pose/Hands. Registering a fake `sounddevice` module in sys.modules
# BEFORE mediapipe is imported makes mediapipe's `import sounddevice`
# resolve instantly to this stub instead of loading the real PortAudio
# binding, so it never hangs. Safe because nothing in this app touches audio.
if "sounddevice" not in sys.modules:
    _fake_sd = types.ModuleType("sounddevice")
    _fake_sd.query_devices = lambda *a, **k: []
    _fake_sd.default = types.SimpleNamespace(device=(None, None))
    _fake_sd.InputStream = None
    _fake_sd.PortAudioError = Exception
    sys.modules["sounddevice"] = _fake_sd

import numpy as np
import mediapipe as mp
from reportlab.lib import colors
from fastapi.templating import Jinja2Templates

# PDF report design tokens (2-color system: navy + grey, white bg) 
PDF_NAVY        = colors.HexColor("#1B2A4A")   # headings / emphasis
PDF_GREY_BORDER = colors.HexColor("#B7BEC9")   # borders / rules
PDF_GREY_BG     = colors.HexColor("#EEF1F5")   # section header band
PDF_GREY_TEXT   = colors.HexColor("#5A6472")   # secondary/meta text
PDF_ROW_ALT     = colors.HexColor("#F7F8FA")   # alternating table row
PDF_BODY_TEXT   = colors.HexColor("#2B2B2B")   # body copy

# Directories 
for d in ("data", "reports", "uploads", "static", "templates", "assets"):
    os.makedirs(d, exist_ok=True)

# Shared Jinja2 templates instance (import this everywhere instead of
# creating a new Jinja2Templates(...) so the `tojson` filter is available
# in every router that renders HTML) 
templates = Jinja2Templates(directory="templates")
templates.env.filters["tojson"] = lambda obj: _json.dumps(obj)

# MediaPipe init (optimized)
# NOTE: Pose / Hands are NOT shared singletons any more. MediaPipe's tracker
# keeps per-video state between frames, so two doctors streaming at the same
# time on one shared Pose object would corrupt each other's tracking. Each
# logged-in user gets their own instances from create_pose() / create_hands()
# (see services/user_context.py); this module only exposes the factories plus
# the stateless drawing helpers / connection tables.
#
# min_detection_confidence / min_tracking_confidence 0.3 (was 0.5) so
# low-light / overexposed webcam frames still register a person;
# model_complexity 1 (was 0) for better accuracy, smooth_landmarks on to reduce
# jitter. If the server CPU is struggling with many users, set
# model_complexity back to 0 in create_pose().
try:
    _mp_pose = mp.solutions.pose
    mp_drawing        = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    POSE_CONNECTIONS  = _mp_pose.POSE_CONNECTIONS
    MEDIAPIPE_READY   = True
    print("MediaPipe Pose available (per-user instances, complexity=1, conf=0.3)")
except Exception as exc:
    print(f"MediaPipe init failed: {exc}")
    _mp_pose = mp_drawing = mp_drawing_styles = POSE_CONNECTIONS = None
    MEDIAPIPE_READY = False

try:
    _mp_hands = mp.solutions.hands
    HAND_CONNECTIONS = _mp_hands.HAND_CONNECTIONS
    print("MediaPipe Hands available (per-user instances, complexity=0, conf=0.3)")
except Exception as exc:
    print(f"MediaPipe Hands init failed: {exc}")
    _mp_hands = HAND_CONNECTIONS = None


def create_pose():
    """A NEW MediaPipe Pose tracker (one per user), or None if MediaPipe isn't usable."""
    if _mp_pose is None:
        return None
    try:
        return _mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )
    except Exception as exc:
        print(f"MediaPipe Pose create failed: {exc}")
        return None


def create_hands():
    """A NEW MediaPipe Hands tracker (one per user), or None if MediaPipe isn't usable."""
    if _mp_hands is None:
        return None
    try:
        return _mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )
    except Exception as exc:
        print(f"MediaPipe Hands create failed: {exc}")
        return None


# Hand landmark indices (21 points per hand) — MCP/PIP/DIP/TIP per finger,
# used for finger-curl angle calculation.
HAND_LANDMARKS = {
    "wrist": 0,
    "thumb_cmc": 1, "thumb_mcp": 2, "thumb_ip": 3, "thumb_tip": 4,
    "index_mcp": 5, "index_pip": 6, "index_dip": 7, "index_tip": 8,
    "middle_mcp": 9, "middle_pip": 10, "middle_dip": 11, "middle_tip": 12,
    "ring_mcp": 13, "ring_pip": 14, "ring_dip": 15, "ring_tip": 16,
    "pinky_mcp": 17, "pinky_pip": 18, "pinky_dip": 19, "pinky_tip": 20,
}

# Key landmarks only (13 joints instead of 33) 
# Indices: nose=0, shoulders=11/12, elbows=13/14, wrists=15/16,
#          hips=23/24, knees=25/26, ankles=27/28
KEY_LANDMARKS = {
    "nose": 0, "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14, "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
}


# Frame size used to un-normalise landmark coords in get_angle*(). It is
# thread-local because each frame is processed start-to-finish on ONE worker
# thread, while different users' frames (possibly different resolutions) run
# on different threads at the same time -- a plain global would let them
# overwrite each other's size mid-frame.
_frame_size = threading.local()


def set_frame_size(w: int, h: int):
    if w and h:
        _frame_size.w, _frame_size.h = float(w), float(h)


def _wh():
    return getattr(_frame_size, "w", 1.0), getattr(_frame_size, "h", 1.0)


def get_angle(p1, p2, p3) -> float:
    w, h = _wh()
    a = np.array([(p1.x - p2.x) * w, (p1.y - p2.y) * h, (p1.z - p2.z) * w])
    b = np.array([(p3.x - p2.x) * w, (p3.y - p2.y) * h, (p3.z - p2.z) * w])
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))))


def get_angle_2d(p1, p2, p3) -> float:
    w, h = _wh()
    a = np.array([(p1.x - p2.x) * w, (p1.y - p2.y) * h])
    b = np.array([(p3.x - p2.x) * w, (p3.y - p2.y) * h])
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))))