import os


def _env_int(name, default):
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name, default):
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# Defaults target low-spec machines. Override with environment variables later
# without changing code, for example DETECT_PROFILE=balanced.
PROFILE = os.getenv("DETECT_PROFILE", "low").strip().lower()

PROFILES = {
    "low": {
        "camera_width": 640,
        "camera_height": 480,
        "camera_fps": 24,
        "yolo_img_size": 384,
        "yolo_max_det": 12,
        "gui_interval_ms": 50,
        "human_detect_interval": 0.12,
        "max_recognition_tracks": 3,
        "max_pending_recognition": 1,
        "min_recognition_face_px": 36,
        "min_face_sharpness": 18.0,
        "face_scan_unknown": 0.45,
        "face_scan_known": 1.00,
        "embed_unknown": 1.50,
        "embed_known": 4.00,
    },
    "balanced": {
        "camera_width": 640,
        "camera_height": 480,
        "camera_fps": 30,
        "yolo_img_size": 416,
        "yolo_max_det": 20,
        "gui_interval_ms": 33,
        "human_detect_interval": 0.07,   # เร็วขึ้น: 80ms → 70ms
        "max_recognition_tracks": 6,     # เพิ่มจาก 5 → 6
        "max_pending_recognition": 2,
        "min_recognition_face_px": 30,   # ผ่อนจาก 32 → 30
        "min_face_sharpness": 14.0,      # ผ่อนจาก 15 → 14
        "face_scan_unknown": 0.25,       # เร็วขึ้น: 300ms → 250ms
        "face_scan_known": 0.70,         # เร็วขึ้น: 800ms → 700ms
        "embed_unknown": 0.90,           # เร็วขึ้น: 1.00s → 0.90s
        "embed_known": 2.20,             # เร็วขึ้น: 2.50s → 2.20s
    },
    "high": {
        # สำหรับเครื่องที่แรง (i7/Ryzen 5+ หรือ GPU)
        "camera_width": 1280,
        "camera_height": 720,
        "camera_fps": 30,
        "yolo_img_size": 640,
        "yolo_max_det": 30,
        "gui_interval_ms": 25,
        "human_detect_interval": 0.05,
        "max_recognition_tracks": 8,
        "max_pending_recognition": 3,
        "min_recognition_face_px": 28,
        "min_face_sharpness": 12.0,
        "face_scan_unknown": 0.18,
        "face_scan_known": 0.55,
        "embed_unknown": 0.70,
        "embed_known": 1.80,
    },
}

SETTINGS = PROFILES.get(PROFILE, PROFILES["low"])

CAMERA_WIDTH = _env_int("CAMERA_WIDTH", SETTINGS["camera_width"])
CAMERA_HEIGHT = _env_int("CAMERA_HEIGHT", SETTINGS["camera_height"])
CAMERA_FPS = _env_int("CAMERA_FPS", SETTINGS["camera_fps"])
YOLO_IMG_SIZE = _env_int("YOLO_IMG_SIZE", SETTINGS["yolo_img_size"])
YOLO_MAX_DET = _env_int("YOLO_MAX_DET", SETTINGS["yolo_max_det"])
GUI_INTERVAL_MS = _env_int("GUI_INTERVAL_MS", SETTINGS["gui_interval_ms"])
HUMAN_DETECT_INTERVAL = _env_float(
    "HUMAN_DETECT_INTERVAL",
    SETTINGS["human_detect_interval"],
)
MAX_RECOGNITION_TRACKS = _env_int(
    "MAX_RECOGNITION_TRACKS",
    SETTINGS["max_recognition_tracks"],
)
MAX_PENDING_RECOGNITION = max(
    1,
    _env_int("MAX_PENDING_RECOGNITION", SETTINGS["max_pending_recognition"]),
)
MIN_RECOGNITION_FACE_PX = _env_int(
    "MIN_RECOGNITION_FACE_PX",
    SETTINGS["min_recognition_face_px"],
)
MIN_FACE_SHARPNESS = _env_float("MIN_FACE_SHARPNESS", SETTINGS["min_face_sharpness"])
FACE_SCAN_INTERVAL_UNKNOWN = _env_float(
    "FACE_SCAN_INTERVAL_UNKNOWN",
    SETTINGS["face_scan_unknown"],
)
FACE_SCAN_INTERVAL_KNOWN = _env_float(
    "FACE_SCAN_INTERVAL_KNOWN",
    SETTINGS["face_scan_known"],
)
EMBED_INTERVAL_UNKNOWN = _env_float("EMBED_INTERVAL_UNKNOWN", SETTINGS["embed_unknown"])
EMBED_INTERVAL_KNOWN = _env_float("EMBED_INTERVAL_KNOWN", SETTINGS["embed_known"])