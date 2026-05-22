import cv2
from pathlib import Path
from ultralytics import YOLO
from runtime_config import YOLO_IMG_SIZE, YOLO_MAX_DET

IMG_SIZE    = YOLO_IMG_SIZE
PERSON_CONF = 0.70
NMS_IOU     = 0.5
MAX_DET     = YOLO_MAX_DET
MIN_W       = 30
MIN_H       = 50

YOLO_MODEL_PATH = Path("Model\\yolov11n.pt")

try:
    import torch
except Exception:
    torch = None

USE_HALF = bool(torch is not None and torch.cuda.is_available())

_model = None

def _get_model():
    global _model
    if _model is None:
        _model = YOLO(YOLO_MODEL_PATH)
        if YOLO_MODEL_PATH.suffix == ".pt":
            _model.fuse()
    return _model


def _letterbox(frame, size=IMG_SIZE):
    """Resize with padding so YOLO sees a square image without aspect distortion."""
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        return None, 1.0, 0, 0

    scale = min(size / w, size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)

    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    padded = cv2.copyMakeBorder(
        resized,
        pad_y, size - new_h - pad_y,
        pad_x, size - new_w - pad_x,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return padded, scale, pad_x, pad_y

def detect_human(frame):
    """
    ตรวจจับคนจากเฟรมภาพด้วย YOLO11n

    Pipeline:
      1. Letterbox frame → IMG_SIZE×IMG_SIZE โดยไม่บิด aspect ratio
      2. Run YOLO inference (classes=[0] = person only)
      3. Scale coordinates กลับขนาดเฟรมจริง
      4. Filter ขนาดขั้นต่ำ MIN_W × MIN_H

    Returns: list of {"box": (x1, y1, x2, y2), "conf": float}
    """
    if frame is None or frame.size == 0:
        return []

    small, scale, pad_x, pad_y = _letterbox(frame)
    if small is None:
        return []

    results = _get_model()( 
        small,
        imgsz=IMG_SIZE,
        conf=PERSON_CONF,
        iou=NMS_IOU,
        classes=[0],
        max_det=MAX_DET,
        verbose=False,
        half=USE_HALF if YOLO_MODEL_PATH.suffix == ".pt" else False,
    )

    h, w = frame.shape[:2]
    detections = []
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1 = max(0, int((x1 - pad_x) / scale))
            y1 = max(0, int((y1 - pad_y) / scale))
            x2 = min(w, int((x2 - pad_x) / scale))
            y2 = min(h, int((y2 - pad_y) / scale))
            if (x2 - x1) < MIN_W or (y2 - y1) < MIN_H:
                continue
            detections.append({"box": (x1, y1, x2, y2), "conf": conf})

    return detections