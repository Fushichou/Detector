import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import urllib.request
import os

# ── โหลด model file ────────────────────────────────────────────────────────────
MODEL_PATH = "blaze_face_short_range.tflite"

def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("[FaceDetect] กำลังดาวน์โหลด MediaPipe model...")
        url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
        urllib.request.urlretrieve(url, MODEL_PATH)
        print("[FaceDetect] ดาวน์โหลดเสร็จแล้ว")

# ── Lazy load detector ─────────────────────────────────────────────────────────
_detector = None

def _get_detector():
    global _detector
    if _detector is None:
        _ensure_model()
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=0.6
        )
        _detector = vision.FaceDetector.create_from_options(options)
    return _detector


def detect_face(roi):
    """
    ตรวจจับใบหน้าใน ROI ด้วย MediaPipe Tasks API (ใหม่)
    คืนค่า list of (x, y, w, h) ในระบบพิกัดของ roi
    """
    if roi is None or roi.size == 0:
        return []

    h, w = roi.shape[:2]
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    results = _get_detector().detect(mp_image)

    if not results.detections:
        return []

    faces = []
    for det in results.detections:
        bb = det.bounding_box
        x1 = max(0, bb.origin_x)
        y1 = max(0, bb.origin_y)
        bw = bb.width
        bh = bb.height

        if bw < 20 or bh < 20:
            continue

        faces.append((x1, y1, bw, bh))

    return faces


def crop_face_fixed(frame, fx, fy, fw, fh, size=112):
    """Crop ใบหน้าพร้อม padding แล้ว resize เป็น size×size สำหรับ embedding"""
    pad = int(max(fw, fh) * 0.2)
    x1 = max(0, fx - pad)
    y1 = max(0, fy - pad)
    x2 = min(frame.shape[1], fx + fw + pad)
    y2 = min(frame.shape[0], fy + fh + pad)

    face_crop = frame[y1:y2, x1:x2]
    if face_crop.size == 0:
        return None

    return cv2.resize(face_crop, (size, size), interpolation=cv2.INTER_AREA)