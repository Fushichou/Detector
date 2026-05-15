import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import urllib.request
import os
import math
import numpy as np

# โหลด model file 
MODEL_PATH = "blaze_face_short_range.tflite"

def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("[FaceDetect] กำลังดาวน์โหลด MediaPipe model...")
        url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
        urllib.request.urlretrieve(url, MODEL_PATH)
        print("[FaceDetect] ดาวน์โหลดเสร็จแล้ว")

# Lazy load detector 
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


def _keypoints_to_pixels(det, width, height):
    keypoints = []
    for kp in det.keypoints or []:
        x = float(kp.x)
        y = float(kp.y)
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            x *= width
            y *= height
        keypoints.append((x, y))
    return keypoints


def detect_face(roi, with_keypoints=False):
    """
    ตรวจจับใบหน้าใน ROI ด้วย MediaPipe Tasks API (ใหม่)
    คืนค่า list of (x, y, w, h) ในระบบพิกัดของ roi
    ถ้า with_keypoints=True จะคืน dict {"box": ..., "keypoints": ...}
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

        box = (x1, y1, bw, bh)
        if with_keypoints:
            faces.append({
                "box": box,
                "keypoints": _keypoints_to_pixels(det, w, h),
            })
        else:
            faces.append(box)

    return faces


def _crop_aligned_by_eyes(frame, fx, fy, fw, fh, keypoints, size):
    if not keypoints or len(keypoints) < 2:
        return None

    eyes = sorted(keypoints[:2], key=lambda p: p[0])
    left_eye, right_eye = eyes[0], eyes[1]
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    eye_dist = math.hypot(dx, dy)
    if eye_dist < 8:
        return None

    center = ((left_eye[0] + right_eye[0]) / 2.0, (left_eye[1] + right_eye[1]) / 2.0)
    angle = math.degrees(math.atan2(dy, dx))
    rot = cv2.getRotationMatrix2D(center, angle, 1.0)

    h, w = frame.shape[:2]
    rotated = cv2.warpAffine(
        frame,
        rot,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    corners = np.array([
        [fx, fy],
        [fx + fw, fy],
        [fx + fw, fy + fh],
        [fx, fy + fh],
    ], dtype=np.float32)
    ones = np.ones((4, 1), dtype=np.float32)
    rotated_corners = np.hstack([corners, ones]).dot(rot.T)

    x1 = int(np.floor(rotated_corners[:, 0].min()))
    y1 = int(np.floor(rotated_corners[:, 1].min()))
    x2 = int(np.ceil(rotated_corners[:, 0].max()))
    y2 = int(np.ceil(rotated_corners[:, 1].max()))

    pad = int(max(x2 - x1, y2 - y1) * 0.2)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    face_crop = rotated[y1:y2, x1:x2]
    if face_crop.size == 0:
        return None

    return cv2.resize(face_crop, (size, size), interpolation=cv2.INTER_AREA)

def crop_face_fixed(frame, fx, fy, fw, fh, size=112, keypoints=None, return_aligned=False):
    """Crop ใบหน้าพร้อม padding แล้ว resize เป็น sizeXsize สำหรับ embedding"""
    aligned = _crop_aligned_by_eyes(frame, fx, fy, fw, fh, keypoints, size)
    if aligned is not None:
        return (aligned, True) if return_aligned else aligned

    pad = int(max(fw, fh) * 0.2)
    x1 = max(0, fx - pad)
    y1 = max(0, fy - pad)
    x2 = min(frame.shape[1], fx + fw + pad)
    y2 = min(frame.shape[0], fy + fh + pad)

    face_crop = frame[y1:y2, x1:x2]
    if face_crop.size == 0:
        return (None, False) if return_aligned else None

    resized = cv2.resize(face_crop, (size, size), interpolation=cv2.INTER_AREA)
    return (resized, False) if return_aligned else resized
