import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import os
import math
import numpy as np

# ── โหลด model file ────────────────────────────────────────────────────────────
MODEL_PATH = r"Model\blaze_face_short_range.tflite"
MIN_FACE_INPUT = 40
MIN_FACE_SIZE = 20
MAX_DETECT_DIM = 384

# พิกัดอ้างอิงมาตรฐาน (Standard Template 5 จุด) ขนาด 112x112 ที่ทำให้ ArcFace ทำงานได้แม่นยำที่สุด
ARC_FACE_TEMPLATE = np.array([
    [30.2946, 51.6963],  # ตาขวา (ในพิกัดภาพด้านซ้าย)
    [65.5318, 51.5014],  # ตาซ้าย (ในพิกัดภาพด้านขวา)
    [48.0252, 71.7366],  # ปลายจมูก
    [33.5493, 92.3655],  # มุมปากขวา (ในพิกัดภาพด้านซ้าย)
    [64.4416, 92.2041]   # มุมปากซ้าย (ในพิกัดภาพด้านขวา)
], dtype=np.float32)

def _ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    raise FileNotFoundError(
        f"Missing face detector model: {MODEL_PATH}. "
        "Place blaze_face_short_range.tflite in the project folder before running."
    )

# ── Lazy load detector ─────────────────────────────────────────────────────────
_detector = None

def _get_detector():
    global _detector
    if _detector is None:
        _ensure_model()
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=0.55
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

def _resize_for_detection(roi):
    h, w = roi.shape[:2]
    max_dim = max(h, w)
    if max_dim <= MAX_DETECT_DIM:
        return roi, 1.0

    scale = MAX_DETECT_DIM / max_dim
    resized = cv2.resize(
        roi,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale

def detect_face(roi, with_keypoints=False):
    """
    ตรวจจับใบหน้าใน ROI ด้วย MediaPipe Tasks API
    คืนค่า list of (x, y, w, h) ในระบบพิกัดของ roi
    ถ้า with_keypoints=True จะคืน dict {"box": ..., "keypoints": ...}
    """
    if roi is None or roi.size == 0:
        return []

    h, w = roi.shape[:2]
    if h < MIN_FACE_INPUT or w < MIN_FACE_INPUT:
        return []

    work, scale = _resize_for_detection(roi)
    work_h, work_w = work.shape[:2]
    rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    results = _get_detector().detect(mp_image)

    if not results.detections:
        return []

    faces = []
    for det in results.detections:
        bb = det.bounding_box
        x1 = max(0, int(bb.origin_x / scale))
        y1 = max(0, int(bb.origin_y / scale))
        x2 = min(w, int((bb.origin_x + bb.width) / scale))
        y2 = min(h, int((bb.origin_y + bb.height) / scale))
        bw = x2 - x1
        bh = y2 - y1

        if bw < MIN_FACE_SIZE or bh < MIN_FACE_SIZE:
            continue

        box = (x1, y1, bw, bh)
        if with_keypoints:
            keypoints = _keypoints_to_pixels(det, work_w, work_h)
            if scale != 1.0:
                keypoints = [(x / scale, y / scale) for x, y in keypoints]
            faces.append({
                "box": box,
                "keypoints": keypoints,
            })
        else:
            faces.append(box)

    return faces

def _align_by_multiple_keypoints(frame, keypoints, size):
    """
    เปลี่ยนจากการใช้ติ่งหู มาเป็นการจำลองพิกัดมุมปากซ้าย-ขวาอิงตามระนาบดวงตา 
    ทำให้ได้พิกัดสมมาตรครบ 5 จุดตามมาตรฐาน ArcFace หน้าจะตรง นิ่ง และแม่นยำสูงสุด
    """
    if not keypoints or len(keypoints) < 4:  # ต้องการอย่างน้อย ตาขวาภาพ, ตาซ้ายภาพ, จมูก, กึ่งกลางปาก
        return None

    try:
        r_eye = np.array(keypoints[0])         # ตาขวาในพิกัดภาพ (ตาซ้ายของคน)
        l_eye = np.array(keypoints[1])         # ตาซ้ายในพิกัดภาพ (ตาขวาของคน)
        nose = np.array(keypoints[2])          # ปลายจมูก
        mouth_center = np.array(keypoints[3])  # จุดกึ่งกลางปากจาก MediaPipe

        # 1. คำนวณหาระนาบความเอียงและระยะห่างของตาเพื่อใช้เป็นความกว้างอ้างอิง (Scale)
        eye_dx = l_eye[0] - r_eye[0]
        eye_dy = l_eye[1] - r_eye[1]
        eye_dist = math.hypot(eye_dx, eye_dy)

        if eye_dist < 4:
            return None

        # คำนวณหา Unit Vector แนวนอน (ux, uy) ตามทิศทางดวงตา ป้องกันบั๊กเวลาเอียงคอ
        ux = eye_dx / eye_dist
        uy = eye_dy / eye_dist

        # 2. จำลองพิกัด "มุมปากขวาภาพ" และ "มุมปากซ้ายภาพ" ออกมาจากจุดกึ่งกลางปาก
        # อิงตามสัดส่วนโครงสร้างสากล (ความกว้างปากซ้าย-ขวารวมกันจะประมาณ 50% ของความกว้างตา)
        mouth_width_half = eye_dist * 0.25 
        
        r_mouth = mouth_center - (np.array([ux, uy]) * mouth_width_half)
        l_mouth = mouth_center + (np.array([ux, uy]) * mouth_width_half)

        # 3. รวมจุดที่คำนวณใหม่ได้ครบ 5 จุดสากลตามลำดับของ Template
        pts_src = np.array([r_eye, l_eye, nose, r_mouth, l_mouth], dtype=np.float32)

        # 4. หา Affine Matrix สำหรับการจัดตำแหน่งแบบไร้ความบิดเบี้ยว (Similarity Transform)
        M, _ = cv2.estimateAffinePartial2D(pts_src, ARC_FACE_TEMPLATE)
        if M is None:
            M = cv2.getAffineTransform(pts_src[:3], ARC_FACE_TEMPLATE[:3])

        # ทำการตัดภาพและปรับให้หน้าตรงเป๊ะเป็นขนาด 112x112 ทันที
        aligned_face = cv2.warpAffine(frame, M, (112, 112), flags=cv2.INTER_CUBIC)

        # หากระบบต้องการขนาดอื่นนอกจาก 112 ให้ resize ปลายทาง
        if size != 112:
            aligned_face = cv2.resize(aligned_face, (size, size), interpolation=cv2.INTER_AREA)

        return aligned_face

    except Exception as e:
        print(f"[detect_face] Multi-point Alignment ล้มเหลว: {e}")
        return None

def crop_face_fixed(frame, fx, fy, fw, fh, size=112, keypoints=None, return_aligned=False):
    """Crop ใบหน้าพร้อมปรับทิศทางโครงสร้างและขนาดให้อยู่กึ่งกลางสากลสำหรับสกัด Embedding"""
    if keypoints is not None:
        aligned = _align_by_multiple_keypoints(frame, keypoints, size)
        if aligned is not None:
            return (aligned, True) if return_aligned else aligned

    # Fallback แบบ Padding เดิมกรณีโมเดลหา Keypoints ไม่เจอหรือไม่ครบจุด
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