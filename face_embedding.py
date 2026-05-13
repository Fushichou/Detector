"""
face_embedding.py
แปลงภาพใบหน้า (112x112) → vector ขนาด 512 มิติ
ใช้ DeepFace (backend: Facenet512) — ไม่ต้องติดตั้ง dlib
"""

import numpy as np
import cv2

# โหลด model ครั้งเดียว
_model = None

def _get_model():
    global _model
    if _model is None:
        from deepface import DeepFace
        # warm-up โหลด model เข้า memory
        dummy = np.zeros((112, 112, 3), dtype=np.uint8)
        try:
            DeepFace.represent(
                dummy,
                model_name="Facenet512",
                enforce_detection=False,
                detector_backend="skip"
            )
        except Exception:
            pass
        _model = DeepFace
    return _model


def get_embedding(face_img):
    """
    รับภาพใบหน้า BGR numpy array (112x112)
    คืนค่า numpy array ขนาด (512,) หรือ None ถ้า error
    """
    if face_img is None or face_img.size == 0:
        return None

    try:
        DeepFace = _get_model()
        rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        result = DeepFace.represent(
            rgb,
            model_name="Facenet512",
            enforce_detection=False,
            detector_backend="skip"   # ข้าม detect ซ้ำ เราทำแล้ว
        )

        vec = np.array(result[0]["embedding"], dtype=np.float32)
        # normalize เป็น unit vector
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    except Exception as e:
        print(f"[Embedding] Error: {e}")
        return None


def cosine_similarity(v1, v2):
    """คำนวณ cosine similarity ระหว่าง 2 vector (0-1, ยิ่งสูงยิ่งเหมือน)"""
    if v1 is None or v2 is None:
        return 0.0
    return float(np.dot(v1, v2))  # unit vector แล้ว dot = cosine
