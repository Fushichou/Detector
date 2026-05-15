import numpy as np
import cv2

_model = None


def _normalize_embedding(result):
    vec = np.array(result[0]["embedding"], dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def _normalize_vector(vec):
    vec = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec

def _get_model():
    """LAZY LOAD + WARM-UP"""
    global _model
    if _model is None:
        from deepface import DeepFace
        # WARM-UP: โหลด ArcFace เข้า VRAM
        dummy = np.zeros((224, 224, 3), dtype=np.uint8)
        try:
            DeepFace.represent(
                dummy,
                model_name="ArcFace",
                enforce_detection=False,
                detector_backend="skip"
            )
        except Exception:
            pass
        _model = DeepFace
    return _model

# แก้ไขใน face_embedding.py

def _represent_rgb(DeepFace, rgb, aligned=False):
    if aligned:
        return DeepFace.represent(
            rgb,
            model_name="ArcFace",
            enforce_detection=False,
            detector_backend="skip",
            align=False
        )

    return DeepFace.represent(
        rgb,
        model_name="ArcFace",
        enforce_detection=False,
        detector_backend="opencv",
        align=True
    )


def get_embedding(face_img, aligned=False, augment=False):
    """
    สร้าง EMBEDDING จากภาพใบหน้า
    Optimization: ถ้าส่งภาพที่ Aligned มาแล้ว จะข้ามขั้นตอนตรวจจับซ้ำภายใน DeepFace
    """
    if face_img is None or face_img.size == 0:
        return None

    try:
        DeepFace = _get_model()
        # แปลงเป็น RGB สำหรับ DeepFace
        rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        embeddings = [_normalize_embedding(_represent_rgb(DeepFace, rgb, aligned))]

        if augment:
            flipped = cv2.flip(rgb, 1)
            embeddings.append(_normalize_embedding(_represent_rgb(DeepFace, flipped, aligned)))

        return _normalize_vector(np.mean(embeddings, axis=0))

    except Exception as e:
        # กรณีภาพมี noise มากจนทำ embedding ไม่ได้
        return None

def cosine_similarity(v1, v2):
    if v1 is None or v2 is None:
        return 0.0
    return float(np.dot(v1, v2))
