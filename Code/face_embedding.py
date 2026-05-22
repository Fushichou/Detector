import numpy as np
import cv2
import traceback

_model = None
_model_load_error = None   # เก็บ error จากการโหลด model ครั้งแรก

def _normalize_vector(vec):
    vec = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec

def _normalize_embedding(result):
    return _normalize_vector(result[0]["embedding"])

def _get_model():
    """Lazy load ArcFace via DeepFace with warm-up pass."""
    global _model, _model_load_error
    if _model_load_error is not None:
        raise RuntimeError(f"ArcFace โหลดไม่สำเร็จก่อนหน้านี้: {_model_load_error}")
    if _model is None:
        try:
            from deepface import DeepFace
        except ImportError as e:
            _model_load_error = str(e)
            raise RuntimeError(
                "ไม่พบ deepface — รัน: pip install deepface"
            ) from e

        dummy = np.zeros((224, 224, 3), dtype=np.uint8)
        try:
            DeepFace.represent(
                dummy,
                model_name="ArcFace",
                enforce_detection=False,
                detector_backend="skip",
                align=False,
            )
        except Exception as e:
            # warm-up อาจ fail บน dummy ดำล้วน — ไม่ถือว่า fatal
            print(f"[face_embedding] warm-up warning (ไม่ร้ายแรง): {e}")

        _model = DeepFace
    return _model

def _represent_rgb(DeepFace, rgb, aligned=False):
    """
    ลอง detector_backend="skip" ก่อน (เร็ว)
    ถ้า fail ให้ fallback เป็น opencv เสมอ ไม่ว่า aligned จะเป็นค่าใด
    """
    try:
        return DeepFace.represent(
            rgb,
            model_name="ArcFace",
            enforce_detection=False,
            detector_backend="skip",
            align=False,
        )
    except Exception as e:
        print(f"[face_embedding] skip-backend fail ({e}), ลอง opencv fallback...")

    return DeepFace.represent(
        rgb,
        model_name="ArcFace",
        enforce_detection=False,
        detector_backend="opencv",
        align=True,
    )

def get_embedding(face_img, aligned=False, augment=False):
    """
    สร้าง embedding จากภาพใบหน้า
    ถ้า aligned=True จะข้ามการตรวจจับซ้ำภายใน DeepFace
    ถ้า augment=True จะเฉลี่ย embedding กับภาพ flip แนวนอนด้วย
    คืน None เมื่อล้มเหลว (error จะถูก print เพื่อ debug)
    """
    if face_img is None or face_img.size == 0:
        return None

    try:
        DeepFace = _get_model()
        rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        embeddings = [_normalize_embedding(_represent_rgb(DeepFace, rgb, aligned))]

        if augment:
            flipped = cv2.flip(rgb, 1)
            embeddings.append(_normalize_embedding(_represent_rgb(DeepFace, flipped, aligned)))

        return _normalize_vector(np.mean(embeddings, axis=0))

    except Exception:
        print("[face_embedding] get_embedding ล้มเหลว:")
        traceback.print_exc()
        return None

def cosine_similarity(v1, v2):
    if v1 is None or v2 is None:
        return 0.0
    return float(np.dot(v1, v2))