import numpy as np
import cv2

_model = None

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
    global _model
    if _model is None:
        from deepface import DeepFace

        dummy = np.zeros((224, 224, 3), dtype=np.uint8)
        try:
            DeepFace.represent(
                dummy,
                model_name="ArcFace",
                enforce_detection=False,
                detector_backend="skip",
                align=False,
            )
        except Exception:
            pass
        _model = DeepFace
    return _model

def _represent_rgb(DeepFace, rgb, aligned=False):
    try:
        return DeepFace.represent(
            rgb,
            model_name="ArcFace",
            enforce_detection=False,
            detector_backend="skip",
            align=False,
        )
    except Exception:
        if aligned:
            raise

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
        return None

def cosine_similarity(v1, v2):
    if v1 is None or v2 is None:
        return 0.0
    return float(np.dot(v1, v2))