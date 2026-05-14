import numpy as np
import cv2

_model = None


def _normalize_embedding(result):
    vec = np.array(result[0]["embedding"], dtype=np.float32)
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

def get_embedding(face_img, aligned=False):
    """
    สร้าง EMBEDDING จากภาพใบหน้าด้วย ArcFace + Face Alignment
    """
    if face_img is None or face_img.size == 0:
        return None

    try:
        DeepFace = _get_model()
        rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        if aligned:
            try:
                result = DeepFace.represent(
                    rgb,
                    model_name="ArcFace",
                    enforce_detection=False,
                    detector_backend="skip",
                    align=False,
                )
                return _normalize_embedding(result)
            except Exception:
                pass

        result = DeepFace.represent(
            rgb,
            model_name="ArcFace",        # เปลี่ยนเป็น ArcFace (แม่นยำกว่า)
            enforce_detection=False,     
            detector_backend="opencv",   # fallback: ให้ DeepFace align เองเมื่อ MediaPipe align ไม่พอ
            align=True                   # สำคัญมาก! บังคับดัดหน้าตรงก่อน Extract Feature
        )

        return _normalize_embedding(result)

    except Exception as e:
        # หากมุมกล้องแย่เกินกว่าจะหา Landmark ได้ ให้ข้ามไป ไม่ต้อง print ให้รก console
        return None

def cosine_similarity(v1, v2):
    if v1 is None or v2 is None:
        return 0.0
    return float(np.dot(v1, v2))
