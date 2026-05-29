import numpy as np
import cv2
import traceback

_model = None
_model_load_error = None   # เก็บ error จากการโหลด model ครั้งแรก

def _normalize_vector(vec):
    """ทำ L2 Normalization ให้เวกเตอร์มีความยาวเท่ากับ 1 เสมอเพื่อเพิ่มความแม่นยำตอนเทียบ Cosine"""
    vec = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec

def _normalize_embedding(result):
    """สกัด Vector ออกจากออบเจกต์ที่ DeepFace ส่งกลับมา"""
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

        # เปลี่ยน dummy อุ่นเครื่องให้ได้ขนาด 112x112 ตรงตามคุณลักษณะจริงของ ArcFace เพื่อความเร็ว
        dummy = np.zeros((112, 112, 3), dtype=np.uint8)
        try:
            DeepFace.represent(
                dummy,
                model_name="ArcFace",
                enforce_detection=False,
                detector_backend="skip",
                align=False,
            )
        except Exception as e:
            print(f"[face_embedding] warm-up warning (ไม่ร้ายแรง): {e}")

        _model = DeepFace
    return _model

def get_embedding(face_img, aligned=True, augment=False):
    """
    สร้าง embedding จากภาพใบหน้าด้วยโครงสร้างสถาปัตยกรรมเดิม (DeepFace)
    [ปรับปรุงเพิ่มเติม]: ล็อกความเร็ว ป้องกันการแอบสแกนซ้ำ และรีไซส์ภาพให้เสถียรก่อนส่งตัวแปร
    """
    if face_img is None or face_img.size == 0:
        return None

    try:
        DeepFace_lib = _get_model()

        if face_img.shape[:2] != (112, 112):
            face_img = cv2.resize(face_img, (112, 112), interpolation=cv2.INTER_AREA)

        rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        res_normal = DeepFace_lib.represent(
            rgb,
            model_name="ArcFace",
            enforce_detection=False,
            detector_backend="skip",
            align=False,
        )
        embeddings = [_normalize_embedding(res_normal)]

        if augment:
            # ทำ Test-Time Augmentation (TTA) สลับภาพซ้ายขวาเพื่อถ่วงดุลความเสถียรเชิงมุม
            flipped = cv2.flip(rgb, 1)
            res_flipped = DeepFace_lib.represent(
                flipped,
                model_name="ArcFace",
                enforce_detection=False,
                detector_backend="skip",
                align=False,
            )
            embeddings.append(_normalize_embedding(res_flipped))

        # รวมเวกเตอร์และหาค่าเฉลี่ยด้วยมิติแนวแกนตั้ง
        return _normalize_vector(np.mean(embeddings, axis=0))

    except Exception:
        print("[face_embedding] get_embedding ล้มเหลว:")
        traceback.print_exc()
        return None

def cosine_similarity(v1, v2):
    if v1 is None or v2 is None:
        return 0.0
    return float(np.dot(v1, v2))