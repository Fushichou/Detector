import cv2

# ===== Haar Face Detection Constants =====
_haar = None
MAX_FACE_SCAN_W = 260  # หากกว้าง > นี้ → ย่อ ROI ก่อนสแกน (ลด CPU load)


def _get_haar():
    """
    LAZY LOAD: Haar cascade classifier
    - โหลด 1 ครั้งตอน first call แล้ว reuse
    - ลดจำนวนครั้ง disk I/O
    """
    global _haar
    if _haar is None:
        _haar = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _haar


def detect_face(roi):
    """
    HAAR CASCADE: ตรวจจับใบหน้าใน ROI ของคน
    
    Optimization:
    - หากกว้าง > MAX_FACE_SCAN_W → ย่อลง เพื่อลด cascade complexity
    - ทำให้ HAAR scan เร็ว 4x
    - Scale back พิกัด → ขนาด ROI ต้นแบบ
    - equalizeHist ช่วย detect ใบหน้า ที่แสงมืด
    
    Returns: list of (x, y, w, h) ใน ROI coordinates
    """
    if roi is None or roi.size == 0:
        return []

    # ===== SCALE DOWN to SCAN =====
    scale = 1.0
    scan = roi
    if roi.shape[1] > MAX_FACE_SCAN_W:
        # ย่อเพื่อลด computation
        scale = MAX_FACE_SCAN_W / roi.shape[1]
        scan = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # ===== PREPROCESSING =====
    gray = cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # ปรับ contrast สำหรับ lighting variation

    # ===== CASCADE DETECT =====
    faces = _get_haar().detectMultiScale(
        gray,
        scaleFactor=1.1,   # ความเร็ว vs ที่ใหญ่แตกต่าง
        minNeighbors=7,    # บุคคลกรรม neighbor ต่ำ → detect ได้ง่ายขึ้น
        minSize=(50, 50),  # ตัดใบหน้าเล็กๆ (noise)
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    if len(faces) == 0:
        return []

    # ===== SCALE BACK =====
    inv = 1.0 / scale
    return [
        (int(x * inv), int(y * inv), int(w * inv), int(h * inv))
        for (x, y, w, h) in faces
    ]


def crop_face_fixed(frame, fx, fy, fw, fh, size=112):
    """Crop ใบหน้าพร้อม padding แล้ว resize เป็น size x size สำหรับ embedding."""
    pad = int(max(fw, fh) * 0.2)
    x1 = max(0, fx - pad)
    y1 = max(0, fy - pad)
    x2 = min(frame.shape[1], fx + fw + pad)
    y2 = min(frame.shape[0], fy + fh + pad)

    face_crop = frame[y1:y2, x1:x2]
    if face_crop.size == 0:
        return None

    return cv2.resize(face_crop, (size, size), interpolation=cv2.INTER_AREA)
