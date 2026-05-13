import cv2

_haar = None

def _get_haar():
    global _haar
    if _haar is None:
        _haar = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _haar


def detect_face(roi):
    """
    ตรวจจับใบหน้าใน ROI (crop ของคน)
    คืนค่า list of (x, y, w, h) ในระบบพิกัดของ roi
    """
    if roi is None or roi.size == 0:
        return []

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = _get_haar().detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) == 0:
        return []

    return [(x, y, w, h) for (x, y, w, h) in faces]


def crop_face_fixed(frame, fx, fy, fw, fh, size=112):
    """
    Crop ใบหน้าจาก frame และ resize เป็นขนาดคงที่ size x size
    เพื่อนำไปแปลงเป็น embedding vector
    """
    pad = int(max(fw, fh) * 0.2)
    x1 = max(0, fx - pad)
    y1 = max(0, fy - pad)
    x2 = min(frame.shape[1], fx + fw + pad)
    y2 = min(frame.shape[0], fy + fh + pad)

    face_crop = frame[y1:y2, x1:x2]
    if face_crop.size == 0:
        return None

    face_resized = cv2.resize(face_crop, (size, size))
    return face_resized
