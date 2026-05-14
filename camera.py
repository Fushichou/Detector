import cv2

def open_camera(index=0):
    """
    CAMERA SETUP: เปิดกล้องพร้อม optimize buffers
    Optimization:
    - BUFFERSIZE=1: ลด latency โดย cache เฟรมลำสุดเท่านั้น
      → ถ้า default=30 อาจล้าหลัง 1 วิ
    Args:
        index: camera device (0=default, 1=external USB etc)
    Raises:
        RuntimeError: ถ้าเปิดกล้องไม่ได้
    """
    # เปิด camera หรือ video
    cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        raise RuntimeError(
            f"ไม่สามารถเปิด source={index} ได้"
        )
    # BUFFER OPTIMIZATION
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return cap
