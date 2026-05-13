import cv2

def open_camera(index=0):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"ไม่สามารถเปิดกล้อง index={index} ได้")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # ลด latency
    return cap
