import cv2
import os
import tkinter as tk
from tkinter import filedialog

VIDEO_EXTENSIONS = (("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm *.m4v *.ts *.mpeg *.mpg"),("All files", "*.*"),)

def ask_video_source() -> str | None:
    """
    เปิด file dialog ให้ผู้ใช้เลือกไฟล์วิดีโอสำหรับทดสอบ
    - เลือกไฟล์  → คืน path ของไฟล์นั้น
    - กด Cancel  → คืน None (ระบบจะเปิดกล้องแทน)
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    path = filedialog.askopenfilename(
        title="เลือกไฟล์วิดีโอเพื่อทดสอบ (Cancel = ใช้กล้อง)",
        filetypes=VIDEO_EXTENSIONS,
    )
    root.destroy()
    return path if path else None

def open_camera(source=0, width=640, height=480, fps=30):
    """
    เปิดได้ทั้ง webcam (source=int) และ video file / RTSP / URL (source=str)
    """
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY

    if isinstance(source, int):
        cap = cv2.VideoCapture(source, backend)

        if not cap.isOpened() and backend != cv2.CAP_ANY:
            cap.release()
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            raise RuntimeError(f"ไม่สามารถเปิดกล้อง source={source} ได้")

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        if width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            cap.set(cv2.CAP_PROP_FPS, fps)
    else:
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            raise RuntimeError(f"ไม่สามารถเปิดไฟล์/stream ได้: {source}")
    return cap