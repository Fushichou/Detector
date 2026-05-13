from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")
model.fuse()

def detect_human(frame):
    small = cv2.resize(frame, (320, 320))
    results = model(
        small,
        imgsz=320,
        conf=0.55,
        iou=0.45,
        classes=[0],
        verbose=False,
        half=False,
        max_det=20,
    )

    h, w = frame.shape[:2]
    sx = w / 320
    sy = h / 320

    detections = []
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1 = max(0, int(x1 * sx))
            y1 = max(0, int(y1 * sy))
            x2 = min(w, int(x2 * sx))
            y2 = min(h, int(y2 * sy))
            if (x2 - x1) < 30 or (y2 - y1) < 50:
                continue
            detections.append({"box": (x1, y1, x2, y2), "conf": conf})

    return detections
