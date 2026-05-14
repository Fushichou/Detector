from ultralytics import YOLO
import cv2

IMG_SIZE = 416
MIN_W = 30
MIN_H = 50

# โหลดและ fuse model ครั้งเดียว STARTUP เพื่อลดเวลา inference รอบถัดไป
# fuse(): รวม BN layers → fewer ops + ขนาด model เล็กลง
model = YOLO("yolo11n.pt")  # nano-sized: ความเร็ว > ความแม่นยำ
model.fuse()

def detect_human(frame):
    """
    YOLO11n detection: ตรวจจับคนจากเฟรมภาพ
    
    Pipeline:
    1. Resize frame → 416x416 (ลด inference time 4x vs 640x640)
    2. Run YOLO inference (classes=[0] = 'person' only)
    3. Scale back coordinates → frame size
    4. Filter: บุคคลต้องขนาดต่ำสุด MIN_W x MIN_H (remove noise)
    
    Returns: list of {"box": (x1,y1,x2,y2), "conf": 0-1}
    """
    # RESIZE: ลดความ resolution ก่อน → เร็ว  20ms → 5ms
    small = cv2.resize(frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    
    # YOLO INFERENCE: ส่งเฟรมเล็กเข้า model
    # conf=0.80: ลด false positive
    # iou=0.5: tight NMS → นับเป็นบุคคลต่างกัน
    # classes=[0]: ตรวจจับแค่ 'person' (skip ไอเทม อื่นๆ)
    # half=False: ใช้ FP32 (เสถียรกว่า เมื่อไม่ต้องการ FP16)
    # max_det=20: max 20 คนต่อเฟรม
    results = model(
        small,
        imgsz=IMG_SIZE,
        conf=0.80,
        iou=0.5,
        classes=[0],
        verbose=False,
        half=False,
    )

    h, w = frame.shape[:2]
    sx = w / IMG_SIZE  # scale x: small → frame
    sy = h / IMG_SIZE  # scale y: small → frame

    detections = []
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            # xyxy: x1, y1, x2, y2 (ใน 320x320 coordinates)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            # SCALE BACK: ทำให้เป็นขนาดเฟรมจริง
            x1 = max(0, int(x1 * sx))
            y1 = max(0, int(y1 * sy))
            x2 = min(w, int(x2 * sx))
            y2 = min(h, int(y2 * sy))
            # FILTER: บุคคล ต้องมีขนาดขั้นต่ำ
            if (x2 - x1) < MIN_W or (y2 - y1) < MIN_H:
                continue
            detections.append({"box": (x1, y1, x2, y2), "conf": conf})

    return detections
