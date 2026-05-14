"""
main.py - Real-time human + face recognition.

โครงสร้างหลัก:
1. camera_thread อ่านเฟรมล่าสุดตลอดเวลา เพื่อลด latency จาก buffer กล้อง
2. detect_thread ทำงานหนัก เช่น YOLO, Haar face, embedding แยกจาก GUI
3. Tkinter main thread แสดงผลจาก state ล่าสุด จึงไม่ค้างระหว่าง inference
"""

import threading
import time
import cv2
from camera import open_camera
from detect_face import crop_face_fixed, detect_face
from detecthuman import detect_human
from face_db import find_match, init_db, load_all
from face_embedding import get_embedding
from gui import FaceRecognitionGUI


# ปรับค่านี้ได้ถ้าเครื่องช้า/เร็วต่างกัน
ALPHA = 0.35
FACE_SIZE = 112
GUI_INTERVAL_MS = 33          # ประมาณ 30 FPS สำหรับ GUI
FACE_SCAN_INTERVAL = 0.50     # ตรวจหน้าไม่ถี่เกินไป ลดงาน Haar
EMBED_INTERVAL = 2.00         # สร้าง embedding ต่อคนไม่เกิน 1 ครั้ง/วินาที

GREEN = (0, 220, 80)
BLUE = (255, 160, 50)
BLACK = (0, 0, 0)

def ema_match(smooth_boxes, humans, alpha=ALPHA, dist_thresh=150):
    #จับคู่กล่องคนเฟรมใหม่กับกล่องเดิม แล้ว smooth เพื่อลดอาการกรอบสั่น.
    new_smooth = []
    used = set()

    for human in humans:
        x1, y1, x2, y2 = human["box"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        best_j, best_dist = None, float("inf")

        for j, old in enumerate(smooth_boxes):
            if j in used:
                continue
            ox1, oy1, ox2, oy2 = old["box"]
            ocx, ocy = (ox1 + ox2) / 2, (oy1 + oy2) / 2
            dist = ((cx - ocx) ** 2 + (cy - ocy) ** 2) ** 0.5
            if dist < best_dist:
                best_j, best_dist = j, dist

        if best_j is not None and best_dist < dist_thresh:
            old = smooth_boxes[best_j]
            used.add(best_j)
            ox1, oy1, ox2, oy2 = old["box"]
            item = old.copy()
            item.update({
                "box": (
                    ox1 + alpha * (x1 - ox1),
                    oy1 + alpha * (y1 - oy1),
                    ox2 + alpha * (x2 - ox2),
                    oy2 + alpha * (y2 - oy2),
                ),
                "conf": human["conf"],
            })
        else:
            item = {
                "box": (float(x1), float(y1), float(x2), float(y2)),
                "conf": human["conf"],
                "identity": "...",
                "sim": 0.0,
                "found": False,
                "face_box": None,
                "face_img": None,
                "face_aligned": False,
                "last_face_ts": 0.0,
                "last_embed_ts": 0.0,
            }

        new_smooth.append(item)

    return new_smooth


def draw_overlay(frame, tracked):
    #วาดกรอบคน/หน้า/ชื่อ ลงบนเฟรมที่ส่งมา.
    h = frame.shape[0]

    for t in tracked:
        x1, y1, x2, y2 = map(int, t["box"])
        identity = t.get("identity", "...")
        sim = t.get("sim", 0.0)
        found = t.get("found", False)
        face_box = t.get("face_box")

        color = (180, 180, 180) if identity == "..." else (GREEN if found else (0, 80, 220))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{identity} ({sim:.0%})" if sim > 0 else identity
        lw, lh = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
        ly = max(y1 - 6, lh + 6)
        cv2.rectangle(frame, (x1, ly - lh - 4), (x1 + lw + 8, ly + 2), color, -1)
        cv2.putText(frame, label, (x1 + 4, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.55, BLACK, 1)

        if face_box:
            fx, fy, fw, fh = face_box
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), BLUE, 2)
    return frame


def main():
    init_db()
    db_records = load_all()
    print(f"[DB] Loaded {len(db_records)} faces")

    cap = open_camera()
    stop_event = threading.Event()
    state_lock = threading.Lock()
    db_lock = threading.Lock()
    latest_frame = {"value": None}
    tracked_state = {"value": []}

    def camera_thread():
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with state_lock:
                latest_frame["value"] = frame  # Overwrite เฟรมเก่า
        cap.release()

    def detect_thread():
        nonlocal db_records
        tracked = []
        last_frame_obj = None

        while not stop_event.is_set():
            with state_lock:
                frame = latest_frame["value"]

            # ข้ามไปถ้า frame ยังไม่ได้อัปเดต หรือ frame เดิม
            if frame is None or frame is last_frame_obj:
                time.sleep(0.005)
                continue

            last_frame_obj = frame
            work = frame.copy()  # Copy เพื่อใช้ในฟังก์ชั่นแบบ stateless
            now = time.perf_counter()

            # ============ YOLO Detection ============
            # Detect ทุกเฟรม เพราะ YOLO เร็ว + ต้องติดตาม smooth
            humans = detect_human(work)
            # EMA smooth: ลด jitter ของ bounding box
            tracked = ema_match(tracked, humans)

            # ============ Per-Person Processing ============
            for person in tracked:
                x1, y1, x2, y2 = map(int, person["box"])
                roi = work[y1:y2, x1:x2]  # Crop ROI ของคนๆ นั้น
                if roi.size == 0:
                    continue

                if now - person.get("last_face_ts", 0.0) >= FACE_SCAN_INTERVAL:
                    person["last_face_ts"] = now
                    faces = detect_face(roi, with_keypoints=True)  # MediaPipe face + eye keypoints
                    if faces:
                        # เลือก face ที่ใหญ่ที่สุด (ปกติจะเป็นหน้าตรง)
                        face = max(faces, key=lambda fc: fc["box"][2] * fc["box"][3])
                        fx, fy, fw, fh = face["box"]
                        person["face_box"] = (fx + x1, fy + y1, fw, fh)
                        face_img, face_aligned = crop_face_fixed(
                            roi,
                            fx,
                            fy,
                            fw,
                            fh,
                            FACE_SIZE,
                            keypoints=face.get("keypoints"),
                            return_aligned=True,
                        )
                        person["face_img"] = face_img
                        person["face_aligned"] = face_aligned
                    else:
                        person["face_box"] = None
                        person["face_img"] = None
                        person["face_aligned"] = False

                # --- DeepFace Embedding (ช่วง 1.0s) ---
                # DeepFace (GPU) หนักสุด ถ้า embed ทุกเฟรม GPU จะ saturate
                if person.get("face_img") is None:
                    continue  # ยังไม่หา face ให้ skip
                if now - person.get("last_embed_ts", 0.0) < EMBED_INTERVAL:
                    continue  # โปรดปรานเวลา ยังไม่ถึง interval

                person["last_embed_ts"] = now
                emb = get_embedding(
                    person["face_img"],
                    aligned=person.get("face_aligned", False),
                )  # DeepFace ArcFace
                if emb is None:
                    continue

                # ค้นหาชื่อในฐานข้อมูล
                with db_lock:
                    records_snapshot = list(db_records)
                name, sim = find_match(emb, records_snapshot)
                person["identity"] = name
                person["sim"] = sim
                person["found"] = name != "Unknown"

            # Copy เฉพาะ dict ชั้นเดียว ไม่ deepcopy image numpy array
            # → ลด memory churn + GC pressure
            with state_lock:
                tracked_state["value"] = [item.copy() for item in tracked]

    t1 = threading.Thread(target=camera_thread, daemon=True)
    t2 = threading.Thread(target=detect_thread, daemon=True)
    t1.start()
    t2.start()

    gui = FaceRecognitionGUI(cam_w=640, cam_h=480, interval_ms=GUI_INTERVAL_MS)
    fps_buf = []
    t_prev = time.perf_counter()

    def loop():
        """GUI LOOP: อัปเดต UI 30 FPS โดยใช้ state ล่าสุดจาก threads"""
        nonlocal t_prev

        with state_lock:
            frame = latest_frame["value"]
            tracked = tracked_state["value"] 

        if frame is None:
            return 

        # ===== FPS Counter =====
        t_now = time.perf_counter()
        fps_buf.append(1.0 / max(t_now - t_prev, 1e-6))
        t_prev = t_now
        if len(fps_buf) > 30:
            fps_buf.pop(0)  # Moving average ของ 30 frame
        fps = sum(fps_buf) / len(fps_buf)

        # ===== Render & Display =====
        display = draw_overlay(frame.copy(), tracked)  # วาดกรอบ+ชื่อ
        gui.update_camera(display)  # Update main canvas
        gui.update_stats(fps, len(tracked))  # FPS + จำนวนคน
        gui.update_faces([  # Update thumbnail cards
            {
                "id": i,
                "face_img": item.get("face_img"),
                "identity": item.get("identity", "..."),
                "sim": item.get("sim", 0.0),
                "found": item.get("found", False),
            }
            for i, item in enumerate(tracked)
        ])

    def on_key(event):
        """KEYBOARD HANDLER: Q=Quit, R=Reload database"""
        nonlocal db_records
        key = (event.char or "").lower()
        if key == "q":
            # ออกจากโปรแกรม
            stop_event.set()  # Signal threads ให้หยุด
            gui.destroy()
        elif key == "r":
            # โหลด DB ใหม่ (เพื่ออัปเดตคนใหม่ที่เพิ่มเข้า DB)
            fresh = load_all()
            with db_lock:
                db_records = fresh
            gui.set_status(f"Reload DB: {len(fresh)} faces")
            print(f"[DB] Reload: {len(fresh)} faces")

    gui.root.bind("<Key>", on_key)
    gui.run(loop)
    stop_event.set()

if __name__ == "__main__":
    main()
