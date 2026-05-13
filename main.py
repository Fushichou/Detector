"""
main.py — Face Recognition System (Optimized for low-end hardware)

3 threads แยกกัน ไม่ block กัน:
  Thread 1 (camera)  : อ่านกล้องเร็วที่สุด ไม่รอใคร
  Thread 2 (detect)  : YOLO + Haar + Embedding ทำงานหลังบ้าน
  Main thread (GUI)  : แสดงผลลื่นตลอด ใช้ผลเดิมระหว่างรอ
"""

import cv2
import time
import threading
import copy
from camera import open_camera
from detecthuman import detect_human
from detect_face import detect_face, crop_face_fixed
from face_embedding import get_embedding
from face_db import init_db, load_all, find_match
from gui import FaceRecognitionGUI

# ─── CONFIG ───────────────────────────────────────────────────────────────────
ALPHA     = 0.35
FACE_SIZE = 112

GREEN  = (0, 220, 80)
BLUE   = (255, 160, 50)
YELLOW = (0, 210, 255)
BLACK  = (0, 0, 0)


# ─── EMA TRACKING ─────────────────────────────────────────────────────────────
def ema_match(smooth_boxes, humans, alpha=ALPHA, dist_thresh=150):
    new_smooth = []
    used = set()
    for human in humans:
        x1, y1, x2, y2 = human["box"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        best_j, best_dist = None, float("inf")
        for j, sb in enumerate(smooth_boxes):
            if j in used:
                continue
            scx = (sb["box"][0] + sb["box"][2]) / 2
            scy = (sb["box"][1] + sb["box"][3]) / 2
            d = ((cx - scx)**2 + (cy - scy)**2) ** 0.5
            if d < best_dist:
                best_dist, best_j = d, j

        if best_j is not None and best_dist < dist_thresh:
            sb = smooth_boxes[best_j]
            used.add(best_j)
            ox1, oy1, ox2, oy2 = sb["box"]
            new_smooth.append({
                "box": (
                    ox1 + alpha * (x1 - ox1),
                    oy1 + alpha * (y1 - oy1),
                    ox2 + alpha * (x2 - ox2),
                    oy2 + alpha * (y2 - oy2),
                ),
                "conf":     human["conf"],
                "identity": sb.get("identity", "..."),
                "sim":      sb.get("sim", 0.0),
                "found":    sb.get("found", False),
                "face_box": sb.get("face_box", None),
                "face_img": sb.get("face_img", None),
            })
        else:
            new_smooth.append({
                "box":      (float(x1), float(y1), float(x2), float(y2)),
                "conf":     human["conf"],
                "identity": "...",
                "sim":      0.0,
                "found":    False,
                "face_box": None,
                "face_img": None,
            })
    return new_smooth


# ─── DRAW OVERLAY ─────────────────────────────────────────────────── ──────────
def draw_overlay(frame, tracked, fps):
    h, w = frame.shape[:2]

    for t in tracked:
        x1, y1, x2, y2 = map(int, t["box"])
        identity = t["identity"]
        sim      = t["sim"]
        found    = t["found"]
        face_box = t["face_box"]

        color = (180, 180, 180) if identity == "..." else (GREEN if found else (0, 80, 220))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{identity} ({sim:.0%})" if sim > 0 else identity
        lw, lh = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
        ly = max(y1 - 6, lh + 6)
        cv2.rectangle(frame, (x1, ly - lh - 4), (x1 + lw + 8, ly + 2), color, -1)
        cv2.putText(frame, label, (x1 + 4, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, BLACK, 1)

        if face_box:
            fx, fy, fw, fh = face_box
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), BLUE, 2)
            cs = 8
            for px, py, dx, dy in [
                (fx, fy, 1, 1), (fx+fw, fy, -1, 1),
                (fx, fy+fh, 1, -1), (fx+fw, fy+fh, -1, -1)
            ]:
                cv2.line(frame, (px, py), (px + dx*cs, py), BLUE, 2)
                cv2.line(frame, (px, py), (px, py + dy*cs), BLUE, 2)

    cv2.putText(frame, "Q=Quit  R=Reload DB", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (130, 130, 130), 1)
    return frame


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    init_db()
    db_records = load_all()
    print(f"[DB] โหลด {len(db_records)} ใบหน้า")

    cap = open_camera()

    # ── shared state (thread-safe ด้วย lock) ──────────────────────────────────
    lock          = threading.Lock()
    latest_frame  = [None]   # เฟรมล่าสุดจากกล้อง
    tracked_state = [[]]     # ผลลัพธ์ล่าสุดจาก detect thread
    stop_flag     = [False]

    # ── Thread 1: อ่านกล้อง ───────────────────────────────────────────────────
    def camera_thread():
        while not stop_flag[0]:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with lock:
                latest_frame[0] = frame   # แค่เก็บเฟรมล่าสุด ไม่ queue
        cap.release()

    # ── Thread 2: detect + recognize (หนัก ทำหลังบ้าน) ─────────────────────
    def detect_thread():
        nonlocal db_records
        tracked = []
        last_frame_id = id(None)

        while not stop_flag[0]:
            with lock:
                frame = latest_frame[0]

            if frame is None or id(frame) == last_frame_id:
                time.sleep(0.01)   # รอเฟรมใหม่
                continue

            last_frame_id = id(frame)
            f = frame.copy()

            # YOLO detect คน
            humans  = detect_human(f)
            tracked = ema_match(tracked, humans)

            # Haar + Embedding บน ROI
            for t in tracked:
                x1, y1, x2, y2 = map(int, t["box"])
                roi = f[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                faces = detect_face(roi)
                if not faces:
                    t["face_box"] = None
                    t["face_img"] = None
                    continue

                fx, fy, fw, fh = max(faces, key=lambda fc: fc[2] * fc[3])
                t["face_box"] = (fx + x1, fy + y1, fw, fh)

                face_img = crop_face_fixed(roi, fx, fy, fw, fh, FACE_SIZE)
                t["face_img"] = face_img

                emb = get_embedding(face_img)
                if emb is None:
                    continue

                name, sim = find_match(emb, db_records)
                t["identity"] = name
                t["sim"]      = sim
                t["found"]    = (name != "Unknown")

            # อัปเดต shared state
            with lock:
                tracked_state[0] = copy.deepcopy(tracked)

    # ── เริ่ม threads ──────────────────────────────────────────────────────────
    t1 = threading.Thread(target=camera_thread, daemon=True)
    t2 = threading.Thread(target=detect_thread,  daemon=True)
    t1.start()
    t2.start()

    # ── GUI loop (main thread) ─────────────────────────────────────────────────
    gui     = FaceRecognitionGUI(cam_w=640, cam_h=480)
    fps_buf = []
    t_prev  = time.perf_counter()

    def loop():
        nonlocal db_records, t_prev

        with lock:
            frame   = latest_frame[0]
            tracked = tracked_state[0]

        if frame is None:
            return

        # FPS
        t_now = time.perf_counter()
        fps_buf.append(1.0 / max(t_now - t_prev, 1e-6))
        t_prev = t_now
        if len(fps_buf) > 30:
            fps_buf.pop(0)
        fps = sum(fps_buf) / len(fps_buf)

        # วาด + ส่ง GUI
        display = draw_overlay(frame.copy(), tracked, fps)
        gui.update_camera(display)
        gui.update_stats(fps, len(tracked))
        gui.update_faces([
            {
                "id":       i,
                "face_img": t["face_img"],
                "identity": t["identity"],
                "sim":      t["sim"],
                "found":    t["found"],
            }
            for i, t in enumerate(tracked)
        ])

        # keyboard (ผ่าน cv2 window ไม่ได้เพราะไม่มี imshow → ใช้ bind แทน)

    def on_key(event):
        nonlocal db_records
        if event.char == 'q':
            stop_flag[0] = True
            gui.destroy()
        elif event.char == 'r':
            db_records = load_all()
            gui.set_status(f"Reload DB: {len(db_records)} ใบหน้า")
            print(f"[DB] Reload: {len(db_records)} ใบหน้า")

    gui.root.bind("<Key>", on_key)
    gui.run(loop)

    stop_flag[0] = True


if __name__ == "__main__":
    main()