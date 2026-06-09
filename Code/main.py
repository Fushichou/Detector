import queue
import threading
import time
import cv2
from collections import deque
from camera import ask_video_source, open_camera
from detect_face import crop_face_fixed, detect_face
from detecthuman import detect_human
from face_db import build_index, find_match, init_db, load_all
from face_embedding import get_embedding
from gui import FaceRecognitionGUI
from db_manager import DBManager
from runtime_config import (
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    EMBED_INTERVAL_KNOWN,
    EMBED_INTERVAL_UNKNOWN,
    FACE_SCAN_INTERVAL_KNOWN,
    FACE_SCAN_INTERVAL_UNKNOWN,
    GUI_INTERVAL_MS,
    HUMAN_DETECT_INTERVAL,
    MAX_PENDING_RECOGNITION,
    MAX_RECOGNITION_TRACKS,
    MIN_FACE_SHARPNESS,
    MIN_RECOGNITION_FACE_PX,
    PROFILE,
)

ALPHA = 0.35
FACE_SIZE = 112
IDENTITY_LOST_TIMEOUT = 1.25
TRACK_MIN_IOU = 0.03
TRACK_MIN_SIZE_RATIO = 0.45

GREEN = (0, 220, 80)
BLUE  = (255, 160, 50)
BLACK = (0, 0, 0)

def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)

def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1  = max(ax1, bx1)
    iy1  = max(ay1, by1)
    ix2  = min(ax2, bx2)
    iy2  = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = box_area(box_a) + box_area(box_b) - inter
    return inter / union if union > 0 else 0.0

def box_size_ratio(box_a, box_b):
    area_a = box_area(box_a)
    area_b = box_area(box_b)
    if area_a <= 0 or area_b <= 0:
        return 0.0
    return min(area_a, area_b) / max(area_a, area_b)

def clamp_box(box, width, height):
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2

def ema_match(smooth_boxes, humans, next_track_id, alpha=ALPHA, dist_thresh=150):
    new_smooth = []
    used = set()

    for human in humans:
        x1, y1, x2, y2 = human["box"]
        new_box = (x1, y1, x2, y2)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        best_j, best_score = None, float("inf")
        for j, old in enumerate(smooth_boxes):
            if j in used:
                continue
            ox1, oy1, ox2, oy2 = old["box"]
            old_box = (ox1, oy1, ox2, oy2)
            ocx = (ox1 + ox2) / 2
            ocy = (oy1 + oy2) / 2

            dist = ((cx - ocx) ** 2 + (cy - ocy) ** 2) ** 0.5
            if dist >= dist_thresh:
                continue

            iou        = box_iou(new_box, old_box)
            size_ratio = box_size_ratio(new_box, old_box)
            if iou < TRACK_MIN_IOU and size_ratio < TRACK_MIN_SIZE_RATIO:
                continue

            score = dist - (iou * 80) - (size_ratio * 20)
            if score < best_score:
                best_j, best_score = j, score

        if best_j is not None:
            old  = smooth_boxes[best_j]
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
                "track_id":          next_track_id["value"],
                "box":               (float(x1), float(y1), float(x2), float(y2)),
                "conf":              human["conf"],
                "identity":          "...",
                "sim":               0.0,
                "found":             False,
                "face_box":          None,
                "face_img":          None,
                "face_aligned":      False,
                "last_face_ts":      0.0,
                "last_face_seen_ts": 0.0,
                "last_embed_ts":     0.0,
                "last_result_ts":    0.0,
                "recognition_pending": False,
            }
            next_track_id["value"] += 1

        new_smooth.append(item)

    return new_smooth

def draw_overlay(frame, tracked):
    for t in tracked:
        box = clamp_box(t["box"], frame.shape[1], frame.shape[0])
        if box is None:
            continue
        x1, y1, x2, y2 = box

        identity = t.get("identity", "...")
        sim      = t.get("sim", 0.0)
        found    = t.get("found", False)
        pending  = t.get("recognition_pending", False)
        face_box = t.get("face_box")

        color = (180, 180, 180) if identity == "..." else (GREEN if found else (0, 80, 220))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{identity} ({sim:.0%})" if sim > 0 else identity
        if pending and identity == "...":
            label = "..."

        lw, lh = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
        ly = max(y1 - 6, lh + 6)
        cv2.rectangle(frame, (x1, ly - lh - 4), (x1 + lw + 8, ly + 2), color, -1)
        cv2.putText(frame, label, (x1 + 4, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.55, BLACK, 1)

        if face_box:
            fx, fy, fw, fh = face_box
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), BLUE, 2)

    return frame

def face_quality_ok(face_img, face_w, face_h, roi_crop=None) -> bool:
    if face_img is None:
        return False
    if min(face_w, face_h) < MIN_RECOGNITION_FACE_PX:
        return False
    src = roi_crop if (roi_crop is not None and roi_crop.size > 0) else face_img
    gray      = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    return sharpness >= MIN_FACE_SHARPNESS

def _recognition_failure(job) -> dict:
    return {
        "track_id":    job["track_id"],
        "name":        None,
        "sim":         0.0,
        "found":       False,
        "submitted_ts": job["submitted_ts"],
    }

def recognition_worker(stop_event, job_queue, result_queue):
    while not stop_event.is_set():
        try:
            job = job_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        try:
            emb = get_embedding(job["face_img"], aligned=job["face_aligned"])
            if emb is None:
                result = _recognition_failure(job)
            else:
                name, sim = find_match(emb, job["db_records"])
                result = {
                    "track_id":    job["track_id"],
                    "name":        name,
                    "sim":         sim,
                    "found":       name != "Unknown",
                    "submitted_ts": job["submitted_ts"],
                }
        except Exception as exc:
            print(f"[Recognition] job failed: {exc}")
            result = _recognition_failure(job)
        finally:
            job_queue.task_done()

        result_queue.put(result)

def drain_recognition_results(tracked, result_queue):
    by_id = {item["track_id"]: item for item in tracked}
    while True:
        try:
            result = result_queue.get_nowait()
        except queue.Empty:
            break

        person = by_id.get(result["track_id"])
        if person is None:
            continue
        person["recognition_pending"] = False
        if result["submitted_ts"] < person.get("last_result_ts", 0.0):
            continue
        person["last_result_ts"] = result["submitted_ts"]
        if result["name"] is None:
            continue
        person["identity"] = result["name"]
        person["sim"]      = result["sim"]
        person["found"]    = result["found"]

def try_submit_recognition(person, db_records, job_queue, now):
    if person.get("recognition_pending", False):
        return
    face_img = person.get("face_img")
    if face_img is None:
        return
    embed_interval = EMBED_INTERVAL_KNOWN if person.get("found", False) else EMBED_INTERVAL_UNKNOWN
    if now - person.get("last_embed_ts", 0.0) < embed_interval:
        return
    job = {
        "track_id":    person["track_id"],
        "face_img":    face_img.copy(),
        "face_aligned": person.get("face_aligned", False),
        "db_records":  db_records,
        "submitted_ts": now,
    }
    try:
        job_queue.put_nowait(job)
    except queue.Full:
        return
    person["recognition_pending"] = True
    person["last_embed_ts"]       = now

def main():
    init_db()
    db_records = build_index(load_all())
    print(f"[Config] profile={PROFILE} camera={CAMERA_WIDTH}x{CAMERA_HEIGHT}@{CAMERA_FPS}")
    print(f"[DB] Loaded {len(db_records)} faces")

    # โหลดทุก model ล่วงหน้าใน background thread ก่อนคนแรกเดินเข้ากล้อง
    def _prewarm_embedding():
        from face_embedding import _get_model
        try:
            _get_model()
            print("[Init] ArcFace model ready")
        except Exception as e:
            print(f"[Init] ArcFace pre-warm failed: {e}")

    def _prewarm_yolo():
        from detecthuman import _get_model as yolo_get
        try:
            yolo_get()
            print("[Init] YOLO model ready")
        except Exception as e:
            print(f"[Init] YOLO pre-warm failed: {e}")

    def _prewarm_mediapipe():
        from detect_face import _get_detector
        try:
            _get_detector()
            print("[Init] MediaPipe detector ready")
        except Exception as e:
            print(f"[Init] MediaPipe pre-warm failed: {e}")

    for _fn in (_prewarm_embedding, _prewarm_yolo, _prewarm_mediapipe):
        threading.Thread(target=_fn, daemon=True).start()

    video_path = ask_video_source()
    if video_path:
        print(f"[Source] ใช้ไฟล์วิดีโอ: {video_path}")
        cap = open_camera(source=video_path)
    else:
        print("[Source] ไม่ได้เลือกไฟล์ → เปิดกล้อง")
        cap = open_camera(width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=CAMERA_FPS)

    stop_event    = threading.Event()
    state_lock    = threading.Lock()
    db_lock       = threading.Lock()
    latest_frame  = {"value": None}
    tracked_state = {"value": []}
    cam_fps_state = {"value": 0.0}   # FPS จริงของกล้อง วัดใน camera_thread
    recognition_jobs    = queue.Queue(maxsize=MAX_PENDING_RECOGNITION)
    recognition_results = queue.Queue()
    camera_switch_request = {"value": None}  # เก็บ source ใหม่ที่ต้องการสลับไป
    
    def camera_thread():
        is_video_file = video_path is not None
        if is_video_file:
            file_fps    = cap.get(cv2.CAP_PROP_FPS)
            frame_delay = 1.0 / file_fps if file_fps > 0 else 1.0 / 25
            print(f"[Camera] video FPS={file_fps:.2f} → frame delay={frame_delay*1000:.1f}ms")
        else:
            frame_delay = 0.0
        t_next = time.perf_counter()
        cam_fps_buf = deque(maxlen=30)
        t_cam_prev  = time.perf_counter()
        current_cap = cap  # เก็บ reference ของ camera object
        
        while not stop_event.is_set():
            # ตรวจสอบว่าต้องสลับ camera source หรือไม่
            with state_lock:
                switch_request = camera_switch_request["value"]
            
            if switch_request is not None and not is_video_file:
                # ถ้ามีขอให้สลับและไม่ได้ใช้ video file
                try:
                    print(f"[Camera] สลับ camera source → {switch_request}")
                    current_cap.release()
                    current_cap = open_camera(
                        source=switch_request,
                        width=CAMERA_WIDTH,
                        height=CAMERA_HEIGHT,
                        fps=CAMERA_FPS
                    )
                    new_w = int(current_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    new_h = int(current_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"[Camera] เปิด camera source {switch_request} สำเร็จ (resolution: {new_w}x{new_h})")
                    with state_lock:
                        camera_switch_request["value"] = None
                except Exception as e:
                    print(f"[Camera] ล้มเหลวในการสลับ camera: {e}")
                    with state_lock:
                        camera_switch_request["value"] = None
            
            ret, frame = current_cap.read()
            if not ret:
                if is_video_file:
                    current_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    t_next = time.perf_counter()
                else:
                    time.sleep(0.02)
                continue

            t_cam_now = time.perf_counter()
            cam_fps_buf.append(1.0 / max(t_cam_now - t_cam_prev, 1e-6))
            t_cam_prev = t_cam_now

            with state_lock:
                latest_frame["value"] = frame
                cam_fps_state["value"] = sum(cam_fps_buf) / len(cam_fps_buf)

            if is_video_file:
                t_next += frame_delay
                sleep_time = t_next - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    t_next = time.perf_counter()

        current_cap.release()

    def detect_thread():
        nonlocal db_records
        tracked        = []
        last_frame_obj = None
        last_detect_ts = 0.0
        next_track_id  = {"value": 1}

        while not stop_event.is_set():
            with state_lock:
                frame = latest_frame["value"]
            if frame is None or frame is last_frame_obj:
                time.sleep(0.005)
                continue

            last_frame_obj = frame
            now = time.perf_counter()
            drain_recognition_results(tracked, recognition_results)

            if now - last_detect_ts < HUMAN_DETECT_INTERVAL:
                continue
            last_detect_ts = now
            
            frame_h, frame_w = frame.shape[:2]
            try:
                humans = detect_human(frame)
            except Exception as exc:
                print(f"[Detect] human detection failed: {exc}")
                time.sleep(0.2)
                continue
            tracked = ema_match(tracked, humans, next_track_id)
            drain_recognition_results(tracked, recognition_results)
            face_candidates = sorted(
                tracked,
                key=lambda item: box_area(item["box"]),
                reverse=True,
            )[:MAX_RECOGNITION_TRACKS]
            for person in face_candidates:
                box = clamp_box(person["box"], frame_w, frame_h)
                if box is None:
                    continue
                x1, y1, x2, y2 = box
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                face_scan_interval = (
                    FACE_SCAN_INTERVAL_KNOWN
                    if person.get("found", False)
                    else FACE_SCAN_INTERVAL_UNKNOWN
                )
                if now - person.get("last_face_ts", 0.0) >= face_scan_interval:
                    person["last_face_ts"] = now
                    try:
                        faces = detect_face(roi, with_keypoints=True)
                    except Exception as exc:
                        print(f"[Detect] face detection failed: {exc}")
                        person["face_box"]     = None
                        person["face_img"]     = None
                        person["face_aligned"] = False
                        continue
                    if faces:
                        face = max(faces, key=lambda fc: fc["box"][2] * fc["box"][3])
                        fx, fy, fw, fh = face["box"]
                        person["last_face_seen_ts"] = now
                        person["face_box"] = (fx + x1, fy + y1, fw, fh)

                        face_img, face_aligned = crop_face_fixed(
                            roi, fx, fy, fw, fh, FACE_SIZE,
                            keypoints=face.get("keypoints"),
                            return_aligned=True,
                        )
                        raw_face = roi[fy:fy + fh, fx:fx + fw]
                        if raw_face.size > 0 and min(fw, fh) < 112:
                            raw_face_enhanced = cv2.resize(raw_face, (112, 112), interpolation=cv2.INTER_CUBIC)
                        else:
                            raw_face_enhanced = raw_face if raw_face.size > 0 else None

                        # ตรวจสอบคุณภาพโดยใช้ภาพที่ผ่านการ Enhanced แล้ว ขอบเขตความชัดจะทำงานได้ดีขึ้นในระยะไกล
                        if face_quality_ok(face_img, fw, fh, roi_crop=raw_face_enhanced):
                            person["face_img"]     = face_img
                            person["face_aligned"] = face_aligned
                        else:
                            # ถ้าภาพเบลอหรือเล็กเกินเกณฑ์จริง ค่อยเคลียร์ค่า
                            person["face_img"]     = None
                            person["face_aligned"] = False
                    else:
                        person["face_box"]     = None
                        person["face_img"]     = None
                        person["face_aligned"] = False
                if (
                    person.get("identity") != "..."
                    and person.get("last_face_seen_ts", 0.0) > 0
                    and now - person.get("last_face_seen_ts", 0.0) >= IDENTITY_LOST_TIMEOUT
                ):
                    person["identity"]      = "..."
                    person["sim"]           = 0.0
                    person["found"]         = False
                    person["last_embed_ts"] = 0.0

                with db_lock:
                    records_snapshot = db_records
                try_submit_recognition(person, records_snapshot, recognition_jobs, now)

            with state_lock:
                tracked_state["value"] = [item.copy() for item in tracked]

    threads = [
        threading.Thread(target=camera_thread, daemon=True),
        threading.Thread(target=detect_thread, daemon=True),
        threading.Thread(
            target=recognition_worker,
            args=(stop_event, recognition_jobs, recognition_results),
            daemon=True,
        ),
    ]
    for thread in threads:
        thread.start()

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if actual_w <= 0 or actual_h <= 0:
        actual_w, actual_h = CAMERA_WIDTH, CAMERA_HEIGHT
    print(f"[Source] resolution จริง: {actual_w}x{actual_h}")

    gui = FaceRecognitionGUI(cam_w=actual_w, cam_h=actual_h, interval_ms=GUI_INTERVAL_MS)

    # ── ตั้งค่า callback สำหรับสลับ camera source ───────────────────────────────────────────
    def switch_camera_source(source):
        """ขอให้สลับ camera source ในเธรด camera"""
        with state_lock:
            camera_switch_request["value"] = source
        camera_switch_status["source"] = source
        camera_switch_status["timestamp"] = time.perf_counter()
        gui.set_status(f"กำลังสลับเป็น Camera {source}...")
        print(f"[UI] ขอสลับเป็น camera source {source}")

    gui.set_camera_callback(switch_camera_source)
    
    # ตัวแปรเพื่อติดตามสถานะการสลับ camera
    camera_switch_status = {"source": None, "timestamp": 0}

    # ── ผูกปุ่ม "จัดการ DB" [จุดที่แก้ไขบั๊กค้าง] ───────────────────────────────────────────
    _db_win_open = {"value": False}   # ป้องกันเปิดซ้ำ

    def open_db_manager():
        nonlocal db_records
        if _db_win_open["value"]:
            return
        _db_win_open["value"] = True

        def on_db_close():
            # สั่งสร้าง Thread ย่อยแยกงานสกัดฟีเจอร์ของคนใหม่ออกไปทำใน Background ทันที
            # ป้องกันการแย่ง RAM/GPU ของตัวสตรีมหลักขณะที่โหลดรูปภาพชุดใหม่เข้าสู่ระบบ
            def bg_reload_db():
                nonlocal db_records
                try:
                    gui.set_status("กำลังอัปเดตและคำนวณฐานข้อมูลใบหน้าใหม่...")
                    print("[DB] เริ่มสแกนและโหลดข้อมูลคนใหม่เข้าฐานข้อมูลแบบ Background...")
                    
                    # คำนวณ Index โครงสร้างความเหมือนไว้บนตัวแปรจำลองก่อน
                    fresh = build_index(load_all())
                    
                    # เมื่อระบบหลังบ้านคำนวณเวกเตอร์เสร็จเรียบร้อย ค่อยสลับค่าใช้งานจริงภายในเสี้ยววินาที
                    with db_lock:
                        db_records = fresh
                        
                    gui.set_status(f"อัปเดตสำเร็จ! พร้อมตรวจจับใบหน้าใหม่แล้ว ({len(fresh)} ใบหน้า)")
                    print(f"[DB] ซิงค์ฐานข้อมูลคนใหม่เสร็จสิ้น! จำนวนใบหน้ารวม: {len(fresh)}")
                except Exception as ex:
                    gui.set_status("เกิดข้อผิดพลาดในการโหลดฐานข้อมูล!")
                    print(f"[DB] Background reloader ล้มเหลว: {ex}")
                finally:
                    _db_win_open["value"] = False

            # เริ่มระบบการประมวลผลหลังบ้านแบบ Async ทันทีที่หน้าต่างผู้จัดการปิดตัวลง
            threading.Thread(target=bg_reload_db, daemon=True).start()

        DBManager(parent=gui.root, on_close=on_db_close)

    gui.set_db_callback(open_db_manager)
    fps_buf = deque(maxlen=30)
    t_prev  = time.perf_counter()

    def loop():
        nonlocal t_prev
        with state_lock:
            frame    = latest_frame["value"]
            tracked  = tracked_state["value"]
            cam_fps  = cam_fps_state["value"]
            current_switch = camera_switch_request["value"]

        # ตรวจสอบว่าการสลับ camera เสร็จสิ้นแล้วหรือยัง
        if (
            current_switch is None
            and camera_switch_status["source"] is not None
            and time.perf_counter() - camera_switch_status["timestamp"] < 2.0
        ):
            # แสดงข้อความสำเร็จ (ทำให้เห็นนาน 2 วินาที)
            if time.perf_counter() - camera_switch_status["timestamp"] < 1.5:
                gui.set_status(f"✓ สลับเป็น Camera {camera_switch_status['source']} สำเร็จ")
        elif current_switch is None and camera_switch_status["source"] is not None:
            # หลังจาก 2 วินาที ให้ reset สถานะ
            camera_switch_status["source"] = None
            gui.set_status("พร้อมทำงาน")

        if frame is None:
            return
        t_now = time.perf_counter()
        fps_buf.append(1.0 / max(t_now - t_prev, 1e-6))
        t_prev = t_now
        gui_fps = sum(fps_buf) / len(fps_buf)

        display = draw_overlay(frame.copy(), tracked)
        gui.update_camera(display)
        gui.update_stats(cam_fps, gui_fps, len(tracked))
        gui.update_faces([
            {
                "id":       item.get("track_id", i),
                "face_img": item.get("face_img"),
                "identity": item.get("identity", "..."),
                "sim":      item.get("sim", 0.0),
                "found":    item.get("found", False),
            }
            for i, item in enumerate(tracked)
        ])

    def on_key(event):
        nonlocal db_records
        key = (event.char or "").lower()
        if key == "q":
            stop_event.set()
            gui.destroy()
        elif key == "r":
            # ปรับเปลี่ยนให้ปุ่มลัดคีย์บอร์ด "R" ทำงานใน Background Thread เช่นเดียวกันป้องกันหน้าจอค้าง
            def manual_reload():
                nonlocal db_records
                fresh = build_index(load_all())
                with db_lock:
                    db_records = fresh
                gui.set_status(f"Manual Reload สำเร็จ: {len(fresh)} ใบหน้า")
            threading.Thread(target=manual_reload, daemon=True).start()
        elif key == "0":
            # กด 0 เพื่อสลับไป camera source 0
            gui.camera_var.set("0")
            switch_camera_source(0)
        elif key == "1":
            # กด 1 เพื่อสลับไป camera source 1
            gui.camera_var.set("1")
            switch_camera_source(1)

    gui.root.bind("<Key>", on_key)
    try:
        gui.run(loop)
    finally:
        stop_event.set()
        for thread in threads:
            thread.join(timeout=1.0)

if __name__ == "__main__":
    main()