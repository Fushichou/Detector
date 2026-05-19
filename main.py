import queue
import threading
import time
import cv2
from camera import ask_video_source, open_camera
from detect_face import crop_face_fixed, detect_face
from detecthuman import detect_human
from face_db import build_index, find_match, init_db, load_all
from face_embedding import get_embedding
from gui import FaceRecognitionGUI
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
ALPHA                = 0.35
FACE_SIZE            = 112
IDENTITY_LOST_TIMEOUT = 1.25
TRACK_MIN_IOU        = 0.03
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

def face_quality_ok(face_img, face_w, face_h) -> bool:
    if face_img is None:
        return False
    if min(face_w, face_h) < MIN_RECOGNITION_FACE_PX:
        return False
    gray      = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
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
    recognition_jobs    = queue.Queue(maxsize=MAX_PENDING_RECOGNITION)
    recognition_results = queue.Queue()
    
    def camera_thread():
        is_video_file = video_path is not None
        if is_video_file:
            file_fps    = cap.get(cv2.CAP_PROP_FPS)
            frame_delay = 1.0 / file_fps if file_fps > 0 else 1.0 / 25
            print(f"[Camera] video FPS={file_fps:.2f} → frame delay={frame_delay*1000:.1f}ms")
        else:
            frame_delay = 0.0
        t_next = time.perf_counter()
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                if is_video_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    t_next = time.perf_counter()
                else:
                    time.sleep(0.02)
                continue

            with state_lock:
                latest_frame["value"] = frame
            if is_video_file:
                t_next += frame_delay
                sleep_time = t_next - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    t_next = time.perf_counter()

        cap.release()

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
            work = frame.copy()
            frame_h, frame_w = work.shape[:2]
            try:
                humans = detect_human(work)
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
                roi = work[y1:y2, x1:x2]
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
                        if face_quality_ok(face_img, fw, fh):
                            person["face_img"]     = face_img
                            person["face_aligned"] = face_aligned
                        else:
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
    fps_buf = []
    t_prev  = time.perf_counter()

    def loop():
        nonlocal t_prev
        with state_lock:
            frame   = latest_frame["value"]
            tracked = tracked_state["value"]

        if frame is None:
            return
        t_now = time.perf_counter()
        fps_buf.append(1.0 / max(t_now - t_prev, 1e-6))
        t_prev = t_now
        if len(fps_buf) > 30:
            fps_buf.pop(0)
        fps = sum(fps_buf) / len(fps_buf)

        display = draw_overlay(frame.copy(), tracked)
        gui.update_camera(display)
        gui.update_stats(fps, len(tracked))
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
            fresh = build_index(load_all())
            with db_lock:
                db_records = fresh
            gui.set_status(f"Reload DB: {len(fresh)} faces")
            print(f"[DB] Reload: {len(fresh)} faces")

    gui.root.bind("<Key>", on_key)
    try:
        gui.run(loop)
    finally:
        stop_event.set()
        for thread in threads:
            thread.join(timeout=1.0)
if __name__ == "__main__":
    main()