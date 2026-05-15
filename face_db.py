"""
face_db.py
ฐานข้อมูลใบหน้า — เก็บ embedding vector ของแต่ละคน
ใช้ SQLite + numpy (ไม่ต้องติดตั้งอะไรเพิ่ม)

โครงสร้าง DB:
  persons(id, name, created_at)
  face_vectors(id, person_id, vector BLOB)
"""
import sqlite3
import numpy as np
import io
import os

DB_PATH = "faces.db"
THRESHOLD = 0.75  # cosine similarity ต่ำกว่านี้ = Unknown
MATCH_MARGIN = 0.035  # อันดับ 1 ต้องชนะอันดับ 2 อย่างน้อยเท่านี้
TOP_K_PER_PERSON = 3  # รวมคะแนนจาก vector ที่ดีที่สุดของแต่ละคน

def _connect():
    return sqlite3.connect(DB_PATH)


def _normalize_vector(vector: np.ndarray):
    vec = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec

def init_db():
    """สร้างตาราง (ถ้ายังไม่มี)"""
    con = _connect()
    con.executescript("""
        CREATE TABLE IF NOT EXISTS persons (
            id   INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS face_vectors (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER REFERENCES persons(id),
            vector    BLOB NOT NULL
        );
    """)
    con.commit()
    con.close()
    print(f"[DB] พร้อมใช้งาน: {os.path.abspath(DB_PATH)}")


# ─── เพิ่มข้อมูล ──────────────────────────────────────────────────────────────

def add_person(name):
    """เพิ่มชื่อคน คืนค่า person_id"""
    con = _connect()
    cur = con.execute("INSERT INTO persons (name) VALUES (?)", (name,))
    pid = cur.lastrowid
    con.commit()
    con.close()
    return pid


def add_vector(person_id, vector: np.ndarray):
    """เพิ่ม embedding vector ของคน"""
    vector = _normalize_vector(vector)
    buf = io.BytesIO()
    np.save(buf, vector)
    blob = buf.getvalue()

    con = _connect()
    con.execute(
        "INSERT INTO face_vectors (person_id, vector) VALUES (?, ?)",
        (person_id, blob)
    )
    con.commit()
    con.close()


def register_face(name, face_embedding: np.ndarray):
    """ลงทะเบียนใบหน้าใหม่ (สร้าง person + เพิ่ม vector)"""
    pid = add_person(name)
    add_vector(pid, face_embedding)
    print(f"[DB] ลงทะเบียน '{name}' (id={pid}) สำเร็จ")
    return pid


# ─── โหลด DB ──────────────────────────────────────────────────────────────────

def load_all():
    """
    โหลด embedding vectors ทั้งหมดจาก DB ได้ตลอด
    
    Returns: list of {
        "name": str,
        "vector": np.ndarray float32 shape (512,) unit vector
    }
    
    Used in: detect_thread (snapshot copy ด้วย db_lock)
    """
    con = _connect()
    rows = con.execute("""
        SELECT p.id, p.name, fv.vector
        FROM face_vectors fv
        JOIN persons p ON p.id = fv.person_id
    """).fetchall()
    con.close()

    records = []
    for person_id, name, blob in rows:
        buf = io.BytesIO(blob)
        vec = np.load(buf)  # numpy array เก็บ blob
        records.append({
            "person_id": person_id,
            "name": name,
            "vector": _normalize_vector(vec),
        })

    return records

# ─── ค้นหา ────────────────────────────────────────────────────────────────────
def find_match(query_vector: np.ndarray, db_records=None):

    if db_records is None:
        db_records = load_all()  # fallback: query DB if no snapshot

    if not db_records:
        return "Unknown", 0.0

    query_vector = _normalize_vector(query_vector)
    person_scores = {}

    # ===== BRUTE-FORCE VECTOR SCORES =====
    for rec in db_records:
        # dot product (เร็ว เพราะ normalize แล้ว)
        sim = float(np.dot(query_vector, rec["vector"]))
        key = rec.get("person_id", rec["name"])
        item = person_scores.setdefault(key, {"name": rec["name"], "scores": []})
        item["scores"].append(sim)

    ranked = []
    for item in person_scores.values():
        scores = sorted(item["scores"], reverse=True)
        top_scores = scores[:TOP_K_PER_PERSON]
        best = top_scores[0]
        top_mean = float(np.mean(top_scores))
        # Blend max and mean so one very good sample wins, but repeated good samples help.
        score = (best * 0.75) + (top_mean * 0.25)
        ranked.append({"name": item["name"], "score": score, "best": best})

    ranked.sort(key=lambda item: item["score"], reverse=True)
    best_match = ranked[0]
    second_score = ranked[1]["score"] if len(ranked) > 1 else -1.0

    # ===== THRESHOLD CHECK =====
    if best_match["score"] < THRESHOLD:
        return "Unknown", best_match["score"]

    if best_match["score"] - second_score < MATCH_MARGIN:
        return "Unknown", best_match["score"]

    return best_match["name"], best_match["score"]
# ─── ตัวอย่างการลงทะเบียน (รันตรงๆ) ─────────────────────────────────────────
if __name__ == "__main__":
    import sys
    init_db()

    if len(sys.argv) == 3 and sys.argv[1] == "add":
        # python face_db.py add "ชื่อ"  (ต้องเปิดกล้องถ่ายเอง)
        import cv2
        from camera import open_camera
        from detect_face import detect_face, crop_face_fixed
        from face_embedding import get_embedding

        name = sys.argv[2]
        cap = open_camera()
        print(f"กด SPACE เพื่อถ่ายภาพลงทะเบียน '{name}' | กด Q เพื่อออก")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Register", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                # detect face จากกล้อง
                faces = detect_face(frame, with_keypoints=True)
                if not faces:
                    print("ไม่พบใบหน้า ลองใหม่")
                    continue

                face = max(faces, key=lambda fc: fc["box"][2] * fc["box"][3])
                fx, fy, fw, fh = face["box"]
                face_img, face_aligned = crop_face_fixed(
                    frame,
                    fx,
                    fy,
                    fw,
                    fh,
                    keypoints=face.get("keypoints"),
                    return_aligned=True,
                )
                emb = get_embedding(face_img, aligned=face_aligned, augment=True)
                if emb is None:
                    print("ไม่สามารถสร้าง embedding ได้")
                    continue

                register_face(name, emb)
                break

            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
