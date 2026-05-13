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
THRESHOLD = 0.55  # cosine similarity ต่ำกว่านี้ = Unknown


def _connect():
    return sqlite3.connect(DB_PATH)


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
    โหลด vector ทั้งหมดจาก DB
    คืนค่า list of {"name": str, "vector": np.ndarray}
    """
    con = _connect()
    rows = con.execute("""
        SELECT p.name, fv.vector
        FROM face_vectors fv
        JOIN persons p ON p.id = fv.person_id
    """).fetchall()
    con.close()

    records = []
    for name, blob in rows:
        buf = io.BytesIO(blob)
        vec = np.load(buf)
        records.append({"name": name, "vector": vec})

    return records


# ─── ค้นหา ────────────────────────────────────────────────────────────────────

def find_match(query_vector: np.ndarray, db_records=None):
    """
    เปรียบเทียบ query_vector กับทุก vector ใน DB
    คืนค่า (name, similarity) หรือ ("Unknown", score)

    db_records: ถ้าส่งมาจะใช้เลย (ไม่ต้อง query DB ซ้ำ)
    """
    if db_records is None:
        db_records = load_all()

    if not db_records:
        return "Unknown", 0.0

    best_name = "Unknown"
    best_sim  = -1.0

    for rec in db_records:
        sim = float(np.dot(query_vector, rec["vector"]))
        if sim > best_sim:
            best_sim  = sim
            best_name = rec["name"]

    if best_sim < THRESHOLD:
        return "Unknown", best_sim

    return best_name, best_sim


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
                faces = detect_face(frame)
                if not faces:
                    print("ไม่พบใบหน้า ลองใหม่")
                    continue

                fx, fy, fw, fh = faces[0]
                face_img = crop_face_fixed(frame, fx, fy, fw, fh)
                emb = get_embedding(face_img)
                if emb is None:
                    print("ไม่สามารถสร้าง embedding ได้")
                    continue

                register_face(name, emb)
                break

            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
