"""
face_db.py — ฐานข้อมูลใบหน้า

เก็บ embedding vector ของแต่ละคนด้วย SQLite + numpy

โครงสร้าง DB:
  persons(id, name, created_at)
  face_vectors(id, person_id, vector BLOB)
"""

import io
import os
import sqlite3
import numpy as np

DB_PATH      = "faces.db"
THRESHOLD    = 0.75   # cosine similarity ต่ำกว่านี้ = Unknown
MATCH_MARGIN = 0.035  # อันดับ 1 ต้องชนะอันดับ 2 อย่างน้อยเท่านี้
TOP_K_PER_PERSON = 3  # รวมคะแนนจาก vector ที่ดีที่สุดของแต่ละคน

# ── In-memory index ──────────────────────────────────────────────────────────
class FaceIndex:
    """Matrix ในหน่วยความจำสำหรับ cosine search ที่รวดเร็ว"""

    def __init__(self, records):
        self.records = list(records)
        self.vectors = self._stack_vectors(self.records)
        self.groups  = self._group_by_person(self.records)

    def __bool__(self):  return bool(self.records)
    def __iter__(self):  return iter(self.records)
    def __len__(self):   return len(self.records)

    @staticmethod
    def _stack_vectors(records):
        if not records:
            return np.empty((0, 0), dtype=np.float32)
        return np.vstack([r["vector"] for r in records]).astype(np.float32, copy=False)

    @staticmethod
    def _group_by_person(records):
        groups = {}
        for idx, rec in enumerate(records):
            key = rec.get("person_id", rec["name"])
            item = groups.setdefault(key, {"name": rec["name"], "indices": []})
            item["indices"].append(idx)
        return [
            {"name": item["name"], "indices": np.asarray(item["indices"], dtype=np.intp)}
            for item in groups.values()
        ]

def build_index(records):
    if isinstance(records, FaceIndex):
        return records
    return FaceIndex(records)
#DB helpers 
def _connect():
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.execute("PRAGMA busy_timeout=10000")
    con.execute("PRAGMA foreign_keys=ON")
    return con

def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    vec = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec
#Schema 
def init_db():
    """สร้างตาราง (ถ้ายังไม่มี)"""
    con = _connect()
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript("""
        CREATE TABLE IF NOT EXISTS persons (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT NOT NULL,
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
# Write 
def add_person(name: str) -> int:
    """เพิ่มชื่อคน คืนค่า person_id"""
    con = _connect()
    cur = con.execute("INSERT INTO persons (name) VALUES (?)", (name,))
    pid = cur.lastrowid
    con.commit()
    con.close()
    return pid

def add_vector(person_id: int, vector: np.ndarray):
    """เพิ่ม embedding vector ของคน"""
    vector = _normalize_vector(vector)
    buf = io.BytesIO()
    np.save(buf, vector)

    con = _connect()
    con.execute(
        "INSERT INTO face_vectors (person_id, vector) VALUES (?, ?)",
        (person_id, buf.getvalue()),
    )
    con.commit()
    con.close()

def register_face(name: str, face_embedding: np.ndarray) -> int:
    """ลงทะเบียนใบหน้าใหม่ (สร้าง person + เพิ่ม vector)"""
    pid = add_person(name)
    add_vector(pid, face_embedding)
    print(f"[DB] ลงทะเบียน '{name}' (id={pid}) สำเร็จ")
    return pid
#Read 
def load_all() -> list[dict]:
    """
    โหลด embedding vectors ทั้งหมดจาก DB

    Returns: list of {"person_id": int, "name": str, "vector": np.ndarray}
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
        vec = np.load(io.BytesIO(blob))
        records.append({
            "person_id": person_id,
            "name":      name,
            "vector":    _normalize_vector(vec),
        })
    return records
#Search 
def _blend_person_score(scores: np.ndarray) -> float:
    if scores.size == 0:
        return 0.0
    if scores.size > TOP_K_PER_PERSON:
        top_scores = np.partition(scores, -TOP_K_PER_PERSON)[-TOP_K_PER_PERSON:]
    else:
        top_scores = scores
    best     = float(np.max(top_scores))
    top_mean = float(np.mean(top_scores))
    return (best * 0.75) + (top_mean * 0.25)

def find_match(query_vector: np.ndarray, db_records=None) -> tuple[str, float]:
    if db_records is None:
        db_records = build_index(load_all())
    else:
        db_records = build_index(db_records)

    if not db_records:
        return "Unknown", 0.0

    query_vector = _normalize_vector(query_vector)
    if db_records.vectors.shape[1] != query_vector.shape[0]:
        return "Unknown", 0.0

    similarities = db_records.vectors @ query_vector

    ranked = sorted(
        [
            {"name": item["name"], "score": _blend_person_score(similarities[item["indices"]])}
            for item in db_records.groups
        ],
        key=lambda x: x["score"],
        reverse=True,
    )

    best         = ranked[0]
    second_score = ranked[1]["score"] if len(ranked) > 1 else -1.0

    if best["score"] < THRESHOLD:
        return "Unknown", best["score"]

    if best["score"] - second_score < MATCH_MARGIN:
        return "Unknown", best["score"]

    return best["name"], best["score"]
# CLI helper 
if __name__ == "__main__":
    import sys

    init_db()

    if len(sys.argv) == 3 and sys.argv[1] == "add":
        import cv2
        from camera import open_camera
        from detect_face import detect_face, crop_face_fixed
        from face_embedding import get_embedding

        name = sys.argv[2]
        cap  = open_camera()
        print(f"กด SPACE เพื่อถ่ายภาพลงทะเบียน '{name}' | กด Q เพื่อออก")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Register", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                faces = detect_face(frame, with_keypoints=True)
                if not faces:
                    print("ไม่พบใบหน้า ลองใหม่")
                    continue

                face = max(faces, key=lambda fc: fc["box"][2] * fc["box"][3])
                fx, fy, fw, fh = face["box"]
                face_img, face_aligned = crop_face_fixed(
                    frame, fx, fy, fw, fh,
                    keypoints=face.get("keypoints"),
                    return_aligned=True,
                )
                emb = get_embedding(face_img, aligned=face_aligned, augment=True)
                if emb is None:
                    print("ไม่สามารถสร้าง embedding ได้")
                    continue

                register_face(name, emb)
                break

            elif key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()