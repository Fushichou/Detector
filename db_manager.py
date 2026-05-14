"""
db_manager.py — จัดการฐานข้อมูลใบหน้า
- ดูรายชื่อทั้งหมด + จำนวน vector ของแต่ละคน
- เพิ่มคนใหม่ผ่านกล้อง (ถ่ายหลายรูปได้)
- ลบรายบุคคล
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import cv2
import threading
import time
from PIL import Image, ImageTk

from camera import open_camera
from detect_face import detect_face, crop_face_fixed
from face_embedding import get_embedding
from face_db import init_db, _connect, add_person, add_vector

# ── Palette ───────────────────────────────────────────────────────────────────
BG        = "#0d1117"
SURFACE   = "#161b22"
SURFACE2  = "#21262d"
BORDER    = "#30363d"
ACCENT    = "#238636"
ACCENT_H  = "#2ea043"
RED       = "#da3633"
RED_H     = "#f85149"
TEXT      = "#e6edf3"
MUTED     = "#8b949e"
BLUE      = "#58a6ff"

FONT_TITLE = ("Consolas", 15, "bold")
FONT_BODY  = ("Consolas", 10)
FONT_SMALL = ("Consolas", 9)

THUMB = 72   # ขนาด thumbnail ในรายการ


# ── DB helpers ────────────────────────────────────────────────────────────────
def db_list_persons():
    """คืนค่า list of (id, name, vector_count)"""
    con = _connect()
    rows = con.execute("""
        SELECT p.id, p.name, COUNT(fv.id) as cnt
        FROM persons p
        LEFT JOIN face_vectors fv ON fv.person_id = p.id
        GROUP BY p.id
        ORDER BY p.name
    """).fetchall()
    con.close()
    return rows   # [(id, name, cnt), ...]

def db_delete_person(person_id):
    con = _connect()
    con.execute("DELETE FROM face_vectors WHERE person_id=?", (person_id,))
    con.execute("DELETE FROM persons WHERE id=?", (person_id,))
    con.commit()
    con.close()

# ── Capture Window (กล้องเพื่อเพิ่มคน) ──────────────────────────────────────
class CaptureWindow:
    """
    หน้าต่างถ่ายรูปเพื่อลงทะเบียนใบหน้า
    - แสดงกล้องสด + กรอบใบหน้าแบบ real-time
    - กด CAPTURE หรือ SPACE เพื่อถ่าย
    - ถ่ายได้หลายรูป บันทึกทีเดียว
    """
    def __init__(self, parent, name, on_done):
        self.name    = name
        self.on_done = on_done
        self.cap     = open_camera()
        self.running = True
        self.samples = []          # รูปใบหน้าที่ถ่ายแล้ว
        self.face_box = None
        self.face_info = None
        self.latest_frame = None

        # ── Window ──────────────────────────────────────────────────────
        self.win = tk.Toplevel(parent)
        self.win.title(f"ลงทะเบียน: {name}")
        self.win.configure(bg=BG)
        self.win.resizable(False, False)
        self.win.grab_set()
        self.win.protocol("WM_DELETE_WINDOW", self._close)

        self._build_ui()

        # thread กล้อง
        self._cam_thread = threading.Thread(target=self._cam_loop, daemon=True)
        self._cam_thread.start()

        self.win.bind("<space>", lambda e: self._capture())
        self._tick()

    def _build_ui(self):
        # Header
        hdr = tk.Frame(self.win, bg=SURFACE, pady=10)
        hdr.pack(fill="x")
        tk.Label(hdr, text=f"ลงทะเบียน  {self.name}", font=FONT_TITLE, fg=TEXT, bg=SURFACE).pack()
        tk.Label(hdr, text="จัดหน้าให้อยู่ในกรอบ แล้วกด CAPTURE (หรือ SPACE)",
                 font=FONT_SMALL, fg=MUTED, bg=SURFACE).pack()

        # กล้อง
        self.cam_lbl = tk.Label(self.win, bg="#000")
        self.cam_lbl.pack()

        # ── Bottom bar ──────────────────────────────────────────────────
        bar = tk.Frame(self.win, bg=SURFACE, pady=10)
        bar.pack(fill="x")

        self.count_lbl = tk.Label(bar, text="รูปที่ถ่าย: 0",
                                  font=FONT_BODY, fg=MUTED, bg=SURFACE)
        self.count_lbl.pack(side="left", padx=16)

        tk.Button(bar, text="✖  ยกเลิก", font=FONT_BODY,
                  bg=SURFACE2, fg=RED, activebackground=RED, activeforeground=TEXT,
                  relief="flat", padx=12, pady=6,
                  command=self._close).pack(side="right", padx=8)

        self.save_btn = tk.Button(bar, text="บันทึก", font=FONT_BODY,
                                  bg=ACCENT, fg=TEXT, activebackground=ACCENT_H,
                                  relief="flat", padx=12, pady=6,
                                  command=self._save, state="disabled")
        self.save_btn.pack(side="right", padx=4)

        tk.Button(bar, text="CAPTURE", font=FONT_BODY,
                  bg=BLUE, fg=BG, activebackground="#79c0ff",
                  relief="flat", padx=12, pady=6,
                  command=self._capture).pack(side="right", padx=4)

    def _cam_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            # detect ใบหน้าแบบ lightweight เพื่อวาด guide
            faces = detect_face(frame, with_keypoints=True)
            face = max(faces, key=lambda fc: fc["box"][2] * fc["box"][3]) if faces else None
            self.face_info = face
            self.face_box = face["box"] if face else None
            self.latest_frame = frame

    def _tick(self):
        if not self.running:
            return
        frame = self.latest_frame
        if frame is not None:
            disp = frame.copy()
            # วาดกรอบใบหน้า
            if self.face_box:
                fx, fy, fw, fh = self.face_box
                cv2.rectangle(disp, (fx, fy), (fx+fw, fy+fh), (0, 210, 100), 2)
                # มุมเน้น
                cs = 12
                for px, py, dx, dy in [
                    (fx, fy, 1, 1), (fx+fw, fy, -1, 1),
                    (fx, fy+fh, 1, -1), (fx+fw, fy+fh, -1, -1)
                ]:
                    cv2.line(disp, (px, py), (px+dx*cs, py), (0, 210, 100), 3)
                    cv2.line(disp, (px, py), (px, py+dy*cs), (0, 210, 100), 3)
            else:
                cv2.putText(disp, "ไม่พบใบหน้า", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 80, 220), 2)

            # แปลงแสดง
            rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.cam_lbl.imgtk = imgtk
            self.cam_lbl.configure(image=imgtk)

        self.win.after(33, self._tick)

    def _capture(self):
        frame = self.latest_frame
        if frame is None:
            return
        if not self.face_box:
            messagebox.showwarning("ไม่พบใบหน้า", "กรุณาจัดหน้าให้อยู่ในกรอบก่อน",
                                   parent=self.win)
            return

        face_info = self.face_info or {}
        fx, fy, fw, fh = self.face_box
        face_img, face_aligned = crop_face_fixed(
            frame,
            fx,
            fy,
            fw,
            fh,
            size=112,
            keypoints=face_info.get("keypoints"),
            return_aligned=True,
        )
        if face_img is None:
            return

        self.samples.append({"image": face_img, "aligned": face_aligned})
        n = len(self.samples)
        self.count_lbl.configure(text=f"รูปที่ถ่าย: {n}")
        if n >= 1:
            self.save_btn.configure(state="normal")

        # flash ขอบเขียวสั้นๆ
        self.cam_lbl.configure(bg="#2ea043")
        self.win.after(120, lambda: self.cam_lbl.configure(bg="#000"))

    def _save(self):
        if not self.samples:
            return
        self.save_btn.configure(state="disabled", text="กำลังบันทึก...")
        self.win.update()

        def _do_save():
            pid = add_person(self.name)
            ok = 0
            for sample in self.samples:
                emb = get_embedding(
                    sample["image"],
                    aligned=sample.get("aligned", False),
                )
                if emb is not None:
                    add_vector(pid, emb)
                    ok += 1
            self.win.after(0, lambda: self._finish(ok))

        threading.Thread(target=_do_save, daemon=True).start()

    def _finish(self, ok):
        if ok == 0:
            messagebox.showerror("ผิดพลาด",
                                 "ไม่สามารถสร้าง embedding ได้\nลองถ่ายใหม่อีกครั้ง",
                                 parent=self.win)
            self.save_btn.configure(state="normal", text="บันทึก")
        else:
            messagebox.showinfo("สำเร็จ",
                                f"บันทึก '{self.name}' สำเร็จ ({ok} รูป)",
                                parent=self.win)
            self._close(refresh=True)

    def _close(self, refresh=False):
        self.running = False
        self.cap.release()
        self.win.destroy()
        self.on_done(refresh)


# ── Main DB Manager ───────────────────────────────────────────────────────────
class DBManager:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Face DB Manager")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)

        self._thumb_cache = {}   # person_id → ImageTk
        self._build_ui()
        self._refresh()

    def _build_ui(self):
        # ── Header ──────────────────────────────────────────────────────
        hdr = tk.Frame(self.root, bg=SURFACE, pady=14)
        hdr.pack(fill="x")

        tk.Label(hdr, text="⬡  Face Database Manager",
                 font=FONT_TITLE, fg=TEXT, bg=SURFACE).pack(side="left", padx=20)

        tk.Button(hdr, text="＋  เพิ่มคนใหม่", font=FONT_BODY,
                  bg=ACCENT, fg=TEXT, activebackground=ACCENT_H,
                  relief="flat", padx=14, pady=6,
                  command=self._add_person).pack(side="right", padx=16)

        # ── Search bar ──────────────────────────────────────────────────
        search_bar = tk.Frame(self.root, bg=BG, pady=8)
        search_bar.pack(fill="x", padx=16)

        tk.Label(search_bar, text="ค้นหา", font=FONT_BODY,
                 fg=MUTED, bg=BG).pack(side="left")

        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", lambda *_: self._refresh())
        search_entry = tk.Entry(search_bar, textvariable=self.search_var,
                                font=FONT_BODY, bg=SURFACE2, fg=TEXT,
                                insertbackground=TEXT, relief="flat",
                                highlightthickness=1, highlightcolor=BLUE,
                                highlightbackground=BORDER)
        search_entry.pack(side="left", fill="x", expand=True,
                          padx=8, ipady=6)

        # ── Stats bar ───────────────────────────────────────────────────
        self.stats_lbl = tk.Label(self.root, text="",
                                  font=FONT_SMALL, fg=MUTED, bg=BG, anchor="w")
        self.stats_lbl.pack(fill="x", padx=20)

        tk.Frame(self.root, bg=BORDER, height=1).pack(fill="x", padx=0, pady=4)

        # ── Scrollable list ─────────────────────────────────────────────
        container = tk.Frame(self.root, bg=BG)
        container.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(container, bg=BG, highlightthickness=0,
                                width=560, height=460)
        sb = ttk.Scrollbar(container, orient="vertical",
                           command=self.canvas.yview)
        self.list_frame = tk.Frame(self.canvas, bg=BG)

        self.list_frame.bind("<Configure>",
                             lambda e: self.canvas.configure(
                                 scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.list_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=sb.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # scroll ด้วย mouse wheel
        self.canvas.bind_all("<MouseWheel>",
                             lambda e: self.canvas.yview_scroll(
                                 -1 if e.delta > 0 else 1, "units"))

        # ── Footer ──────────────────────────────────────────────────────
        ft = tk.Frame(self.root, bg=SURFACE, pady=8)
        ft.pack(fill="x")
        tk.Label(ft, text="คลิก  ✖ DELETE  เพื่อลบรายบุคคล",
                 font=FONT_SMALL, fg=MUTED, bg=SURFACE).pack()

    # ── Refresh list ──────────────────────────────────────────────────────────
    def _refresh(self):
        q = self.search_var.get().strip().lower()
        rows = db_list_persons()
        if q:
            rows = [r for r in rows if q in r[1].lower()]

        # ลบ widget เก่า
        for w in self.list_frame.winfo_children():
            w.destroy()

        total_vec = sum(r[2] for r in rows)
        self.stats_lbl.configure(
            text=f"{len(rows)} คน  ·  {total_vec} vectors รวม")

        if not rows:
            tk.Label(self.list_frame,
                     text="ไม่พบข้อมูล" if q else "ยังไม่มีข้อมูลในฐานข้อมูล",
                     font=FONT_BODY, fg=MUTED, bg=BG).pack(pady=40)
            return

        for pid, name, cnt in rows:
            self._make_row(pid, name, cnt)

    def _make_row(self, pid, name, cnt):
        # ── Row container ────────────────────────────────────────────────
        row = tk.Frame(self.list_frame, bg=SURFACE2,
                       highlightthickness=1, highlightbackground=BORDER)
        row.pack(fill="x", padx=12, pady=4)

        # Avatar placeholder (วงกลมสี)
        colors = ["#58a6ff","#3fb950","#d2a8ff","#ffa657","#ff7b72","#79c0ff"]
        color  = colors[pid % len(colors)]
        avatar = tk.Canvas(row, width=THUMB, height=THUMB,
                           bg=SURFACE2, highlightthickness=0)
        avatar.pack(side="left", padx=12, pady=10)
        avatar.create_oval(4, 4, THUMB-4, THUMB-4, fill=color, outline="")
        initials = name[:2].upper()
        avatar.create_text(THUMB//2, THUMB//2, text=initials,
                           font=("Consolas", 18, "bold"), fill="white")

        # ── Info ─────────────────────────────────────────────────────────
        info = tk.Frame(row, bg=SURFACE2)
        info.pack(side="left", fill="both", expand=True, pady=10)

        tk.Label(info, text=name, font=("Consolas", 12, "bold"),
                 fg=TEXT, bg=SURFACE2, anchor="w").pack(fill="x")

        badge_row = tk.Frame(info, bg=SURFACE2)
        badge_row.pack(fill="x", pady=2)

        tk.Label(badge_row, text=f"ID: {pid}",
                 font=FONT_SMALL, fg=MUTED, bg=SURFACE2).pack(side="left")
        tk.Label(badge_row, text=f"  ·  {cnt} vector{'s' if cnt!=1 else ''}",
                 font=FONT_SMALL, fg=MUTED, bg=SURFACE2).pack(side="left")

        # ── Buttons ──────────────────────────────────────────────────────
        btn_frame = tk.Frame(row, bg=SURFACE2)
        btn_frame.pack(side="right", padx=12)

        tk.Button(btn_frame,
                  text="เพิ่มรูป",
                  font=FONT_SMALL,
                  bg=SURFACE, fg=BLUE,
                  activebackground=SURFACE2, activeforeground=BLUE,
                  relief="flat", padx=10, pady=5,
                  command=lambda p=pid, n=name: self._add_photo(p, n)
                  ).pack(pady=2)

        tk.Button(btn_frame,
                  text="Delete",
                  font=FONT_SMALL,
                  bg=SURFACE, fg=RED,
                  activebackground=RED, activeforeground=TEXT,
                  relief="flat", padx=10, pady=5,
                  command=lambda p=pid, n=name: self._delete(p, n)
                  ).pack(pady=2)

    # ── Actions ───────────────────────────────────────────────────────────────
    def _add_person(self):
        name = simpledialog.askstring(
            "เพิ่มคนใหม่", "ชื่อ-นามสกุล:",
            parent=self.root)
        if not name or not name.strip():
            return
        CaptureWindow(self.root, name.strip(),
                      on_done=lambda refresh: self._on_capture_done(refresh))

    def _add_photo(self, person_id, name):
        """เพิ่มรูปให้คนที่มีอยู่แล้ว"""
        def on_done(refresh):
            if refresh:
                self._refresh()

        # สร้าง capture window แต่บันทึกใส่ person_id เดิม
        win = CaptureWindow.__new__(CaptureWindow)
        win.name    = name
        win.on_done = on_done
        win.cap     = open_camera()
        win.running = True
        win.samples = []
        win.face_box = None
        win.face_info = None
        win.latest_frame = None

        win.win = tk.Toplevel(self.root)
        win.win.title(f"เพิ่มรูป: {name}")
        win.win.configure(bg=BG)
        win.win.resizable(False, False)
        win.win.grab_set()
        win.win.protocol("WM_DELETE_WINDOW", win._close)
        win._build_ui()

        # override _save ให้ใส่ person_id เดิม
        def _save_existing():
            win.save_btn.configure(state="disabled", text="กำลังบันทึก...")
            win.win.update()
            def _do():
                ok = 0
                for sample in win.samples:
                    emb = get_embedding(
                        sample["image"],
                        aligned=sample.get("aligned", False),
                    )
                    if emb is not None:
                        add_vector(person_id, emb)
                        ok += 1
                win.win.after(0, lambda: win._finish(ok))
            threading.Thread(target=_do, daemon=True).start()

        win._save = _save_existing

        win._cam_thread = threading.Thread(target=win._cam_loop, daemon=True)
        win._cam_thread.start()
        win.win.bind("<space>", lambda e: win._capture())
        win._tick()

    def _delete(self, person_id, name):
        ok = messagebox.askyesno(
            "ยืนยันการลบ",
            f"ลบ  '{name}'  ออกจากฐานข้อมูล?\n\nจะลบ vector ทั้งหมดของคนนี้ด้วย",
            icon="warning", parent=self.root)
        if ok:
            db_delete_person(person_id)
            self._refresh()

    def _on_capture_done(self, refresh):
        if refresh:
            self._refresh()

    def run(self):
        self.root.mainloop()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    app = DBManager()
    app.run()
