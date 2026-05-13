"""
gui.py — GUI แยกไฟล์ด้วย Tkinter
แสดง:
  - ซ้าย  : ภาพจากกล้องพร้อม overlay กรอบ
  - ขวา   : รายการใบหน้าที่ crop + ผลการเทียบ DB (เจอ/ไม่เจอ)
  - ล่าง  : สถานะระบบ FPS / จำนวนคน
"""

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np


PANEL_W    = 200   # ความกว้าง panel ขวา (รูปใบหน้า)
FACE_THUMB = 96    # ขนาด thumbnail ใบหน้าใน panel
FOUND_COL  = "#22c55e"   # เขียว = เจอใน DB
UNKNOWN_COL = "#ef4444"  # แดง = ไม่เจอ
WAIT_COL   = "#94a3b8"   # เทา = รอ


class FaceRecognitionGUI:
    def __init__(self, cam_w=640, cam_h=480):
        self.root = tk.Tk()
        self.root.title("Face Recognition System")
        self.root.configure(bg="#0f172a")
        self.root.resizable(False, False)

        self._build_ui(cam_w, cam_h)
        self._face_cards = {}   # track_id → card widgets

    # ─── สร้าง UI ─────────────────────────────────────────────────────────────
    def _build_ui(self, cam_w, cam_h):
        # ── Header ────────────────────────────────────────────────────────────
        header = tk.Frame(self.root, bg="#1e293b", height=48)
        header.pack(fill="x")

        tk.Label(header, text="Face",
                 font=("Helvetica", 14, "bold"),
                 fg="white", bg="#1e293b").pack(side="left", padx=16, pady=10)

        self.lbl_fps = tk.Label(header, text="FPS: --",
                                font=("Helvetica", 11),
                                fg="#94a3b8", bg="#1e293b")
        self.lbl_fps.pack(side="right", padx=16)

        self.lbl_persons = tk.Label(header, text="Persons: 0",
                                    font=("Helvetica", 11),
                                    fg="#94a3b8", bg="#1e293b")
        self.lbl_persons.pack(side="right", padx=8)

        # ── Body ──────────────────────────────────────────────────────────────
        body = tk.Frame(self.root, bg="#0f172a")
        body.pack(fill="both")

        # กล้อง (ซ้าย)
        self.cam_label = tk.Label(body, bg="#000000",
                                  width=cam_w, height=cam_h)
        self.cam_label.pack(side="left")

        # Panel ขวา (รายการใบหน้า)
        right = tk.Frame(body, bg="#1e293b", width=PANEL_W)
        right.pack(side="left", fill="y")
        right.pack_propagate(False)

        tk.Label(right, text="Detected Faces",
                 font=("Helvetica", 11, "bold"),
                 fg="white", bg="#1e293b").pack(pady=(12, 6))

        sep = tk.Frame(right, bg="#334155", height=1)
        sep.pack(fill="x", padx=8)

        # scroll area
        canvas = tk.Canvas(right, bg="#1e293b",
                           width=PANEL_W - 4, highlightthickness=0)
        scrollbar = ttk.Scrollbar(right, orient="vertical",
                                  command=canvas.yview)
        self.face_frame = tk.Frame(canvas, bg="#1e293b")

        self.face_frame.bind("<Configure>", lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")))

        canvas.create_window((0, 0), window=self.face_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # ── Footer ────────────────────────────────────────────────────────────
        footer = tk.Frame(self.root, bg="#1e293b", height=32)
        footer.pack(fill="x")

        self.lbl_status = tk.Label(
            footer,
            text="กด Q เพื่อออก  |  กด R เพื่อ Reload DB",
            font=("Helvetica", 9),
            fg="#64748b", bg="#1e293b"
        )
        self.lbl_status.pack(pady=6)

    # ─── อัปเดตภาพกล้อง ──────────────────────────────────────────────────────
    def update_camera(self, frame_bgr):
        """รับ BGR numpy array → แสดงบน label"""
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        self.cam_label.imgtk = imgtk
        self.cam_label.configure(image=imgtk)

    # ─── อัปเดต panel ใบหน้า ─────────────────────────────────────────────────
    def update_faces(self, face_list):
        """
        face_list: list of dict {
            "id"       : unique int (track index)
            "face_img" : numpy BGR 112x112 หรือ None
            "identity" : str
            "sim"      : float
            "found"    : bool
        }
        """
        # ลบ card เก่าที่ไม่มีแล้ว
        current_ids = {f["id"] for f in face_list}
        for fid in list(self._face_cards.keys()):
            if fid not in current_ids:
                self._face_cards[fid].destroy()
                del self._face_cards[fid]

        # สร้าง / อัปเดต card
        for i, face in enumerate(face_list):
            fid = face["id"]
            if fid not in self._face_cards:
                card = self._make_card(fid)
                self._face_cards[fid] = card

            self._update_card(self._face_cards[fid], face)
            self._face_cards[fid].pack(fill="x", padx=8, pady=4)

    def _make_card(self, fid):
        card = tk.Frame(self.face_frame, bg="#0f172a",
                        relief="flat", bd=0)
        card.fid        = fid
        card.img_label  = tk.Label(card, bg="#0f172a")
        card.img_label.pack(pady=(8, 4))

        card.name_label = tk.Label(card, text="...",
                                   font=("Helvetica", 10, "bold"),
                                   fg="white", bg="#0f172a")
        card.name_label.pack()

        card.sim_label = tk.Label(card, text="",
                                  font=("Helvetica", 9),
                                  fg="#94a3b8", bg="#0f172a")
        card.sim_label.pack()

        card.status_label = tk.Label(card, text="",
                                     font=("Helvetica", 9, "bold"),
                                     bg="#0f172a")
        card.status_label.pack(pady=(2, 8))

        sep = tk.Frame(card, bg="#1e293b", height=1)
        sep.pack(fill="x")
        return card

    def _update_card(self, card, face):
        # thumbnail
        if face["face_img"] is not None:
            img = cv2.cvtColor(face["face_img"], cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img).resize(
                (FACE_THUMB, FACE_THUMB), Image.LANCZOS)

            # ขอบกลม (วงกลม mask)
            mask = Image.new("L", (FACE_THUMB, FACE_THUMB), 0)
            from PIL import ImageDraw
            ImageDraw.Draw(mask).ellipse(
                (0, 0, FACE_THUMB, FACE_THUMB), fill=255)
            img.putalpha(mask)

            imgtk = ImageTk.PhotoImage(image=img)
            card.img_label.imgtk = imgtk
            card.img_label.configure(image=imgtk)

        # ชื่อ + similarity
        identity = face["identity"]
        sim      = face["sim"]
        found    = face["found"]

        card.name_label.configure(text=identity if identity else "...")

        if sim > 0:
            card.sim_label.configure(text=f"ความคล้าย {sim:.0%}")
        else:
            card.sim_label.configure(text="")

        if identity in ("...", None):
            card.status_label.configure(text="⏳ กำลังวิเคราะห์",
                                        fg=WAIT_COL)
        elif found:
            card.status_label.configure(text="✅ พบในฐานข้อมูล",
                                        fg=FOUND_COL)
        else:
            card.status_label.configure(text="❌ ไม่พบในฐานข้อมูล",
                                        fg=UNKNOWN_COL)

    # ─── อัปเดต header ────────────────────────────────────────────────────────
    def update_stats(self, fps, n_persons):
        self.lbl_fps.configure(text=f"FPS: {fps:.1f}")
        self.lbl_persons.configure(text=f"Persons: {n_persons}")

    def set_status(self, text):
        self.lbl_status.configure(text=text)

    # ─── loop ─────────────────────────────────────────────────────────────────
    def tick(self, callback_fn):
        """เรียก callback_fn() ทุก 15ms (≈66fps cap)"""
        callback_fn()
        self.root.after(15, self.tick, callback_fn)

    def run(self, callback_fn):
        self.root.after(15, self.tick, callback_fn)
        self.root.mainloop()

    def destroy(self):
        try:
            self.root.destroy()
        except Exception:
            pass
