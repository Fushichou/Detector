"""
gui.py - Tkinter UI สำหรับแสดงกล้องและผลจำใบหน้า.

แนวคิด optimize:
- ภาพกล้องอัปเดตตามรอบที่กำหนดใน main.py
- thumbnail ใบหน้าอัปเดตเฉพาะเมื่อข้อมูลเปลี่ยนจริง
- mask วงกลมสร้างครั้งเดียวแล้ว reuse เพื่อลดงาน PIL
"""

import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageDraw, ImageTk

PANEL_W = 200
FACE_THUMB = 96
FOUND_COL = "#22c55e"
UNKNOWN_COL = "#ef4444"
WAIT_COL = "#94a3b8"

class FaceRecognitionGUI:
    def __init__(self, cam_w=640, cam_h=480, interval_ms=33):
        self.root = tk.Tk()
        self.root.title("Face Recognition System")
        self.root.configure(bg="#0f172a")
        self.root.resizable(False, False)

        self.interval_ms = interval_ms
        self._face_cards = {}
        self._thumb_mask = self._make_thumb_mask()

        self._build_ui(cam_w, cam_h)

    def _make_thumb_mask(self):
        """สร้าง alpha mask วงกลมสำหรับ thumbnail ครั้งเดียว."""
        mask = Image.new("L", (FACE_THUMB, FACE_THUMB), 0)
        ImageDraw.Draw(mask).ellipse((0, 0, FACE_THUMB, FACE_THUMB), fill=255)
        return mask

    def _build_ui(self, cam_w, cam_h):
        """สร้าง layout หลัก: header, กล้อง, panel รายชื่อ, footer."""
        header = tk.Frame(self.root, bg="#1e293b", height=48)
        header.pack(fill="x")

        tk.Label(header, text="Face", font=("Helvetica", 14, "bold"),
                 fg="white", bg="#1e293b").pack(side="left", padx=16, pady=10)

        self.lbl_fps = tk.Label(header, text="FPS: --", font=("Helvetica", 11),
                                fg="#94a3b8", bg="#1e293b")
        self.lbl_fps.pack(side="right", padx=16)

        self.lbl_persons = tk.Label(header, text="Persons: 0", font=("Helvetica", 11),
                                    fg="#94a3b8", bg="#1e293b")
        self.lbl_persons.pack(side="right", padx=8)

        body = tk.Frame(self.root, bg="#0f172a")
        body.pack(fill="both")

        self.cam_label = tk.Label(body, bg="#000000", width=cam_w, height=cam_h)
        self.cam_label.pack(side="left")

        right = tk.Frame(body, bg="#1e293b", width=PANEL_W)
        right.pack(side="left", fill="y")
        right.pack_propagate(False)

        tk.Label(right, text="Detected Faces", font=("Helvetica", 11, "bold"),
                 fg="white", bg="#1e293b").pack(pady=(12, 6))

        tk.Frame(right, bg="#334155", height=1).pack(fill="x", padx=8)

        canvas = tk.Canvas(right, bg="#1e293b", width=PANEL_W - 4, highlightthickness=0)
        scrollbar = ttk.Scrollbar(right, orient="vertical", command=canvas.yview)
        self.face_frame = tk.Frame(canvas, bg="#1e293b")

        self.face_frame.bind(
            "<Configure>",
            lambda _e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=self.face_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        footer = tk.Frame(self.root, bg="#1e293b", height=32)
        footer.pack(fill="x")

        self.lbl_status = tk.Label(
            footer,
            text="กด Q เพื่อออก  |  กด R เพื่อ Reload DB",
            font=("Helvetica", 9),
            fg="#64748b",
            bg="#1e293b",
        )
        self.lbl_status.pack(pady=6)

    def update_camera(self, frame_bgr):
        """
        แปลง BGR กล้อง → RGB Tk image ที่เหมาะ UI
        - หลีกเลี่ยง JPEG ซ้ำ โดยใช้ format ที่ PIL เร็ว
        """
        # RGB conversion: บังคับ uint8 เพื่อหลีกเลี่ยง slow path
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # ใช้ tobytes() แม้จะเร็วกว่า fromarray ในบางกรณี
        pil_img = Image.fromarray(img_rgb, mode="RGB")
        imgtk = ImageTk.PhotoImage(image=pil_img)
        self.cam_label.imgtk = imgtk  # เก็บ reference ไม่เช่นนั้น GC ลบ
        self.cam_label.configure(image=imgtk)

    def update_faces(self, face_list):
        """
        จัดการ card ใบหน้า: เพิ่ม/ลบ/อัปเดต
        - อัปเดตเฉพาะ card ที่ข้อมูลเปลี่ยน (ลดอาการกระตุก)
        - ลบ card ของคนที่หายไปแล้ว
        """
        current_ids = {face["id"] for face in face_list}
        for fid in list(self._face_cards.keys()):
            if fid not in current_ids:
                self._face_cards[fid].destroy()
                del self._face_cards[fid]

        for face in face_list:
            fid = face["id"]
            if fid not in self._face_cards:
                self._face_cards[fid] = self._make_card(fid)

            card = self._face_cards[fid]
            self._update_card(card, face)
            if not card.winfo_ismapped():
                card.pack(fill="x", padx=8, pady=4)

    def _make_card(self, fid):
        """
        สร้าง card widget สำหรับบุคคลที่ track
        - card.last_key ใช้เพื่อตรวจหลีกเลี่ยง re-render ที่ไม่จำเป็น
        """
        card = tk.Frame(self.face_frame, bg="#0f172a", relief="flat", bd=0)
        card.fid = fid
        card.last_key = None
        card.img_label = tk.Label(card, bg="#0f172a")
        card.img_label.pack(pady=(8, 4))

        card.name_label = tk.Label(card, text="...", font=("Helvetica", 10, "bold"),
                                   fg="white", bg="#0f172a")
        card.name_label.pack()

        card.sim_label = tk.Label(card, text="", font=("Helvetica", 9),
                                  fg="#94a3b8", bg="#0f172a")
        card.sim_label.pack()

        card.status_label = tk.Label(card, text="", font=("Helvetica", 9, "bold"),
                                     bg="#0f172a")
        card.status_label.pack(pady=(2, 8))

        tk.Frame(card, bg="#1e293b", height=1).pack(fill="x")
        return card

    def _update_card(self, card, face):
        """
        อัปเดต card เมื่อข้อมูล identity/sim/รูป เปลี่ยนจากรอบก่อน
        - ตรวจสอบ key เพื่อหลีกเลี่ยง re-render ถ้าข้อมูลเหมือน
        - resize + mask + convert ครั้งเดียว เมื่อมีการเปลี่ยนแปลง
        """
        key = (
            face.get("identity"),
            round(face.get("sim", 0.0), 2),
            face.get("found"),
            id(face.get("face_img")),  # object id เพื่อตรวจ frame ใหม่
        )
        if key == card.last_key:
            return  # ข้อมูลเหมือน ข้ามการ render ทั้งหมด
        card.last_key = key

        # อัปเดตรูปเมื่อมี face_img
        face_img = face.get("face_img")
        if face_img is not None:
            # RGB + resize + circular mask เพื่อให้ดูดี
            img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb).resize((FACE_THUMB, FACE_THUMB), Image.LANCZOS)
            img_pil.putalpha(self._thumb_mask)  # reuse mask เพื่อประหยัด
            imgtk = ImageTk.PhotoImage(image=img_pil)
            card.img_label.imgtk = imgtk
            card.img_label.configure(image=imgtk)

        identity = face.get("identity") or "..."
        sim = face.get("sim", 0.0)
        found = face.get("found", False)

        card.name_label.configure(text=identity)
        card.sim_label.configure(text=f"ความคล้าย {sim:.0%}" if sim > 0 else "")

        if identity == "...":
            card.status_label.configure(text="กำลังวิเคราะห์", fg=WAIT_COL)
        elif found:
            card.status_label.configure(text="พบในฐานข้อมูล", fg=FOUND_COL)
        else:
            card.status_label.configure(text="ไม่พบในฐานข้อมูล", fg=UNKNOWN_COL)

    def update_stats(self, fps, n_persons):
        self.lbl_fps.configure(text=f"FPS: {fps:.1f}")
        self.lbl_persons.configure(text=f"Persons: {n_persons}")

    def set_status(self, text):
        self.lbl_status.configure(text=text)

    def tick(self, callback_fn):
        callback_fn()
        self.root.after(self.interval_ms, self.tick, callback_fn)

    def run(self, callback_fn):
        self.root.after(self.interval_ms, self.tick, callback_fn)
        self.root.mainloop()

    def destroy(self):
        try:
            self.root.destroy()
        except Exception:
            pass
