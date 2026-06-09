"""
gui.py — Tkinter UI สำหรับแสดงกล้องและผลจำใบหน้า

แนวคิด:
- ภาพกล้องอัปเดตตามรอบที่กำหนดใน main.py
- thumbnail ใบหน้าอัปเดตเฉพาะเมื่อข้อมูลเปลี่ยนจริง
- mask วงกลมสร้างครั้งเดียวแล้ว reuse
- cam_label ปรับขนาดอัตโนมัติตาม resolution จริงของ source
"""

import tkinter as tk 
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk

PANEL_W     = 200
FACE_THUMB  = 96
FOUND_COL   = "#22c55e"
UNKNOWN_COL = "#ef4444"
WAIT_COL    = "#94a3b8"
MAX_CAM_W = 1280
MAX_CAM_H = 800

class FaceRecognitionGUI:
    def __init__(self, cam_w=640, cam_h=480, interval_ms=33):
        self.root = tk.Tk()
        self.root.title("Face Recognition System")
        self.root.configure(bg="#0f172a")
        self.root.resizable(True, True)
        self.root.geometry("1280x720")  # Default window size

        self.interval_ms = interval_ms
        self._face_cards = {}
        self._thumb_mask = self._make_thumb_mask()
        self.camera_var = tk.StringVar(value="0")
        self._camera_callback = None

        # Store original camera resolution
        self.cam_w_orig = cam_w
        self.cam_h_orig = cam_h

        # Initialize display size
        self.disp_w = cam_w
        self.disp_h = cam_h
        self._display_scale = 1.0

        # Pre-allocate letterbox background
        self._cam_buf = np.zeros((self.disp_h, self.disp_w, 3), dtype=np.uint8)
        self._cam_params = None

        print(f"[GUI] Camera resolution: {cam_w}x{cam_h}")
        self._build_ui()
        
        # Bind window resize event to update display size
        self.root.bind("<Configure>", self._on_window_configure)

    def _make_thumb_mask(self):
        mask = Image.new("L", (FACE_THUMB, FACE_THUMB), 0)
        ImageDraw.Draw(mask).ellipse((0, 0, FACE_THUMB, FACE_THUMB), fill=255)
        return mask

    def _build_ui(self):
        header = tk.Frame(self.root, bg="#1e293b", height=48)
        header.pack(fill="x")

        tk.Label(header, text="Face", font=("Helvetica", 14, "bold"),
                 fg="white", bg="#1e293b").pack(side="left", padx=16, pady=10)

        self.lbl_fps = tk.Label(header, text="FPS: --",
                                font=("Helvetica", 11), fg="#94a3b8", bg="#1e293b")
        self.lbl_fps.pack(side="right", padx=16)

        # Camera source switcher
        tk.Label(header, text="Camera:", font=("Helvetica", 10),
                 fg="#94a3b8", bg="#1e293b").pack(side="right", padx=4)
        
        camera_combo = ttk.Combobox(header, textvariable=self.camera_var, 
                                    values=["0", "1"], state="readonly", width=3,
                                    font=("Helvetica", 10))
        camera_combo.pack(side="right", padx=4, pady=8)
        camera_combo.bind("<<ComboboxSelected>>", self._on_camera_change)

        self._db_btn = tk.Button(
            header,
            text="Edit info",
            font=("Helvetica", 10, "bold"),
            bg="#1e3a5f",
            fg="#58a6ff",
            activebackground="#1d4ed8",
            activeforeground="white",
            relief="flat",
            padx=12,
            pady=4,
            cursor="hand2",
            command=self._on_db_click,
        )
        self._db_btn.pack(side="right", padx=8, pady=8)
        self._db_callback = None

        self.lbl_persons = tk.Label(header, text="Persons: 0",
                                    font=("Helvetica", 11), fg="#94a3b8", bg="#1e293b")
        self.lbl_persons.pack(side="right", padx=8)

        body = tk.Frame(self.root, bg="#0f172a")
        body.pack(fill="both", expand=True)

        # Left side: camera display (expandable)
        left = tk.Frame(body, bg="#000000")
        left.pack(side="left", fill="both", expand=True)
        left.pack_propagate(False)
        
        self.cam_label = tk.Label(left, bg="#000000")
        self.cam_label.pack(fill="both", expand=True)

        right = tk.Frame(body, bg="#1e293b", width=PANEL_W)
        right.pack(side="right", fill="y")
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

    def set_db_callback(self, fn):
        """ผูก callback ที่จะเรียกเมื่อกดปุ่ม จัดการ DB"""
        self._db_callback = fn

    def _on_db_click(self):
        if self._db_callback:
            self._db_callback()

    def set_camera_callback(self, fn):
        """ผูก callback ที่จะเรียกเมื่อเปลี่ยน camera source"""
        self._camera_callback = fn

    def _on_camera_change(self, event):
        if self._camera_callback:
            source = int(self.camera_var.get())
            self._camera_callback(source)

    def _on_window_configure(self, event):
        """Handle window resize events to update camera display scale"""
        # Get the left frame size (camera display area)
        if hasattr(self, 'cam_label') and self.cam_label.winfo_exists():
            available_w = self.cam_label.winfo_width()
            available_h = self.cam_label.winfo_height()
            
            if available_w > 1 and available_h > 1:
                # Calculate scale to fit camera in available space
                scale = min(available_w / self.cam_w_orig, available_h / self.cam_h_orig, 1.0)
                new_w = int(self.cam_w_orig * scale)
                new_h = int(self.cam_h_orig * scale)
                
                # Only update if size changed
                if new_w != self.disp_w or new_h != self.disp_h:
                    old_size = (self.disp_w, self.disp_h)
                    self.disp_w = new_w
                    self.disp_h = new_h
                    self._display_scale = scale
                    
                    # Recreate buffer with new size
                    self._cam_buf = np.zeros((self.disp_h, self.disp_w, 3), dtype=np.uint8)
                    self._cam_params = None  # Reset cache to recalculate
                    
                    print(f"[GUI] Resized display: {old_size} → {new_w}x{new_h} (scale: {scale:.2f})")


    def update_camera(self, frame_bgr):
        """
        แปลง BGR → RGB แล้ว letterbox ให้พอดี disp_w × disp_h
        - ปรับขนาดตามหน้าต่างและกล้องโดยอัตโนมัติ
        - เขียนลง numpy buffer โดยตรง
        """
        h, w = frame_bgr.shape[:2]

        # Ensure buffer exists and is correct size
        if self._cam_buf.shape[:2] != (self.disp_h, self.disp_w):
            self._cam_buf = np.zeros((self.disp_h, self.disp_w, 3), dtype=np.uint8)

        # Cache resize params — frame size ไม่เปลี่ยนระหว่าง session
        if self._cam_params is None or self._cam_params[0] != (h, w):
            scale = min(self.disp_w / w, self.disp_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            pad_x = (self.disp_w - new_w) // 2
            pad_y = (self.disp_h - new_h) // 2
            self._cam_params = ((h, w), scale, new_w, new_h, pad_x, pad_y)

        _, scale, new_w, new_h, pad_x, pad_y = self._cam_params

        if new_w != w or new_h != h:
            frame_bgr = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Clear buffer and write frame
        self._cam_buf.fill(0)
        self._cam_buf[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = img_rgb

        imgtk = ImageTk.PhotoImage(image=Image.fromarray(self._cam_buf))
        self.cam_label.imgtk = imgtk
        self.cam_label.configure(image=imgtk)

    def update_faces(self, face_list):
        """เพิ่ม / ลบ / อัปเดต card ใบหน้า"""
        current_ids = {face["id"] for face in face_list}
        for fid in list(self._face_cards):
            if fid not in current_ids:
                self._face_cards.pop(fid).destroy()

        for face in face_list:
            fid = face["id"]
            if fid not in self._face_cards:
                self._face_cards[fid] = self._make_card(fid)

            card = self._face_cards[fid]
            self._update_card(card, face)
            if not card.winfo_ismapped():
                card.pack(fill="x", padx=8, pady=4)

    def _make_card(self, fid):
        card = tk.Frame(self.face_frame, bg="#0f172a", relief="flat", bd=0)
        card.fid      = fid
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
        key = (
            face.get("identity"),
            round(face.get("sim", 0.0), 2),
            face.get("found"),
            id(face.get("face_img")),
        )
        if key == card.last_key:
            return
        card.last_key = key

        face_img = face.get("face_img")
        if face_img is not None:
            img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb).resize((FACE_THUMB, FACE_THUMB), Image.LANCZOS)
            img_pil.putalpha(self._thumb_mask)
            imgtk = ImageTk.PhotoImage(image=img_pil)
            card.img_label.imgtk = imgtk
            card.img_label.configure(image=imgtk)
        else:
            card.img_label.imgtk = None
            card.img_label.configure(image="")

        identity = face.get("identity") or "..."
        sim      = face.get("sim", 0.0)
        found    = face.get("found", False)

        card.name_label.configure(text=identity)
        card.sim_label.configure(text=f"ความคล้าย {sim:.0%}" if sim > 0 else "")

        if identity == "...":
            card.status_label.configure(text="กำลังวิเคราะห์", fg=WAIT_COL)
        elif found:
            card.status_label.configure(text="พบในฐานข้อมูล", fg=FOUND_COL)
        else:
            card.status_label.configure(text="ไม่พบในฐานข้อมูล", fg=UNKNOWN_COL)

    def update_stats(self, cam_fps, gui_fps, n_persons):
        self.lbl_fps.configure(text=f"Cam: {cam_fps:.2f} FPS")
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