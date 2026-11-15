import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from models.opencv_baseline import colorize
from src import postprocessing

MAX_SIDE = 1024      # büyük görüntüleri otomatik küçültme sınırı
PREVIEW = (460, 460)  # önizleme kutusu boyutu


def read_image(path):
    """Türkçe karakterli yollarda da çalışır. Okunamazsa None döndürür."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def resize_max(img, max_side=MAX_SIDE):
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return img, False
    new = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return new, True


def to_gray3(img):
    return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def to_tk(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # Tkinter RGB ister
    pil = Image.fromarray(rgb)
    pil.thumbnail(PREVIEW)
    return ImageTk.PhotoImage(pil)


class App:
    def __init__(self, root):
        self.root = root
        root.title("Görüntü Renklendirme")
        self.gray = None
        self.result = None

        top = tk.Frame(root)
        top.pack(pady=8)
        tk.Button(top, text="Görüntü Seç", command=self.open_image).pack(side="left", padx=4)
        self.use_post = tk.BooleanVar(value=True)
        tk.Checkbutton(top, text="Post-processing", variable=self.use_post).pack(side="left", padx=8)
        self.btn_color = tk.Button(top, text="Renklendir", command=self.run, state="disabled")
        self.btn_color.pack(side="left", padx=4)
        self.btn_save = tk.Button(top, text="Kaydet", command=self.save, state="disabled")
        self.btn_save.pack(side="left", padx=4)

        mid = tk.Frame(root)
        mid.pack(padx=10)
        self.left = tk.Label(mid, text="Gri", width=60, height=25, relief="sunken")
        self.left.pack(side="left", padx=5)
        self.right = tk.Label(mid, text="Renkli", width=60, height=25, relief="sunken")
        self.right.pack(side="left", padx=5)

        self.status = tk.Label(root, text="Bir görüntü seçin.", anchor="w")
        self.status.pack(fill="x", padx=10, pady=6)

    def open_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Görüntüler", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if not path:
            return
        img = read_image(path)
        if img is None:
            messagebox.showerror("Hata", "Bu dosya görüntü olarak okunamadı.\nJPG veya PNG deneyin.")
            return
        img, resized = resize_max(img)
        self.gray = to_gray3(img)
        self.result = None
        self.tk_left = to_tk(self.gray)
        self.left.config(image=self.tk_left, text="", width=0, height=0)
        self.right.config(image="", text="Renkli", width=60, height=25)
        self.btn_color.config(state="normal")
        self.btn_save.config(state="disabled")
        note = " (büyük görüntü küçültüldü)" if resized else ""
        self.status.config(text=f"Yüklendi: {path}{note}")

    def run(self):
        self.status.config(text="İşleniyor...")
        self.root.update_idletasks()
        try:
            out = colorize(self.gray)
            if self.use_post.get():
                out = postprocessing.refine(out, saturation=1.2)
        except Exception as e:
            messagebox.showerror("Hata", f"Renklendirme başarısız:\n{e}")
            self.status.config(text="Hata oluştu.")
            return
        self.result = out
        self.tk_right = to_tk(out)
        self.right.config(image=self.tk_right, text="", width=0, height=0)
        self.btn_save.config(state="normal")
        self.status.config(text="Tamamlandı.")

    def save(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".jpg", filetypes=[("JPG", "*.jpg"), ("PNG", "*.png")]
        )
        if not path:
            return
        ext = ".png" if path.lower().endswith(".png") else ".jpg"
        ok, buf = cv2.imencode(ext, self.result)
        if ok:
            buf.tofile(path)
            self.status.config(text=f"Kaydedildi: {path}")


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
    