import os
import cv2
import numpy as np

from src import preprocessing, postprocessing

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINETUNED_WEIGHTS = os.path.join(ROOT, "models", "eccv16_finetuned.pt")

MODEL_LABELS = {
    "Fine-tuned (PyTorch)": "finetuned",
    "OpenCV baseline": "opencv",
}


def finetuned_available():
    return os.path.exists(FINETUNED_WEIGHTS)


def get_colorize(model):
    # Import'lar burada: opencv'yi seçen torch yüklemek zorunda kalmasın.
    if model == "finetuned":
        from models.colorization_net import colorize
    elif model == "opencv":
        from models.opencv_baseline import colorize
    else:
        raise ValueError(f"Bilinmeyen model: {model}")
    return colorize


def read_image(path):
    """Türkçe karakterli yollarda da çalışır. Okunamazsa None döndürür."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def keep_original_L(orig_bgr, out_bgr):
    """Çıktının renklerini (ab) alır, parlaklığı (L) orijinalden geri koyar."""
    lab_o = cv2.cvtColor(orig_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    lab_c = cv2.cvtColor(out_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    lab_c[:, :, 0] = lab_o[:, :, 0]
    res = cv2.cvtColor(lab_c, cv2.COLOR_Lab2BGR)
    return (np.clip(res, 0, 1) * 255).astype(np.uint8)


def process(img_bgr, model="finetuned", use_pre=False, use_post=True, saturation=1.2):
    """BGR uint8 alır, renklendirilmiş BGR uint8 döndürür (arayüz de bunu kullanır)."""
    colorize = get_colorize(model)
    inp = preprocessing.apply_clahe(img_bgr) if use_pre else img_bgr
    out = colorize(inp)
    if use_pre:
        out = keep_original_L(img_bgr, out)
    if use_post:
        out = postprocessing.refine(out, saturation)
    return out


def run_pipeline(image_path, use_pre=False, use_post=True, model="finetuned"):
    img = read_image(image_path)
    if img is None:
        raise ValueError(f"Görüntü okunamadı: {image_path}")
    return process(img, model, use_pre, use_post)


if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else "finetuned"
    cv2.imwrite(f"samples/{model}_raw.jpg", run_pipeline(path, use_post=False, model=model))
    cv2.imwrite(f"samples/{model}_post.jpg", run_pipeline(path, use_post=True, model=model))
    print(f"Kaydedildi: samples/{model}_raw.jpg ve samples/{model}_post.jpg")