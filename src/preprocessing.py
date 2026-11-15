import cv2


def apply_clahe(img_bgr, clip_limit=2.0, tile_size=8):
    """Kontrastı artırır (sadece L kanalında), renk bilgisine dokunmaz."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)