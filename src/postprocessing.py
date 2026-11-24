import cv2
import numpy as np


def guided_smooth_color(img_bgr, radius=8, eps=100):
    """Renk kanallarını (a, b) parlaklığı (L) rehber alarak yumuşatır.
    Renk taşmalarını azaltır, kenarları korur."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    a = cv2.ximgproc.guidedFilter(L, a, radius, eps)
    b = cv2.ximgproc.guidedFilter(L, b, radius, eps)
    return cv2.cvtColor(cv2.merge([L, a, b]), cv2.COLOR_LAB2BGR)


def adjust_saturation(img_bgr, factor=1.3):
    """Renkleri canlandırır. factor > 1 artırır, < 1 azaltır."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def refine(img_bgr, saturation=1.0):
    out = guided_smooth_color(img_bgr)
    if saturation != 1.0:
        out = adjust_saturation(out, saturation)
    return out