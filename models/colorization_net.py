import os
import cv2
import numpy as np
import torch
from models.colorizers import eccv16

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(MODEL_DIR, "eccv16_finetuned.pt")  # fine-tuned weights go here
SIZE = 256

_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = None


def _load_model():
    """Load the model on the first call and keep it in memory."""
    global _model
    if _model is None:
        if os.path.exists(WEIGHTS):
            m = eccv16(pretrained=False)
            m.load_state_dict(torch.load(WEIGHTS, map_location=_device))
        else:
            m = eccv16(pretrained=True)  # fall back to the original pretrained weights
        _model = m.to(_device).eval()
    return _model


def colorize(img_bgr):
    """Take a BGR uint8 image, return a colorized BGR uint8 image."""
    model = _load_model()
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    h, w = img_bgr.shape[:2]

    lab = cv2.cvtColor(img_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    L = lab[:, :, 0]  # 0-100, original size

    L_small = cv2.resize(L, (SIZE, SIZE))
    t = torch.from_numpy(L_small)[None, None].to(_device)  # (1,1,256,256)
    with torch.no_grad():
        ab = model(t)[0].cpu().numpy().transpose(1, 2, 0)  # (256,256,2)

    ab = cv2.resize(ab, (w, h))
    lab_out = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    bgr_out = cv2.cvtColor(lab_out, cv2.COLOR_Lab2BGR)
    return (np.clip(bgr_out, 0, 1) * 255).astype(np.uint8)