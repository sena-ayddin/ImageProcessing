import os
import cv2
import numpy as np

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.join(MODEL_DIR, "colorization_deploy_v2.prototxt")
WEIGHTS = os.path.join(MODEL_DIR, "colorization_release_v2.caffemodel")
POINTS = os.path.join(MODEL_DIR, "pts_in_hull.npy")

_net = None


def _load_net():
    global _net
    if _net is None:
        net = cv2.dnn.readNetFromCaffe(PROTO, WEIGHTS)
        pts = np.load(POINTS).transpose().reshape(2, 313, 1, 1)
        net.getLayer(net.getLayerId("class8_ab")).blobs = [pts.astype(np.float32)]
        net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [
            np.full([1, 313], 2.606, dtype=np.float32)
        ]
        _net = net
    return _net


def colorize(img_bgr):
    """BGR uint8 görüntü alır, renklendirilmiş BGR uint8 görüntü döndürür."""
    net = _load_net()
    h, w = img_bgr.shape[:2]

    scaled = img_bgr.astype(np.float32) / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2Lab)
    L = lab[:, :, 0]

    L_resized = cv2.resize(L, (224, 224)) - 50
    net.setInput(cv2.dnn.blobFromImage(L_resized))
    ab = net.forward()[0].transpose((1, 2, 0))
    ab = cv2.resize(ab, (w, h))

    lab_out = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    bgr_out = cv2.cvtColor(lab_out, cv2.COLOR_Lab2BGR)
    return (np.clip(bgr_out, 0, 1) * 255).astype(np.uint8)