import csv
import glob
import os

import cv2
import numpy as np

from evaluation.metrics import METRIC_COLS, compute_metrics
from src import postprocessing
from src.pipeline import process, read_image

SRC = "data/test_30"
MAX_SIDE = 1024
OUT_CSV = "samples/compare_models.csv"

MODELS = ["opencv", "finetuned"]
POSTS = {
    "post yok": lambda x: x,
    "sadece guided filter": postprocessing.guided_smooth_color,
    "sadece saturation 1.1": lambda x: postprocessing.adjust_saturation(x, 1.1),
    "sadece saturation 1.2": lambda x: postprocessing.adjust_saturation(x, 1.2),
    "guided + saturation 1.2": lambda x: postprocessing.refine(x, 1.2),
}


def resize_max(img):
    h, w = img.shape[:2]
    s = MAX_SIDE / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main():
    paths = sorted(glob.glob(os.path.join(SRC, "colored_*.jpg")))
    scores = {(m, p): [] for m in MODELS for p in POSTS}

    for path in paths:
        ref = resize_max(read_image(path))
        gray = cv2.cvtColor(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        for model in MODELS:
            raw = process(gray, model, use_pre=False, use_post=False)
            for pname, fn in POSTS.items():
                m = compute_metrics(to_rgb(ref), to_rgb(fn(raw)))
                scores[(model, pname)].append([m[c] for c in METRIC_COLS])
        print("Tamam:", os.path.basename(path))

    os.makedirs("samples", exist_ok=True)
    header = ["Model", "Post-processing"] + METRIC_COLS
    print(f"\n{'Model':10} {'Post-processing':25} " + " ".join(f"{c:>11}" for c in METRIC_COLS))
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for (model, pname), vals in scores.items():
            mean = np.mean(vals, axis=0)
            w.writerow([model, pname] + [f"{x:.4f}" for x in mean])
            print(f"{model:10} {pname:25} " + " ".join(f"{x:11.4f}" for x in mean))
    print("\nKaydedildi:", OUT_CSV)


if __name__ == "__main__":
    main()