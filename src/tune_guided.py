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
OUT_CSV = "samples/guided_grid.csv"

MODELS = ["opencv", "finetuned"]
RADII = [2, 4, 8, 16, 32]
EPSS = [25, 100, 400, 1600]


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
    configs = [("yok", 0, 0)] + [("guided", r, e) for r in RADII for e in EPSS]
    scores = {(m, c): [] for m in MODELS for c in configs}

    for path in paths:
        ref = resize_max(read_image(path))
        ref_rgb = to_rgb(ref)
        gray = cv2.cvtColor(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

        for model in MODELS:
            raw = process(gray, model, use_pre=False, use_post=False)
            for cfg in configs:
                name, r, e = cfg
                out = raw if name == "yok" else postprocessing.guided_smooth_color(raw, radius=r, eps=e)
                m = compute_metrics(ref_rgb, to_rgb(out))
                scores[(model, cfg)].append([m[c] for c in METRIC_COLS])
        print("Tamam:", os.path.basename(path))

    os.makedirs("samples", exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Model", "radius", "eps"] + METRIC_COLS)
        for (model, (name, r, e)), vals in scores.items():
            mean = np.mean(vals, axis=0)
            w.writerow([model, r, e] + [f"{x:.4f}" for x in mean])

    ab_idx = METRIC_COLS.index("MAE_lab_ab")
    for model in MODELS:
        rows = []
        for cfg in configs:
            mean = np.mean(scores[(model, cfg)], axis=0)
            rows.append((cfg, mean))
        rows.sort(key=lambda x: x[1][ab_idx])

        print(f"\n=== {model} (MAE_lab_ab'ye göre, en iyi 8 ve 'yok') ===")
        print(f"{'radius':>6} {'eps':>5} " + " ".join(f"{c:>11}" for c in METRIC_COLS))
        shown = rows[:8] + [r for r in rows[8:] if r[0][0] == "yok"]
        for (name, r, e), mean in shown:
            label_r, label_e = ("-", "-") if name == "yok" else (r, e)
            print(f"{label_r:>6} {label_e:>5} " + " ".join(f"{x:11.4f}" for x in mean))

    print("\nTüm tablo:", OUT_CSV)


if __name__ == "__main__":
    main()