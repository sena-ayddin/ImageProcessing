from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from evaluation.metrics import METRIC_COLS, METRIC_LABELS, compute_metrics
from src.pipeline import run_pipeline

IMG_DIR = Path("data/raw/val_256")
SPLIT_FILE = Path("data/splits/test.txt")
OUTPUT_DIR = Path("data/output")
NUM_TEST_IMAGES = 50  # Set to None to evaluate all 1,000 test images

VARIANTS = [
    ("1. Model Only (Raw)", False, False),
    ("2. Pre + Model", True, False),
    ("3. Model + Post", False, True),
    ("4. Full Hybrid (Pre + Model + Post)", True, True),
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    test_files = [line.strip() for line in SPLIT_FILE.read_text(encoding="utf-8").splitlines() if line.strip()][:NUM_TEST_IMAGES]
    gt_cache = {f: cv2.cvtColor(cv2.imread(str(IMG_DIR / f)), cv2.COLOR_BGR2RGB) for f in test_files}

    records = []
    for variant_name, use_pre, use_post in VARIANTS:
        for fname in tqdm(test_files, desc=variant_name):
            out_bgr = run_pipeline(str(IMG_DIR / fname), use_pre=use_pre, use_post=use_post)
            pred_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)

            row = compute_metrics(gt_cache[fname], pred_rgb)
            row["Variant"] = variant_name
            row["filename"] = fname
            records.append(row)

    full_df = pd.DataFrame(records)
    summary = full_df.groupby("Variant")[METRIC_COLS].mean().rename(columns=METRIC_LABELS).T
    summary.index.name = "Metric"

    full_df.to_csv(OUTPUT_DIR / "pipeline_final_results.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "pipeline_summary.csv")

    print(f"\n=== PIPELINE EVALUATION ({len(test_files)} test images) ===")
    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()