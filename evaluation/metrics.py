import argparse
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import pandas as pd
from skimage.metrics import mean_squared_error, structural_similarity
from tqdm import tqdm

OUTPUT_DIR = Path("data/output")

METRIC_LABELS = {
    "MAE_RGB": "MAE_RGB (Mean Absolute Error - RGB) ↓",
    "MSE_RGB": "MSE_RGB (Mean Squared Error - RGB) ↓",
    "MAE_lab_ab": "MAE_lab_ab (Mean Absolute Error - CIE Lab a*b*) ↓",
    "SSIM": "SSIM (Structural Similarity Index Measure) ↑",
}
METRIC_COLS = list(METRIC_LABELS.keys())


def compute_metrics(gt_rgb: np.ndarray, pred_rgb: np.ndarray) -> Dict[str, float]:
    gt_f = gt_rgb.astype(np.float32)
    pr_f = pred_rgb.astype(np.float32)

    gt_lab = cv2.cvtColor(gt_f / 255.0, cv2.COLOR_RGB2LAB)
    pr_lab = cv2.cvtColor(pr_f / 255.0, cv2.COLOR_RGB2LAB)

    return {
        "MAE_RGB": float(np.mean(np.abs(gt_f - pr_f))),
        "MSE_RGB": float(mean_squared_error(gt_rgb, pred_rgb)),
        "MAE_lab_ab": float(np.mean(np.abs(gt_lab[:, :, 1:3] - pr_lab[:, :, 1:3]))),
        "SSIM": float(structural_similarity(gt_rgb, pred_rgb, channel_axis=-1, data_range=255)),
    }


def evaluate_directory(gt_dir: str, pred_dir: str, split_file: Optional[str] = None) -> pd.DataFrame:
    gt_root, pred_root = Path(gt_dir), Path(pred_dir)

    if split_file:
        filenames = [line.strip() for line in Path(split_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        filenames = sorted({p.name for p in gt_root.iterdir()} & {p.name for p in pred_root.iterdir()})

    records = []
    for fname in tqdm(filenames, desc="Evaluating directory"):
        gt_bgr = cv2.imread(str(gt_root / fname))
        pred_bgr = cv2.imread(str(pred_root / fname))
        if gt_bgr is None or pred_bgr is None:
            continue

        if gt_bgr.shape != pred_bgr.shape:
            pred_bgr = cv2.resize(pred_bgr, (gt_bgr.shape[1], gt_bgr.shape[0]))

        row = compute_metrics(cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB), cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2RGB))
        row["filename"] = fname
        records.append(row)

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Evaluate colorization outputs and export CSV.")
    parser.add_argument("--gt_dir", default="data/raw/val_256")
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--split_file", default=None)
    parser.add_argument("--model_name", default="Baseline")
    parser.add_argument("--out_csv", default=str(OUTPUT_DIR / "results_baseline.csv"))
    args = parser.parse_args()

    df = evaluate_directory(args.gt_dir, args.pred_dir, args.split_file)
    print(f"\n=== Evaluation Summary: {args.model_name} ({len(df)} images) ===")
    for k, v in df[METRIC_COLS].mean().items():
        print(f"  {METRIC_LABELS[k]:<50}: {v:.4f}")

    out_path = Path(args.out_csv)
    if out_path.parent == Path("."):
        out_path = OUTPUT_DIR / out_path.name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved per-image metrics CSV -> {out_path}")


if __name__ == "__main__":
    main()