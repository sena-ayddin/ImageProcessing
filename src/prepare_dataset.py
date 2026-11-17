import random
import tarfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
IMG_DIR = RAW_DIR / "val_256"
SPLIT_DIR = DATA_DIR / "splits"

PLACES_VAL_URL = "http://data.csail.mit.edu/places/places365/val_256.tar"
IMG_SIZE = (256, 256)
TARGET_TOTAL = 10000
MIN_MEAN_CHROMA = 8.0
SEED = 42


def download_and_extract():
    if IMG_DIR.exists():
        print(f"Images already extracted at {IMG_DIR}")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tar_path = RAW_DIR / "val_256.tar"
    if not tar_path.exists():
        print(f"Downloading Places365 val_256 from {PLACES_VAL_URL}...")
        urllib.request.urlretrieve(PLACES_VAL_URL, tar_path)

    print("Extracting archive...")
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=RAW_DIR)


def is_suitable_image(img_path: Path) -> bool:
    bgr = cv2.imread(str(img_path))
    if bgr is None or (bgr.shape[1], bgr.shape[0]) != IMG_SIZE:
        return False

    lab = cv2.cvtColor(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    chroma = np.hypot(lab[:, :, 1], lab[:, :, 2])
    return float(np.mean(chroma)) >= MIN_MEAN_CHROMA


def build_splits():
    all_images = sorted(IMG_DIR.glob("*.jpg"))
    valid_files = [p.name for p in tqdm(all_images, desc="Filtering") if is_suitable_image(p)]

    if len(valid_files) < TARGET_TOTAL:
        raise RuntimeError(f"Found only {len(valid_files)} valid images; need {TARGET_TOTAL}.")

    random.Random(SEED).shuffle(valid_files)
    selected = valid_files[:TARGET_TOTAL]

    splits = {
        "train.txt": selected[:8000],
        "val.txt": selected[8000:9000],
        "test.txt": selected[9000:],
        "pilot_train.txt": selected[:80],
        "pilot_val.txt": selected[80:100],
    }

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, items in splits.items():
        out_path = SPLIT_DIR / filename
        out_path.write_text("\n".join(items) + "\n", encoding="utf-8")
        print(f"Saved {len(items):>5} entries -> {out_path}")


if __name__ == "__main__":
    download_and_extract()
    build_splits()