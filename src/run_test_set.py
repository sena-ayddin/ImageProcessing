import os
import glob
import cv2
from models.opencv_baseline import colorize
from src import postprocessing

SRC = "data/test_30"
OUT = "data/test_30_results"
MAX_SIDE = 1024


def resize_max(img, max_side=MAX_SIDE):
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def to_gray3(img):
    return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def main():
    for sub in ["original", "gray", "raw", "post"]:
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)

    for path in sorted(glob.glob(os.path.join(SRC, "*.jpg"))):
        name = os.path.basename(path)
        img = cv2.imread(path)
        if img is None:
            print("Okunamadi:", name)
            continue

        img = resize_max(img)
        gray = to_gray3(img)
        raw = colorize(gray)
        post = postprocessing.refine(raw, saturation=1.2)

        if name.startswith("colored_"):
            cv2.imwrite(os.path.join(OUT, "original", name), img)
        cv2.imwrite(os.path.join(OUT, "gray", name), gray)
        cv2.imwrite(os.path.join(OUT, "raw", name), raw)
        cv2.imwrite(os.path.join(OUT, "post", name), post)
        print("Tamam:", name)


if __name__ == "__main__":
    main()