import cv2
import numpy as np
from src import postprocessing
from models.opencv_baseline import colorize
from src.pipeline import load_image


def make_comparison(image_path, out_path="samples/comparison.jpg"):
    img = load_image(image_path)
    gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

    raw = colorize(gray)
    smooth = postprocessing.guided_smooth_color(raw)
    sat = postprocessing.adjust_saturation(raw, 1.2)
    both = postprocessing.adjust_saturation(smooth, 1.2)

    panels = [gray, raw, smooth, sat, both]
    labels = ["Gri", "Ham model", "Guided filter", "Saturation 1.2", "Ikisi birden"]
    for p, t in zip(panels, labels):
        cv2.putText(p, t, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imwrite(out_path, np.hstack(panels))
    print("Kaydedildi:", out_path)


if __name__ == "__main__":
    import sys
    make_comparison(sys.argv[1])
    