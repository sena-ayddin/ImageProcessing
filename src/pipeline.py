import cv2
from models.opencv_baseline import colorize
from src import preprocessing, postprocessing


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Görüntü okunamadı: {path}")
    return img


def run_pipeline(image_path, use_pre=False, use_post=True):
    img = load_image(image_path)
    if use_pre:
        img = preprocessing.apply_clahe(img)
    out = colorize(img)
    if use_post:
        out = postprocessing.refine(out)
    return out


if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    cv2.imwrite("samples/output_raw.jpg", run_pipeline(path, use_post=False))
    cv2.imwrite("samples/output_post.jpg", run_pipeline(path, use_post=True))
    print("Kaydedildi: samples/output_raw.jpg ve samples/output_post.jpg")
    