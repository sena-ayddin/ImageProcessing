from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

NORM_PARAMS = {
    "zhang": (50.0, 100.0, 110.0),
    "standard": (0.0, 100.0, 128.0),
}


class ColorizationDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        split_file: str,
        img_size: Tuple[int, int] = (256, 256),
        norm_mode: str = "zhang",
    ):
        if norm_mode not in NORM_PARAMS:
            raise ValueError(f"norm_mode must be one of {list(NORM_PARAMS)}")

        self.img_dir = Path(img_dir)
        self.img_size = img_size
        self.norm_mode = norm_mode
        self.filenames = [line.strip() for line in Path(split_file).read_text(encoding="utf-8").splitlines() if line.strip()]

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.img_dir / self.filenames[idx]
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        if (bgr.shape[1], bgr.shape[0]) != self.img_size:
            bgr = cv2.resize(bgr, self.img_size, interpolation=cv2.INTER_AREA)

        rgb_float = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        lab = cv2.cvtColor(rgb_float, cv2.COLOR_RGB2LAB)

        l_shift, l_scale, ab_scale = NORM_PARAMS[self.norm_mode]
        l_norm = (lab[:, :, 0] - l_shift) / l_scale
        ab_norm = np.clip(lab[:, :, 1:3] / ab_scale, -1.0, 1.0)

        x = torch.from_numpy(l_norm[None].astype(np.float32))
        y = torch.from_numpy(ab_norm.transpose(2, 0, 1).astype(np.float32))
        return x, y


def create_dataloaders(
    img_dir: str = "data/raw/val_256",
    split_dir: str = "data/splits",
    batch_size: int = 32,
    num_workers: int = 2,
    img_size: Tuple[int, int] = (256, 256),
    norm_mode: str = "zhang",
) -> Dict[str, DataLoader]:
    split_root = Path(split_dir)
    pin = torch.cuda.is_available()
    loaders = {}

    for name in ("pilot_train", "pilot_val", "train", "val", "test"):
        ds = ColorizationDataset(img_dir, str(split_root / f"{name}.txt"), img_size, norm_mode)
        is_train = "train" in name
        loaders[name] = DataLoader(
            ds,
            batch_size=min(batch_size, len(ds)),
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=pin,
            drop_last=is_train,
        )
    return loaders