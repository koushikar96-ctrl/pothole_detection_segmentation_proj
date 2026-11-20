from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def paired_stems(data_root: Path, split: str = "train") -> List[str]:
    image_dir, mask_dir = _data_directories(Path(data_root), split)
    images = {path.stem for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS}
    masks = {path.stem for path in mask_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS}
    return sorted(images & masks)


def _data_directories(data_root: Path, split: str = "train") -> Tuple[Path, Path]:
    if (data_root / "JPEGImages").exists():
        return data_root / "JPEGImages", data_root / "SegmentationClass"
    return data_root / "images" / split, data_root / "masks" / split


def _find_file(directory: Path, stem: str) -> Path:
    matches = [path for path in directory.glob(f"{stem}.*") if path.suffix.lower() in IMAGE_EXTENSIONS]
    if not matches:
        raise FileNotFoundError(f"No file found for {stem} in {directory}")
    return matches[0]


def build_train_augmentation(image_size: int = 256) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.3),
            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), shear=(-8, 8), p=0.25),
        ],
        additional_targets={"mask": "mask"},
    )


def build_validation_augmentation(image_size: int = 256) -> A.Compose:
    return A.Compose([A.Resize(image_size, image_size)], additional_targets={"mask": "mask"})


class PotholeSegmentationDataset(Dataset):
    def __init__(self, data_root: Path, stems: List[str], image_size: int = 256, training: bool = False, split: str = "train") -> None:
        self.data_root = Path(data_root)
        self.stems = stems
        self.image_dir, self.mask_dir = _data_directories(self.data_root, split)
        self.transform = build_train_augmentation(image_size) if training else build_validation_augmentation(image_size)

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        stem = self.stems[index]
        image = cv2.imread(str(_find_file(self.image_dir, stem)), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(_find_file(self.mask_dir, stem)), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise ValueError(f"Unable to read paired sample {stem}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=image, mask=mask)
        image = transformed["image"].astype(np.float32) / 255.0
        mask = (transformed["mask"] > 0).astype(np.float32)
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1).copy())
        mask_tensor = torch.from_numpy(mask[None, ...].copy())
        return image_tensor, mask_tensor


def summarize_sample(dataset: PotholeSegmentationDataset, index: int = 0) -> Dict[str, object]:
    image, mask = dataset[index]
    return {
        "image_shape": list(image.shape),
        "mask_shape": list(mask.shape),
        "image_range": [float(image.min()), float(image.max())],
        "mask_values": sorted(torch.unique(mask).tolist()),
        "mask_pixels": int(mask.sum().item()),
    }