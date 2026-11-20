"""Create ROI image/mask pairs for HARP-UNet training."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_file(folder: Path, stem: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = folder / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def prepare_split(source_root: Path, output_root: Path, split: str, margin: float) -> int:
    image_dir = source_root / "images" / split
    mask_dir = source_root / "masks" / split
    output_image_dir = output_root / "images" / split
    output_mask_dir = output_root / "masks" / split
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)

    created = 0
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        mask_path = find_file(mask_dir, image_path.stem)
        if mask_path is None:
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            continue

        binary = (mask > 0).astype(np.uint8)
        component_count, component_labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        for component_index in range(1, component_count):
            x, y, width, height, area = stats[component_index]
            if area < 16:
                continue

            pad_x = max(2, int(width * margin))
            pad_y = max(2, int(height * margin))
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(image.shape[1], x + width + pad_x)
            y2 = min(image.shape[0], y + height + pad_y)

            roi_image = image[y1:y2, x1:x2]
            roi_mask = (component_labels[y1:y2, x1:x2] == component_index).astype(np.uint8) * 255
            roi_stem = f"{image_path.stem}_roi_{component_index:03d}"
            cv2.imwrite(str(output_image_dir / f"{roi_stem}.jpg"), roi_image)
            cv2.imwrite(str(output_mask_dir / f"{roi_stem}.png"), roi_mask)
            created += 1

    return created


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ROI crops for HARP-UNet training")
    parser.add_argument("--source", type=Path, default=Path("prepared_dataset"))
    parser.add_argument("--output", type=Path, default=Path("prepared_roi_dataset"))
    parser.add_argument("--margin", type=float, default=0.15)
    args = parser.parse_args()

    total = 0
    for split in ("train", "val", "test"):
        count = prepare_split(args.source, args.output, split, args.margin)
        print(f"{split}: {count} ROI samples")
        total += count
    print(f"Total ROI samples: {total}")


if __name__ == "__main__":
    main()