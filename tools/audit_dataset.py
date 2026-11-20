from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config import AUDIT_DIR, DATASET_DIR, RANDOM_SEED

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def files_by_stem(directory: Path) -> dict[str, Path]:
    return {path.stem: path for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_dataset() -> dict[str, object]:
    images = files_by_stem(DATASET_DIR / "JPEGImages")
    masks = files_by_stem(DATASET_DIR / "SegmentationClass")
    paired_stems = sorted(images.keys() & masks.keys())
    unmasked_stems = sorted(images.keys() - masks.keys())
    rows = []
    size_mismatches = []
    mask_values: set[int] = set()
    for stem in paired_stems:
        with Image.open(images[stem]) as image, Image.open(masks[stem]) as mask:
            image_size = image.size
            mask_size = mask.size
            values = {value for value, count in enumerate(mask.convert("L").histogram()) if count}
        mask_values.update(values)
        if image_size != mask_size:
            size_mismatches.append(stem)
        rows.append({
            "stem": stem,
            "image": str(images[stem].relative_to(DATASET_DIR)),
            "mask": str(masks[stem].relative_to(DATASET_DIR)),
            "image_width": image_size[0],
            "image_height": image_size[1],
            "mask_width": mask_size[0],
            "mask_height": mask_size[1],
            "image_sha256": file_hash(images[stem]),
        })
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    with (AUDIT_DIR / "paired_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys() if rows else ["stem"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "seed": RANDOM_SEED,
        "total_images": len(images),
        "total_masks": len(masks),
        "paired_samples": len(paired_stems),
        "images_without_masks": len(unmasked_stems),
        "unmasked_stems": unmasked_stems,
        "mask_values": sorted(mask_values),
        "size_mismatches": size_mismatches,
    }
    (AUDIT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (AUDIT_DIR / "README.txt").write_text(
        "Review unmasked_stems manually before adding any as negative detection samples.\n",
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(audit_dataset(), indent=2))