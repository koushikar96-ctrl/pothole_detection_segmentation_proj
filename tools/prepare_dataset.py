from __future__ import annotations

import csv
import json
import random
import shutil
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "dataset"
OUTPUT = ROOT / "prepared_dataset"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SEED = 42


def files_by_stem(directory: Path) -> dict[str, Path]:
    return {path.stem: path for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS}


def _register_unique_pair(pair_map: dict[str, tuple[Path, Path]], stem: str, image_path: Path, mask_path: Path) -> None:
    candidate = stem
    suffix = 1
    while candidate in pair_map:
        candidate = f"{stem}_{suffix}"
        suffix += 1
    pair_map[candidate] = (image_path, mask_path)


def _candidate_image_mask_dirs(source_root: Path) -> list[tuple[Path, Path]]:
    candidates: list[tuple[Path, Path]] = []
    seen: set[tuple[str, str]] = set()
    variants = [
        ("JPEGImages", "SegmentationClass"),
        ("images", "masks"),
        ("Images", "Masks"),
    ]
    for image_name, mask_name in variants:
        image_dir = source_root / image_name
        mask_dir = source_root / mask_name
        if image_dir.exists() and mask_dir.exists():
            canonical_pair = (
                str(image_dir.resolve()),
                str(mask_dir.resolve()),
            )
            if canonical_pair in seen:
                continue
            seen.add(canonical_pair)
            candidates.append((image_dir, mask_dir))

    for split in ("train", "val", "test"):
        image_dir = source_root / "images" / split
        mask_dir = source_root / "masks" / split
        if image_dir.exists() and mask_dir.exists():
            canonical_pair = (
                str(image_dir.resolve()),
                str(mask_dir.resolve()),
            )
            if canonical_pair in seen:
                continue
            seen.add(canonical_pair)
            candidates.append((image_dir, mask_dir))

    return candidates


def collect_paired_samples(source_roots: list[Path] | None = None) -> dict[str, tuple[Path, Path]]:
    roots = [Path(root) for root in (source_roots or [SOURCE])]
    pairs: dict[str, tuple[Path, Path]] = {}

    for source_root in roots:
        candidates = _candidate_image_mask_dirs(source_root)
        if not candidates:
            raise ValueError(
                f"Unsupported dataset layout in {source_root}. Expected a folder like JPEGImages/SegmentationClass, "
                "images/masks, or images/<split>/masks/<split>/ ."
            )

        for image_dir, mask_dir in candidates:
            images = files_by_stem(image_dir)
            masks = files_by_stem(mask_dir)
            for stem in sorted(images.keys() & masks.keys()):
                final_stem = stem
                suffix = 1
                while final_stem in pairs:
                    final_stem = f"{stem}_{suffix}"
                    suffix += 1
                pairs[final_stem] = (images[stem], masks[stem])

    return pairs


def yolo_boxes(mask_path: Path) -> list[tuple[float, float, float, float]]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Unable to read mask: {mask_path}")
    height, width = mask.shape
    binary = (mask > 0).astype("uint8")
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes = []
    for x, y, box_width, box_height, area in stats[1:]:
        if area < 20:
            continue
        boxes.append(
            (
                (x + box_width / 2) / width,
                (y + box_height / 2) / height,
                box_width / width,
                box_height / height,
            )
        )
    return boxes


def write_yolo_label(label_path: Path, boxes: list[tuple[float, float, float, float]]) -> None:
    label_path.write_text(
        "".join(f"0 {x:.6f} {y:.6f} {width:.6f} {height:.6f}\n" for x, y, width, height in boxes),
        encoding="utf-8",
    )


def _count_image_files(source_root: Path) -> int:
    for image_name in ("JPEGImages", "Images", "images"):
        image_dir = source_root / image_name
        if image_dir.exists():
            total = 0
            if image_dir.is_dir():
                for child in image_dir.iterdir():
                    if child.is_dir():
                        total += len(files_by_stem(child))
                    else:
                        total += len(files_by_stem(image_dir))
                        break
                return total
            return len(files_by_stem(image_dir))
    return 0


def _count_mask_files(source_root: Path) -> int:
    for mask_name in ("SegmentationClass", "Masks", "masks"):
        mask_dir = source_root / mask_name
        if mask_dir.exists():
            total = 0
            if mask_dir.is_dir():
                for child in mask_dir.iterdir():
                    if child.is_dir():
                        total += len(files_by_stem(child))
                    else:
                        total += len(files_by_stem(mask_dir))
                        break
                return total
            return len(files_by_stem(mask_dir))
    return 0


def prepare_dataset(
    source_roots: list[Path] | None = None,
    output_dir: Path = OUTPUT,
    seed: int = SEED,
) -> dict[str, object]:
    roots = [Path(root) for root in (source_roots or [SOURCE])]
    paired_map = collect_paired_samples(roots)
    paired = sorted(paired_map)
    unmasked: list[str] = []

    if len(paired) < 3:
        raise ValueError("At least three paired samples are required.")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    for split in ("train", "val", "test"):
        for category in ("images", "labels", "masks"):
            (output_dir / category / split).mkdir(parents=True, exist_ok=True)

    shuffled = list(paired)
    random.Random(seed).shuffle(shuffled)
    train_end = round(len(shuffled) * 0.70)
    val_end = train_end + round(len(shuffled) * 0.15)
    split_stems = {"train": shuffled[:train_end], "val": shuffled[train_end:val_end], "test": shuffled[val_end:]}
    manifest_rows = []
    for split, stems in split_stems.items():
        for stem in stems:
            image_path, mask_path = paired_map[stem]
            image_copy = output_dir / "images" / split / f"{stem}{image_path.suffix}"
            mask_copy = output_dir / "masks" / split / f"{stem}{mask_path.suffix}"
            shutil.copy2(image_path, image_copy)
            shutil.copy2(mask_path, mask_copy)
            boxes = yolo_boxes(mask_path)
            write_yolo_label(output_dir / "labels" / split / f"{stem}.txt", boxes)
            manifest_rows.append({"stem": stem, "split": split, "boxes": len(boxes)})

    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stem", "split", "boxes"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    (output_dir / "unmasked_for_review.csv").write_text("stem\n" + "\n".join(unmasked) + "\n", encoding="utf-8")
    yaml_text = (
        f"path: {output_dir.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 1\n"
        "names: [pothole]\n"
    )
    (output_dir / "dataset.yaml").write_text(yaml_text, encoding="utf-8")
    summary = {
        "seed": seed,
        "source_roots": [str(root) for root in roots],
        "source_images": sum(_count_image_files(path) for path in roots),
        "source_masks": sum(_count_mask_files(path) for path in roots),
        "paired_samples": len(paired),
        "unmasked_for_review": len(unmasked),
        "splits": {split: len(stems) for split, stems in split_stems.items()},
        "total_boxes": sum(row["boxes"] for row in manifest_rows),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare a merged pothole segmentation dataset from one or more source roots.")
    parser.add_argument(
        "--source",
        dest="sources",
        action="append",
        default=[],
        help="Dataset root containing JPEGImages/SegmentationClass or images/masks splits. Repeat to merge several sources.",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT, help="Prepared output directory")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed used for splitting")
    args = parser.parse_args()
    selected_sources = [Path(value) for value in args.sources] if args.sources else [SOURCE]
    print(json.dumps(prepare_dataset(source_roots=selected_sources, output_dir=args.output, seed=args.seed), indent=2))
