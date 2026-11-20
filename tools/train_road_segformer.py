from __future__ import annotations

import argparse
import json
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.road_segmentation import DEFAULT_ROAD_SEGMENTATION_MODEL_ID  # noqa: E402

ROAD_LABEL = 1
IGNORE_LABEL = 255
CLASS_TO_ID = {"background": 0, "road": ROAD_LABEL}
ROAD_ANNOTATION_LABELS = {"road_surface", "marking"}
IDD_ROAD_ANNOTATION_LABELS = {"road"}


def load_samples(dataset_dir: Path) -> list[tuple[Path, np.ndarray]]:
    """Read the Kaggle CVAT polygons and rasterize road plus marking regions."""
    annotation_path = dataset_dir / "annotations.xml"
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Missing road segmentation annotations: {annotation_path}")

    root = ET.parse(annotation_path).getroot()
    samples: list[tuple[Path, np.ndarray]] = []
    for image_node in root.findall("image"):
        image_path = dataset_dir / image_node.attrib["name"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image referenced by annotations: {image_path}")
        width = int(image_node.attrib["width"])
        height = int(image_node.attrib["height"])
        mask = np.zeros((height, width), dtype=np.uint8)
        polygons = sorted(
            image_node.findall("polygon"),
            key=lambda polygon: int(polygon.attrib.get("z_order", 0)),
        )
        for polygon in polygons:
            coordinates = [
                tuple(round(float(value)) for value in point.split(","))
                for point in polygon.attrib["points"].split(";")
                if point
            ]
            if len(coordinates) < 3:
                continue
            class_id = (
                ROAD_LABEL
                if polygon.attrib["label"] in ROAD_ANNOTATION_LABELS
                else CLASS_TO_ID["background"]
            )
            cv2.fillPoly(mask, [np.asarray(coordinates, dtype=np.int32)], class_id)

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to decode road-training image: {image_path}")
        if image.shape[:2] != mask.shape:
            raise ValueError(
                f"Image and polygon dimensions differ for {image_path}: "
                f"{image.shape[:2]} vs {mask.shape}"
            )
        samples.append((image_path, mask))

    if len(samples) < 2:
        raise ValueError("At least two annotated road images are needed for train/validation.")
    return samples


def load_idd_samples(
    dataset_dir: Path,
    split: str,
    image_size: tuple[int, int],
    limit: int | None,
    seed: int,
) -> list[tuple[Path, np.ndarray]]:
    """Rasterize IDD road polygons into resized binary masks."""
    annotation_dir = dataset_dir / "gtFine" / split
    if not annotation_dir.is_dir():
        raise FileNotFoundError(f"Missing IDD {split} annotations: {annotation_dir}")

    annotation_paths = sorted(annotation_dir.rglob("*_gtFine_polygons.json"))
    if not annotation_paths:
        raise ValueError(f"No IDD polygon annotations found under {annotation_dir}")
    if limit is not None and len(annotation_paths) > limit:
        annotation_paths = sorted(random.Random(seed).sample(annotation_paths, limit))

    target_width, target_height = image_size
    samples: list[tuple[Path, np.ndarray]] = []
    for annotation_path in annotation_paths:
        relative_path = annotation_path.relative_to(annotation_dir)
        image_name = annotation_path.name.replace(
            "_gtFine_polygons.json", "_leftImg8bit.jpg"
        )
        image_path = dataset_dir / "leftImg8bit" / split / relative_path.parent / image_name
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing IDD image for annotation {annotation_path}: {image_path}")

        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        source_width = int(annotation["imgWidth"])
        source_height = int(annotation["imgHeight"])
        if source_width < 1 or source_height < 1:
            raise ValueError(f"Invalid image dimensions in IDD annotation: {annotation_path}")

        mask = np.zeros((target_height, target_width), dtype=np.uint8)
        scale = np.asarray(
            [
                max(0, target_width - 1) / max(1, source_width - 1),
                max(0, target_height - 1) / max(1, source_height - 1),
            ],
            dtype=np.float64,
        )
        road_polygons: list[np.ndarray] = []
        fallback_polygons: list[np.ndarray] = []
        objects = [
            item
            for item in annotation.get("objects", [])
            if not item.get("deleted", 0) and item.get("draw") is not False
        ]
        for item in objects:
            label = str(item.get("label", ""))
            if label not in IDD_ROAD_ANNOTATION_LABELS and "fallback" not in label.casefold():
                continue
            points = np.asarray(item.get("polygon", []), dtype=np.float64)
            if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
                continue
            points = np.rint(points * scale).astype(np.int32)
            points[:, 0] = np.clip(points[:, 0], 0, target_width - 1)
            points[:, 1] = np.clip(points[:, 1], 0, target_height - 1)
            if label in IDD_ROAD_ANNOTATION_LABELS:
                road_polygons.append(points)
            else:
                fallback_polygons.append(points)
        for polygon in road_polygons:
            cv2.fillPoly(mask, [polygon], ROAD_LABEL)
        for polygon in fallback_polygons:
            cv2.fillPoly(mask, [polygon], IGNORE_LABEL)
        samples.append((image_path, mask))

    return samples


class RoadPolygonDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[Path, np.ndarray]],
        indices: list[int],
        image_size: tuple[int, int],
        image_mean: list[float],
        image_std: list[float],
        augment: bool,
    ) -> None:
        self.samples = samples
        self.indices = indices
        self.image_size = image_size
        self.image_mean = np.asarray(image_mean, dtype=np.float32).reshape(1, 1, 3)
        self.image_std = np.asarray(image_std, dtype=np.float32).reshape(1, 1, 3)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask = self.samples[self.indices[index]]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to decode road-training image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        width, height = self.image_size
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

        if self.augment and random.random() < 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)

        image_float = image.astype(np.float32) / 255.0
        image_float = (image_float - self.image_mean) / self.image_std
        image_tensor = torch.from_numpy(image_float.transpose(2, 0, 1).copy())
        mask_tensor = torch.from_numpy(mask.astype(np.int64))
        return image_tensor, mask_tensor


def calculate_iou(
    model: SegformerForSemanticSegmentation,
    loader: DataLoader,
    device: torch.device,
    road_class_id: int,
) -> float:
    model.eval()
    intersection = 0
    union = 0
    with torch.inference_mode():
        for images, masks in loader:
            images = images.to(device)
            logits = model(pixel_values=images).logits
            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            predictions = logits.argmax(dim=1).cpu()
            truth = masks
            valid = truth != IGNORE_LABEL
            intersection += int(
                ((predictions == road_class_id) & (truth == ROAD_LABEL) & valid).sum()
            )
            union += int(
                (((predictions == road_class_id) | (truth == ROAD_LABEL)) & valid).sum()
            )
    return intersection / union if union else 1.0


def train(args: argparse.Namespace) -> Path:
    if args.license_acknowledged is not True:
        if args.dataset_format == "idd":
            raise ValueError(
                "Confirm the IDD dataset's CC BY 4.0 attribution terms with "
                "--license-acknowledged."
            )
        raise ValueError(
            "Confirm private, non-commercial use and the CC BY-NC-ND restrictions "
            "with --license-acknowledged."
        )
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("Epochs, batch size, and learning rate must be positive.")
    if args.max_train_samples < 1 or args.max_validation_samples < 1:
        raise ValueError("Training and validation sample limits must be positive.")

    output_dir = args.output_dir
    metrics_path = output_dir / "training_metrics.json"
    previous_metrics: dict[str, object] = {}
    if args.resume:
        if not metrics_path.is_file():
            raise FileNotFoundError(
                f"Cannot resume without existing training metrics: {metrics_path}"
            )
        previous_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not isinstance(previous_metrics, dict):
            raise ValueError(f"Invalid training metrics format: {metrics_path}")
        if previous_metrics.get("dataset_format") != args.dataset_format:
            raise ValueError("Cannot resume with a different dataset format.")
        if Path(str(previous_metrics.get("dataset", ""))).resolve() != args.dataset_dir.resolve():
            raise ValueError("Cannot resume with a different dataset directory.")
        if previous_metrics.get("input_size") != [args.width, args.height]:
            raise ValueError("Cannot resume with a different input size.")
        if previous_metrics.get("seed") != args.seed:
            raise ValueError("Cannot resume with a different seed.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    image_size = (args.width, args.height)
    if args.dataset_format == "idd":
        training_samples = load_idd_samples(
            args.dataset_dir,
            "train",
            image_size,
            args.max_train_samples,
            args.seed,
        )
        validation_samples = load_idd_samples(
            args.dataset_dir,
            "val",
            image_size,
            args.max_validation_samples,
            args.seed + 1,
        )
        samples = training_samples + validation_samples
        training_indices = list(range(len(training_samples)))
        validation_indices = list(range(len(training_samples), len(samples)))
    else:
        samples = load_samples(args.dataset_dir)
        permutation = np.random.default_rng(args.seed).permutation(len(samples)).tolist()
        validation_count = max(1, round(len(samples) * args.validation_fraction))
        validation_indices = sorted(permutation[:validation_count])
        training_indices = sorted(permutation[validation_count:])
        if len(training_indices) > args.max_train_samples:
            training_indices = sorted(
                random.Random(args.seed).sample(training_indices, args.max_train_samples)
            )
        if len(validation_indices) > args.max_validation_samples:
            validation_indices = sorted(
                random.Random(args.seed + 1).sample(
                    validation_indices, args.max_validation_samples
                )
            )

    if args.resume:
        start_epoch = int(previous_metrics.get("epochs_run", 0)) + 1
        if args.epochs < start_epoch:
            raise ValueError(
                f"Target epochs ({args.epochs}) must be greater than the "
                f"completed epoch count ({start_epoch - 1})."
            )
        last_model_dir = output_dir / "last"
        resume_model_dir = last_model_dir if (last_model_dir / "config.json").is_file() else output_dir
        processor = AutoImageProcessor.from_pretrained(resume_model_dir)
        model = SegformerForSemanticSegmentation.from_pretrained(resume_model_dir)
    else:
        start_epoch = 1
        processor = AutoImageProcessor.from_pretrained(args.model)
        model = SegformerForSemanticSegmentation.from_pretrained(
            args.model,
            num_labels=2,
            id2label={0: "background", ROAD_LABEL: "road"},
            label2id={"background": 0, "road": ROAD_LABEL},
            ignore_mismatched_sizes=True,
        )
    processor.size = {"height": args.height, "width": args.width}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    mean = list(processor.image_mean)
    std = list(processor.image_std)
    training_data = RoadPolygonDataset(
        samples, training_indices, image_size, mean, std, augment=True
    )
    validation_data = RoadPolygonDataset(
        samples, validation_indices, image_size, mean, std, augment=False
    )
    generator = torch.Generator().manual_seed(args.seed)
    training_loader = DataLoader(
        training_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        baseline_iou = float(previous_metrics["baseline_validation_road_iou"])
        best_iou = float(previous_metrics["best_validation_road_iou"])
        history = list(previous_metrics.get("history", []))
        best_epoch = int(previous_metrics.get("best_epoch", previous_metrics["epochs_run"]))
        source_model = str(previous_metrics.get("source_model", args.model))
        if (
            previous_metrics.get("train_images") != len(training_indices)
            or previous_metrics.get("validation_images") != len(validation_indices)
            or previous_metrics.get("validation_indices") != validation_indices
        ):
            raise ValueError("The training/validation split differs from the saved run.")
    else:
        baseline_model = SegformerForSemanticSegmentation.from_pretrained(args.model).to(device).eval()
        baseline_road_ids = [
            int(class_id)
            for class_id, label in baseline_model.config.id2label.items()
            if str(label).strip().casefold() == "road"
        ]
        if len(baseline_road_ids) != 1:
            raise RuntimeError(f"Baseline model must define exactly one road class, got {baseline_road_ids}.")
        baseline_iou = calculate_iou(
            baseline_model, validation_loader, device, baseline_road_ids[0]
        )
        print(f"baseline_validation_road_iou={baseline_iou:.4f}", flush=True)
        del baseline_model
        best_iou = -1.0
        history: list[dict[str, float]] = []
        best_epoch = 0
        source_model = args.model

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        loss_total = 0.0
        for images, masks in training_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(pixel_values=images).logits
            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            loss = F.cross_entropy(logits, masks, ignore_index=IGNORE_LABEL)
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach())

        validation_iou = calculate_iou(model, validation_loader, device, ROAD_LABEL)
        epoch_loss = loss_total / max(len(training_loader), 1)
        history.append({"epoch": epoch, "training_loss": epoch_loss, "validation_iou": validation_iou})
        print(
            f"epoch {epoch}/{args.epochs} "
            f"training_loss={epoch_loss:.4f} validation_road_iou={validation_iou:.4f}",
            flush=True,
        )
        if validation_iou > best_iou:
            best_iou = validation_iou
            best_epoch = epoch
            model.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)
        last_model_dir = output_dir / "last"
        model.save_pretrained(last_model_dir)
        processor.save_pretrained(last_model_dir)
        metrics = {
            "source_model": source_model,
            "dataset": str(args.dataset_dir),
            "dataset_format": args.dataset_format,
            "train_images": len(training_indices),
            "validation_images": len(validation_indices),
            "validation_indices": validation_indices,
            "input_size": [args.width, args.height],
            "seed": args.seed,
            "epochs_run": epoch,
            "best_epoch": best_epoch,
            "best_validation_road_iou": best_iou,
            "baseline_validation_road_iou": baseline_iou,
            "validation_iou_delta": best_iou - baseline_iou,
            "history": history,
            "license": (
                "CC BY 4.0"
                if args.dataset_format == "idd"
                else "CC BY-NC-ND 4.0; private non-commercial training only"
            ),
        }
        metrics_path.write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8",
        )

    if args.dataset_format == "idd":
        license_notice = (
            "# Fine-tuned model attribution\n\n"
            "This checkpoint was fine-tuned on the Indian Driving Dataset (IDD), "
            "using the public IDD20KII Kaggle dataset by Mansehaj Preet, which is "
            "listed under CC BY 4.0. Retain this attribution and review the original "
            "dataset and model licenses before redistributing this checkpoint.\n\n"
            "IDD citation: Girish Varma, Anbumani Subramanian, Anoop Babu, Peter "
            "Kontschieder, and C. V. Jawahar, 'IDD: A Dataset for Exploring Problems "
            "of Autonomous Navigation in Unconstrained Environments,' WACV 2019.\n"
        )
    else:
        license_notice = (
            "# Fine-tuned model license notice\n\n"
            "This checkpoint was fine-tuned using the Kaggle Road Segmentation Dataset "
            "by Unique Data (`trainingdatapro`), licensed under CC BY-NC-ND 4.0. "
            "The checkpoint is retained locally for non-commercial research only. "
            "Do not distribute this adapted checkpoint or use it commercially without "
            "permission from the dataset provider. See "
            "[the dataset license](../../dataset/road_segmentation_kaggle/DATASET_LICENSE.md).\n"
        )
    (output_dir / "LICENSE_NOTICE.md").write_text(license_notice, encoding="utf-8")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune SegFormer-B0 on road polygon annotations."
    )
    parser.add_argument(
        "--dataset-format",
        choices=("cvat", "idd"),
        default="cvat",
        help="Use the included CVAT annotations or IDD polygon annotations.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=ROOT / "dataset" / "road_segmentation_kaggle",
        help="Dataset root; for IDD, this folder contains gtFine and leftImg8bit.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "road_segformer_finetuned",
    )
    parser.add_argument("--model", default=DEFAULT_ROAD_SEGMENTATION_MODEL_ID)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--max-train-samples", type=int, default=5000)
    parser.add_argument("--max-validation-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from the output directory's latest model and metrics.",
    )
    parser.add_argument(
        "--license-acknowledged",
        action="store_true",
        help="Confirm private, non-commercial use and no distribution of adapted model weights.",
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
