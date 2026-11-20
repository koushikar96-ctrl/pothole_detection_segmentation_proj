"""Evaluate HARP-UNet and generate report-ready segmentation visualizations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harp_unet import HARPUNet
from data import paired_stems


def find_file(folder: Path, stem: str) -> Path:
    for suffix in (".jpg", ".jpeg", ".png", ".bmp"):
        candidate = folder / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No image or mask found for {stem} in {folder}")


def load_pair(data_root: Path, split: str, stem: str, image_size: int) -> tuple[np.ndarray, np.ndarray]:
    image_dir = data_root / "images" / split
    mask_dir = data_root / "masks" / split
    image = cv2.imread(str(find_file(image_dir, stem)), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(find_file(mask_dir, stem)), cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        raise ValueError(f"Could not read pair {stem}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    return image_rgb, (mask > 0).astype(np.uint8)


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    model = HARPUNet().to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=True)

    stems = paired_stems(args.data_root, args.split)
    tp = tn = fp = fn = 0
    sample_outputs: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]] = []

    with torch.no_grad():
        for index, stem in enumerate(stems):
            image, target = load_pair(args.data_root, args.split, stem, args.image_size)
            tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).float().div(255).unsqueeze(0).to(device)
            prediction = (torch.sigmoid(model(tensor))[0, 0].cpu().numpy() >= args.threshold).astype(np.uint8)
            tp += int(np.logical_and(prediction == 1, target == 1).sum())
            tn += int(np.logical_and(prediction == 0, target == 0).sum())
            fp += int(np.logical_and(prediction == 1, target == 0).sum())
            fn += int(np.logical_and(prediction == 0, target == 1).sum())
            if len(sample_outputs) < args.examples and (index % max(1, len(stems) // max(1, args.examples)) == 0):
                sample_outputs.append((image, target, prediction, stem))

    eps = 1e-8
    dice = 2 * tp / max(2 * tp + fp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    metrics = {
        "model": "HARP-UNet",
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.data_root),
        "split": args.split,
        "samples": len(stems),
        "input_size": args.image_size,
        "threshold": args.threshold,
        "true_positive_pixels": tp,
        "true_negative_pixels": tn,
        "false_positive_pixels": fp,
        "false_negative_pixels": fn,
        "dice": round(dice, 6),
        "iou": round(iou, 6),
        "precision": round(precision, 6),
        "recall_sensitivity": round(recall, 6),
        "specificity": round(specificity, 6),
        "pixel_accuracy": round(accuracy, 6),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    cm = np.array([[tn, fp], [fn, tp]], dtype=np.int64)
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.imshow(cm, cmap="Blues")
    axis.set_title("HARP-UNet pixel confusion matrix")
    axis.set_xlabel("Predicted mask pixel")
    axis.set_ylabel("Ground-truth mask pixel")
    axis.set_xticks([0, 1], ["Background", "Pothole"])
    axis.set_yticks([0, 1], ["Background", "Pothole"])
    for row in range(2):
        for column in range(2):
            axis.text(column, row, f"{cm[row, column]:,}", ha="center", va="center", fontsize=12)
    figure.tight_layout()
    figure.savefig(args.output_dir / "pixel_confusion_matrix.png", dpi=180)
    plt.close(figure)

    summary_path = args.training_summary
    if summary_path and summary_path.exists():
        history = json.loads(summary_path.read_text(encoding="utf-8")).get("history", [])
        if history:
            epochs = [item["epoch"] for item in history]
            figure, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].plot(epochs, [item["train_loss"] for item in history], label="Train loss")
            axes[0].plot(epochs, [item["validation_loss"] for item in history], label="Validation loss")
            axes[0].set_title("HARP-UNet loss curve")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].legend()
            axes[1].plot(epochs, [item["validation_dice"] for item in history], label="Dice")
            axes[1].plot(epochs, [item["validation_iou"] for item in history], label="IoU")
            axes[1].set_title("Validation segmentation metrics")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Score")
            axes[1].set_ylim(0, 1)
            axes[1].legend()
            figure.tight_layout()
            figure.savefig(args.output_dir / "training_curves.png", dpi=180)
            plt.close(figure)

    if sample_outputs:
        figure, axes = plt.subplots(len(sample_outputs), 3, figsize=(11, 4 * len(sample_outputs)))
        axes = np.atleast_2d(axes)
        for row, (image, target, prediction, stem) in enumerate(sample_outputs):
            axes[row, 0].imshow(image)
            axes[row, 0].set_title(f"Input: {stem}")
            axes[row, 1].imshow(target, cmap="gray", vmin=0, vmax=1)
            axes[row, 1].set_title("Ground-truth mask")
            axes[row, 2].imshow(prediction, cmap="gray", vmin=0, vmax=1)
            axes[row, 2].set_title("HARP-UNet prediction")
            for column in range(3):
                axes[row, column].axis("off")
        figure.tight_layout()
        figure.savefig(args.output_dir / "qualitative_predictions.png", dpi=180)
        plt.close(figure)

    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-summary", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--examples", type=int, default=6)
    parser.add_argument("--device", default="cpu")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
