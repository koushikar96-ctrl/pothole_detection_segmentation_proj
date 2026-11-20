"""Generate report-ready visualizations for the full-image HARP-UNet model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize full-image HARP-UNet metrics")
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("artifacts/harp_training/evaluation/metrics.json"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("artifacts/harp_training/training_summary.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/harp_training/evaluation"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    summary = json.loads(args.summary.read_text(encoding="utf-8"))

    metric_keys = [
        "dice",
        "iou",
        "precision",
        "recall_sensitivity",
        "specificity",
        "pixel_accuracy",
    ]
    metric_labels = [
        "Dice",
        "IoU",
        "Precision",
        "Recall",
        "Specificity",
        "Pixel accuracy",
    ]
    values = [float(metrics[key]) for key in metric_keys]

    figure, axis = plt.subplots(figsize=(10, 5.5))
    bars = axis.bar(metric_labels, values, color="#2878b5")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("Full-image HARP-UNet evaluation metrics")
    axis.tick_params(axis="x", rotation=20)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )
    figure.tight_layout()
    figure.savefig(args.output_dir / "full_image_metric_scores.png", dpi=200)
    plt.close(figure)

    history = summary.get("history", [])
    if history:
        epochs = [item["epoch"] for item in history]
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(epochs, [item["train_loss"] for item in history], marker="o", label="Train loss")
        axes[0].plot(epochs, [item["validation_loss"] for item in history], marker="o", label="Validation loss")
        axes[0].set_title("Training and validation loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[1].plot(epochs, [item["validation_dice"] for item in history], marker="o", label="Dice")
        axes[1].plot(epochs, [item["validation_iou"] for item in history], marker="o", label="IoU")
        axes[1].set_title("Validation segmentation metrics")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Score")
        axes[1].set_ylim(0, 1)
        axes[1].legend()
        figure.tight_layout()
        figure.savefig(args.output_dir / "full_image_training_curves.png", dpi=200)
        plt.close(figure)

    report = {
        "model": "Full-image HARP-UNet",
        "metrics": metrics,
        "training": {
            "epochs": summary.get("epochs"),
            "image_size": summary.get("image_size"),
            "batch_size": summary.get("batch_size"),
            "best_dice": summary.get("best_dice"),
        },
        "visualizations": [
            "full_image_metric_scores.png",
            "full_image_training_curves.png",
            "pixel_confusion_matrix.png",
            "qualitative_predictions.png",
        ],
    }
    (args.output_dir / "full_image_harp_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
