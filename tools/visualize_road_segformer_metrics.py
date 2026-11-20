"""Plot training loss and validation IoU from a SegFormer metrics JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize road SegFormer training metrics"
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("artifacts/road_segformer_idd20k/training_metrics.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/road_segformer_idd20k/training_curves.png"),
    )
    args = parser.parse_args()

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    history = metrics.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError(f"No training history found in {args.metrics}")

    epochs = [int(item["epoch"]) for item in history]
    losses = [float(item["training_loss"]) for item in history]
    validation_ious = [float(item["validation_iou"]) for item in history]
    baseline_iou = float(metrics["baseline_validation_road_iou"])

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(epochs, losses, color="#2878b5", marker="o", markersize=3)
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        epochs,
        validation_ious,
        color="#2a9d8f",
        marker="o",
        markersize=3,
        label="Validation road IoU",
    )
    axes[1].axhline(
        baseline_iou,
        color="#e76f51",
        linestyle="--",
        label=f"Cityscapes baseline ({baseline_iou:.3f})",
    )
    axes[1].set_title("Validation road IoU")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("IoU")
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    figure.suptitle(
        f"IDD20KII SegFormer training metrics ({metrics['epochs_run']} epochs)"
    )
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved training curves to {args.output}")


if __name__ == "__main__":
    main()
