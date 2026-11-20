from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data import PotholeSegmentationDataset, paired_stems
from harp_unet import HARPUNet


def dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * targets).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def combined_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return nn.functional.binary_cross_entropy_with_logits(logits, targets) + dice_loss(logits, targets)


def segmentation_metrics(logits: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
    predictions = (torch.sigmoid(logits) >= 0.5).float()
    intersection = (predictions * targets).sum(dim=(1, 2, 3))
    prediction_area = predictions.sum(dim=(1, 2, 3))
    target_area = targets.sum(dim=(1, 2, 3))
    dice = ((2.0 * intersection + 1.0) / (prediction_area + target_area + 1.0)).mean().item()
    union = prediction_area + target_area - intersection
    iou = ((intersection + 1.0) / (union + 1.0)).mean().item()
    return dice, iou


def train_harp(
    data_root: Path,
    output_dir: Path,
    epochs: int = 30,
    image_size: int = 256,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    seed: int = 42,
    device: str | None = None,
    resume: bool = False,
) -> Dict[str, object]:
    """Train HARP-UNet on the prepared pothole dataset.

    Parameters
    ----------
    data_root:
        Root of the prepared dataset (contains ``images/`` and ``masks/`` splits).
    output_dir:
        Directory where ``best.pt``, ``last.pt``, and ``training_summary.json``
        are written.
    epochs:
        Total number of epochs to train (including already-completed ones when
        resuming).
    image_size:
        Square input resolution fed to the model.
    batch_size:
        Samples per mini-batch.
    learning_rate:
        Initial AdamW learning rate.
    seed:
        Random seed for reproducibility.
    device:
        Torch device override, such as ``cpu`` or ``cuda``. If omitted, use
        CUDA when available and CPU otherwise.
    resume:
        If *True*, load ``output_dir/last.pt`` and continue training from the
        saved epoch.  The optimizer state and best-dice record are restored so
        the run is fully seamless.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_stems = paired_stems(data_root, "train")
    validation_stems = paired_stems(data_root, "val")
    if not train_stems or not validation_stems:
        raise ValueError("Prepared train and validation pairs are required.")

    train_data = PotholeSegmentationDataset(data_root, train_stems, image_size, training=True, split="train")
    validation_data = PotholeSegmentationDataset(data_root, validation_stems, image_size, training=False, split="val")
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False, num_workers=0)

    training_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = HARPUNet().to(training_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ resume
    start_epoch = 1
    best_dice = -1.0
    history: list[dict] = []

    last_ckpt = output_dir / "last.pt"
    if resume and last_ckpt.exists():
        print(f"[resume] Loading checkpoint from {last_ckpt}")
        ckpt = torch.load(last_ckpt, map_location=training_device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_dice = ckpt.get("best_dice", -1.0)
        history = ckpt.get("history", [])
        print(f"[resume] Resuming from epoch {start_epoch} (best Dice so far: {best_dice:.4f})")
    elif resume:
        print(f"[resume] No checkpoint found at {last_ckpt} — starting from scratch.")
    # -----------------------------------------------------------------------

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_total = 0.0
        for images, masks in train_loader:
            images, masks = images.to(training_device), masks.to(training_device)
            optimizer.zero_grad(set_to_none=True)
            loss = combined_loss(model(images), masks)
            loss.backward()
            optimizer.step()
            train_total += loss.item() * images.size(0)

        model.eval()
        validation_total = 0.0
        dice_total = 0.0
        iou_total = 0.0
        with torch.no_grad():
            for images, masks in validation_loader:
                images, masks = images.to(training_device), masks.to(training_device)
                logits = model(images)
                validation_total += combined_loss(logits, masks).item() * images.size(0)
                dice, iou = segmentation_metrics(logits, masks)
                dice_total += dice * images.size(0)
                iou_total += iou * images.size(0)

        record = {
            "epoch": epoch,
            "train_loss": train_total / len(train_data),
            "validation_loss": validation_total / len(validation_data),
            "validation_dice": dice_total / len(validation_data),
            "validation_iou": iou_total / len(validation_data),
        }
        history.append(record)
        print(record)

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_dice": best_dice,
            "history": history,
            "metrics": record,
        }
        torch.save(checkpoint, output_dir / "last.pt")

        if record["validation_dice"] > best_dice:
            best_dice = record["validation_dice"]
            checkpoint["best_dice"] = best_dice
            torch.save(checkpoint, output_dir / "best.pt")
            print(f"  [BEST] New best Dice: {best_dice:.4f} - saved best.pt")

    summary = {
        "model": "HARP-UNet",
        "data_root": str(data_root),
        "device": str(training_device),
        "train_samples": len(train_data),
        "validation_samples": len(validation_data),
        "epochs": epochs,
        "image_size": image_size,
        "batch_size": batch_size,
        "best_dice": best_dice,
        "history": history,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary