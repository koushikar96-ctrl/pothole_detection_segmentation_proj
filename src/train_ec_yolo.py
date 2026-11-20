from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ec_yolo import ECYOLODetector


class YOLODataset(Dataset):
    def __init__(self, data_root: Path, split: str = "train", image_size: int = 640, training: bool = False) -> None:
        self.image_dir = Path(data_root) / "images" / split
        self.label_dir = Path(data_root) / "labels" / split
        self.images = sorted(path for path in self.image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
        self.image_size = image_size
        self.training = training

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.images[index]
        image = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        labels = []
        label_path = self.label_dir / f"{path.stem}.txt"
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                values = line.split()
                if len(values) == 5:
                    labels.append([float(value) for value in values[1:]])
        boxes = np.asarray(labels, dtype=np.float32).reshape(-1, 4)
        if self.training and random.random() < 0.5:
            image = image[:, ::-1].copy()
            if len(boxes):
                boxes[:, 0] = 1.0 - boxes[:, 0]
        return torch.from_numpy(image.transpose(2, 0, 1).copy()).float() / 255.0, torch.from_numpy(boxes)


def collate_batch(batch):
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)


def build_targets(targets: list[torch.Tensor], output_shapes: dict[str, torch.Size], device: torch.device) -> dict[str, torch.Tensor]:
    result = {name: torch.zeros((len(targets), 6, shape[-2], shape[-1]), device=device) for name, shape in output_shapes.items()}
    for batch_index, boxes in enumerate(targets):
        for x, y, width, height in boxes.tolist():
            scale_name = "p3" if max(width, height) < 0.15 else "p4" if max(width, height) < 0.35 else "p5"
            target = result[scale_name]
            grid_height, grid_width = target.shape[-2:]
            grid_x = min(grid_width - 1, int(x * grid_width))
            grid_y = min(grid_height - 1, int(y * grid_height))
            target[batch_index, 0, grid_y, grid_x] = x * grid_width - grid_x
            target[batch_index, 1, grid_y, grid_x] = y * grid_height - grid_y
            target[batch_index, 2, grid_y, grid_x] = width
            target[batch_index, 3, grid_y, grid_x] = height
            target[batch_index, 4, grid_y, grid_x] = 1.0
            target[batch_index, 5, grid_y, grid_x] = 1.0
    return result


def detection_loss(predictions: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> torch.Tensor:
    total = predictions["p3"].new_zeros(())
    for name, prediction in predictions.items():
        target = targets[name]
        positive = target[:, 4:5] > 0
        total = total + nn.functional.binary_cross_entropy_with_logits(prediction[:, 4:6], target[:, 4:6])
        if positive.any():
            total = total + nn.functional.smooth_l1_loss(
                torch.sigmoid(prediction[:, :2][positive.expand_as(prediction[:, :2])]),
                target[:, :2][positive.expand_as(prediction[:, :2])],
            )
            total = total + nn.functional.smooth_l1_loss(
                torch.sigmoid(prediction[:, 2:4][positive.expand_as(prediction[:, :2])]),
                target[:, 2:4][positive.expand_as(prediction[:, :2])],
            )
    return total


def train_ec_yolo(
    data_root: Path,
    output_dir: Path,
    epochs: int = 50,
    batch_size: int = 4,
    image_size: int = 640,
    learning_rate: float = 1e-3,
    resume: bool = False,
) -> Dict[str, object]:
    """Train EC-YOLO on the prepared pothole dataset.

    Parameters
    ----------
    data_root:
        Root of the prepared dataset.
    output_dir:
        Destination for ``best.pt``, ``last.pt``, and ``training_summary.json``.
    epochs:
        Total training epochs (including already-completed ones when resuming).
    batch_size:
        Mini-batch size.
    image_size:
        Square detector input resolution.
    learning_rate:
        Initial AdamW learning rate.
    resume:
        If *True*, load ``output_dir/last.pt`` and continue from the saved epoch.
    """
    random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = YOLODataset(data_root, "train", image_size, True)
    validation_data = YOLODataset(data_root, "val", image_size, False)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_batch, num_workers=0)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False, collate_fn=collate_batch, num_workers=0)

    model = ECYOLODetector().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ resume
    start_epoch = 1
    best_validation_loss = float("inf")
    history: list[dict] = []

    last_ckpt = output_dir / "last.pt"
    if resume and last_ckpt.exists():
        print(f"[resume] Loading checkpoint from {last_ckpt}")
        ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_validation_loss = ckpt.get("best_validation_loss", float("inf"))
        history = ckpt.get("history", [])
        print(f"[resume] Resuming from epoch {start_epoch} (best val loss so far: {best_validation_loss:.5f})")
    elif resume:
        print(f"[resume] No checkpoint found at {last_ckpt} — starting from scratch.")
    # -----------------------------------------------------------------------

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = 0.0
        for images, targets in train_loader:
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(images)
            target_maps = build_targets(targets, {name: value.shape for name, value in predictions.items()}, device)
            loss = detection_loss(predictions, target_maps)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for images, targets in validation_loader:
                images = images.to(device)
                predictions = model(images)
                target_maps = build_targets(targets, {name: value.shape for name, value in predictions.items()}, device)
                validation_loss += detection_loss(predictions, target_maps).item() * images.size(0)

        record = {
            "epoch": epoch,
            "train_loss": train_loss / max(1, len(train_data)),
            "validation_loss": validation_loss / max(1, len(validation_data)),
        }
        history.append(record)
        print(record)

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_validation_loss": best_validation_loss,
            "history": history,
            "metrics": record,
        }
        torch.save(checkpoint, output_dir / "last.pt")

        if record["validation_loss"] < best_validation_loss:
            best_validation_loss = record["validation_loss"]
            checkpoint["best_validation_loss"] = best_validation_loss
            torch.save(checkpoint, output_dir / "best.pt")
            print(f"  [BEST] New best val loss: {best_validation_loss:.5f} - saved best.pt")

    summary = {
        "model": "EC-YOLO",
        "device": str(device),
        "train_samples": len(train_data),
        "validation_samples": len(validation_data),
        "epochs": epochs,
        "image_size": image_size,
        "best_validation_loss": best_validation_loss,
        "best_checkpoint": str(output_dir / "best.pt"),
        "history": history,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary