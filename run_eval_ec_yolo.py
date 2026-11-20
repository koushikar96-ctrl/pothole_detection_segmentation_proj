"""EC-YOLO Evaluation Script
============================
Computes Precision, Recall, F1, mAP50, and mAP50-95 on a dataset split
using IoU-based matching between predicted and ground-truth bounding boxes.

Usage
-----
# Evaluate best checkpoint on the test split:
python run_eval_ec_yolo.py \
    --checkpoint artifacts/ec_yolo_training/best.pt \
    --split test

# Quick smoke-test with random weights (no checkpoint needed):
python run_eval_ec_yolo.py --split test --no-checkpoint

Output
------
Prints a results table to stdout and saves
``artifacts/ec_yolo_eval/eval_results.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ec_yolo import ECYOLODetector, decode_predictions


# ---------------------------------------------------------------------------
# IoU helpers
# ---------------------------------------------------------------------------

def box_iou_single(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute IoU between one predicted box and one GT box (xyxy absolute pixels)."""
    ix1 = max(pred[0], gt[0])
    iy1 = max(pred[1], gt[1])
    ix2 = min(pred[2], gt[2])
    iy2 = min(pred[3], gt[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_pred = (pred[2] - pred[0]) * (pred[3] - pred[1])
    area_gt = (gt[2] - gt[0]) * (gt[3] - gt[1])
    union = area_pred + area_gt - inter
    return inter / union if union > 0 else 0.0


def compute_iou_matrix(preds: np.ndarray, gts: np.ndarray) -> np.ndarray:
    """Return an (N_pred, N_gt) IoU matrix."""
    if len(preds) == 0 or len(gts) == 0:
        return np.zeros((len(preds), len(gts)), dtype=np.float32)
    matrix = np.zeros((len(preds), len(gts)), dtype=np.float32)
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            matrix[i, j] = box_iou_single(p, g)
    return matrix


# ---------------------------------------------------------------------------
# AP computation
# ---------------------------------------------------------------------------

def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute average precision using the 101-point interpolation (COCO style)."""
    ap = 0.0
    for thr in np.linspace(0.0, 1.0, 101):
        prec_at_rec = precisions[recalls >= thr]
        ap += prec_at_rec.max() if prec_at_rec.size > 0 else 0.0
    return ap / 101.0


def eval_at_iou(all_predictions: list[tuple[np.ndarray, np.ndarray]], all_gts: list[np.ndarray], iou_threshold: float) -> dict[str, float]:
    """Evaluate precision / recall / AP for a single IoU threshold.

    Parameters
    ----------
    all_predictions:
        List of (boxes_xyxy, confidences) per image.  Boxes are absolute pixels.
    all_gts:
        List of GT boxes (xyxy absolute pixels) per image.
    iou_threshold:
        IoU threshold for a true-positive match.
    """
    # Flatten all predictions with image index and sort by descending confidence
    records: list[tuple[float, int, np.ndarray]] = []  # (conf, image_idx, box)
    for img_idx, (boxes, confs) in enumerate(all_predictions):
        for box, conf in zip(boxes, confs):
            records.append((float(conf), img_idx, box))
    records.sort(key=lambda r: -r[0])

    total_gt = sum(len(g) for g in all_gts)
    matched: list[set[int]] = [set() for _ in all_gts]  # matched GT indices per image

    tp_list: list[int] = []
    fp_list: list[int] = []

    for conf, img_idx, pred_box in records:
        gts = all_gts[img_idx]
        if len(gts) == 0:
            tp_list.append(0)
            fp_list.append(1)
            continue
        ious = np.array([box_iou_single(pred_box, g) for g in gts])
        best_j = int(ious.argmax())
        if ious[best_j] >= iou_threshold and best_j not in matched[img_idx]:
            matched[img_idx].add(best_j)
            tp_list.append(1)
            fp_list.append(0)
        else:
            tp_list.append(0)
            fp_list.append(1)

    tp_cum = np.cumsum(tp_list).astype(float)
    fp_cum = np.cumsum(fp_list).astype(float)
    recalls = tp_cum / max(total_gt, 1)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

    ap = compute_ap(recalls, precisions)
    final_recall = float(recalls[-1]) if len(recalls) else 0.0
    final_precision = float(precisions[-1]) if len(precisions) else 0.0
    f1 = 2 * final_precision * final_recall / max(final_precision + final_recall, 1e-9)

    return {
        "iou_threshold": iou_threshold,
        "precision": round(final_precision, 4),
        "recall": round(final_recall, 4),
        "f1": round(f1, 4),
        "ap": round(ap, 4),
        "total_predictions": len(records),
        "total_gt": total_gt,
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_eval(
    data_root: Path,
    output_dir: Path,
    checkpoint: Path | None,
    split: str = "test",
    conf_threshold: float = 0.25,
    image_size: int = 640,
    device_str: str | None = None,
) -> dict:
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    model = ECYOLODetector().to(device).eval()
    if checkpoint and checkpoint.exists():
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
        model.load_state_dict(state)
        print(f"Loaded checkpoint: {checkpoint}")
    else:
        print("WARNING: Running with random (untrained) weights — results are meaningless.")

    image_dir = data_root / "images" / split
    label_dir = data_root / "labels" / split
    image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")
    print(f"Evaluating {len(image_paths)} images from '{split}' split …")

    all_predictions: list[tuple[np.ndarray, np.ndarray]] = []
    all_gts: list[np.ndarray] = []

    for img_path in image_paths:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            all_predictions.append((np.empty((0, 4)), np.empty((0,))))
            all_gts.append(np.empty((0, 4)))
            continue

        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized.transpose(2, 0, 1).copy()).float().div(255).unsqueeze(0).to(device)

        with torch.no_grad():
            raw_preds = decode_predictions(model(tensor), threshold=conf_threshold)[0]

        # Convert normalised [0,image_size] coords back to original pixel space
        pred_boxes = []
        pred_confs = []
        for x1, y1, x2, y2, conf in raw_preds:
            pred_boxes.append([x1 * w / image_size, y1 * h / image_size, x2 * w / image_size, y2 * h / image_size])
            pred_confs.append(conf)

        all_predictions.append((np.array(pred_boxes, dtype=np.float32).reshape(-1, 4),
                                 np.array(pred_confs, dtype=np.float32)))

        # Load YOLO labels (cx, cy, bw, bh normalised) → xyxy absolute pixels
        gt_boxes = []
        label_path = label_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    cx, cy, bw, bh = (float(v) for v in parts[1:])
                    gt_boxes.append([
                        (cx - bw / 2) * w,
                        (cy - bh / 2) * h,
                        (cx + bw / 2) * w,
                        (cy + bh / 2) * h,
                    ])
        all_gts.append(np.array(gt_boxes, dtype=np.float32).reshape(-1, 4))

    # ---------------------------------------------------------------- metrics
    iou_thresholds = np.arange(0.50, 1.00, 0.05).round(2).tolist()
    per_iou: list[dict] = []
    for thr in iou_thresholds:
        per_iou.append(eval_at_iou(all_predictions, all_gts, thr))

    metrics_50 = next(m for m in per_iou if abs(m["iou_threshold"] - 0.50) < 1e-6)
    map50 = metrics_50["ap"]
    map50_95 = float(np.mean([m["ap"] for m in per_iou]))

    summary = {
        "split": split,
        "checkpoint": str(checkpoint) if checkpoint else "none",
        "conf_threshold": conf_threshold,
        "num_images": len(image_paths),
        "precision_at_50": metrics_50["precision"],
        "recall_at_50": metrics_50["recall"],
        "f1_at_50": metrics_50["f1"],
        "mAP50": round(map50, 4),
        "mAP50_95": round(map50_95, 4),
        "per_iou_threshold": per_iou,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "eval_results.json"
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ----------------------------------------------------------------- print
    print("\n" + "=" * 55)
    print(f"  EC-YOLO Evaluation — split: {split}")
    print("=" * 55)
    print(f"  Images evaluated : {len(image_paths)}")
    print(f"  Conf threshold   : {conf_threshold}")
    print(f"  Precision @0.50  : {metrics_50['precision']:.4f}")
    print(f"  Recall    @0.50  : {metrics_50['recall']:.4f}")
    print(f"  F1        @0.50  : {metrics_50['f1']:.4f}")
    print(f"  mAP50            : {map50:.4f}")
    print(f"  mAP50-95         : {map50_95:.4f}")
    print("=" * 55)
    print(f"\nFull results saved to: {results_path}\n")
    return summary


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate EC-YOLO: precision, recall, mAP50, mAP50-95")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts" / "ec_yolo_training" / "best.pt",
                        help="Path to EC-YOLO checkpoint (default: artifacts/ec_yolo_training/best.pt)")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="Skip checkpoint loading (random weights — for smoke testing only)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                        help="Dataset split to evaluate on (default: test)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold for predicted detections (default: 0.25)")
    parser.add_argument("--image-size", type=int, default=640,
                        help="Detector input resolution (default: 640)")
    parser.add_argument("--data-root", type=Path, default=ROOT / "prepared_dataset",
                        help="Prepared dataset root (default: prepared_dataset/)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "ec_yolo_eval",
                        help="Output directory for eval_results.json")
    args = parser.parse_args()

    checkpoint = None if args.no_checkpoint else args.checkpoint
    run_eval(
        data_root=args.data_root,
        output_dir=args.output_dir,
        checkpoint=checkpoint,
        split=args.split,
        conf_threshold=args.conf,
        image_size=args.image_size,
    )
