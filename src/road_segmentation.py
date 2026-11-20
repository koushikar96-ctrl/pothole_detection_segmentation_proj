from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_ROAD_SEGMENTATION_MODEL_ID = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
IDD_ROAD_SEGMENTATION_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "artifacts" / "road_segformer_idd20k"
)
FINETUNED_ROAD_SEGMENTATION_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "artifacts" / "road_segformer_finetuned"
)
LOCAL_ROAD_SEGMENTATION_MODEL_PATH = next(
    (
        model_path
        for model_path in (
            IDD_ROAD_SEGMENTATION_MODEL_PATH,
            FINETUNED_ROAD_SEGMENTATION_MODEL_PATH,
        )
        if (model_path / "config.json").is_file()
    ),
    None,
)
DEFAULT_ROAD_SEGMENTATION_MODEL = str(
    LOCAL_ROAD_SEGMENTATION_MODEL_PATH or DEFAULT_ROAD_SEGMENTATION_MODEL_ID
)


class RoadSegmenter:
    """Predict a road-surface mask using a pretrained semantic-segmentation model."""

    def __init__(self, model_id: str, device: torch.device) -> None:
        self.model_id = model_id
        self.device = device
        self.processor: Any = None
        self.model: Any = None
        self.road_class_id: int | None = None

    def _load_model(self) -> None:
        try:
            from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for road segmentation. Install it with `pip install transformers`."
            ) from exc

        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = SegformerForSemanticSegmentation.from_pretrained(self.model_id).to(self.device).eval()

        labels = self.model.config.id2label
        road_ids = [
            int(class_id)
            for class_id, label in labels.items()
            if str(label).strip().casefold() == "road"
        ]
        if len(road_ids) != 1:
            raise RuntimeError(
                f"Road segmentation model {self.model_id!r} must define exactly one 'road' class; "
                f"found {len(road_ids)}."
            )
        self.road_class_id = road_ids[0]

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return a uint8 road mask at the input image resolution."""
        if self.model is None or self.processor is None:
            self._load_model()

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=image_rgb, return_tensors="pt")
        model_inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}

        height, width = image_bgr.shape[:2]
        scale = min(1.0, 1024.0 / max(height, width))
        output_size = (max(1, round(height * scale)), max(1, round(width * scale)))
        with torch.inference_mode():
            logits = self.model(**model_inputs).logits
            logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
            labels = logits[0].argmax(dim=0).cpu().numpy().astype(np.uint8)

        road_mask = (labels == self.road_class_id).astype(np.uint8)
        if road_mask.shape != (height, width):
            road_mask = cv2.resize(road_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        return road_mask
