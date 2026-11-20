from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

try:
    from .area_health import estimate_area, health_assessment
    from .ec_yolo import ECYOLODetector, decode_predictions
    from .harp_unet import HARPUNet
    from .roi import extract_roi, paste_mask
    from .road_segmentation import RoadSegmenter
except ImportError:
    from area_health import estimate_area, health_assessment
    from ec_yolo import ECYOLODetector, decode_predictions
    from harp_unet import HARPUNet
    from roi import extract_roi, paste_mask
    from road_segmentation import RoadSegmenter


class PotholePipeline:
    """End-to-end pothole detection and segmentation pipeline.

    Parameters
    ----------
    detection_checkpoint:
        Path to an EC-YOLO ``*.pt`` checkpoint.  If *None* or the path does
        not exist the detector runs with random weights (useful for smoke tests).
    segmentation_checkpoint:
        Path to a HARP-UNet ``*.pt`` checkpoint.  Same semantics as above.
    device:
        PyTorch device string.  Defaults to CUDA if available, else CPU.
    threshold:
        Confidence threshold applied to both detection decoding and mask
        binarisation.
    """

    @staticmethod
    def _normalize_checkpoint_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Strip training wrappers and normalize the most common checkpoint naming differences."""
        if not isinstance(state, dict):
            return {}

        normalized: dict[str, torch.Tensor] = {}
        for key, value in state.items():
            if not isinstance(key, str):
                continue
            cleaned = key.replace("module.", "").replace("model.", "")
            if cleaned.startswith("backbone.") and cleaned.startswith("backbone."):
                cleaned = "backbone_neck." + cleaned[len("backbone."):]
            if cleaned.startswith("backbone_neck.") and cleaned.startswith("backbone_neck."):
                cleaned = cleaned.replace("backbone_neck.", "backbone_neck.", 1)
            normalized[cleaned] = value
        return normalized

    def _load_weights_compatibly(self, model: torch.nn.Module, checkpoint_path: Path | str) -> None:
        """Load only compatible keys from a checkpoint and ignore wrapper differences."""
        if not checkpoint_path or not Path(checkpoint_path).exists():
            return

        ckpt = torch.load(Path(checkpoint_path), map_location=self.device, weights_only=False)
        if isinstance(ckpt, dict):
            state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
        else:
            state = ckpt

        if not isinstance(state, dict):
            return

        state = self._normalize_checkpoint_state(state)
        model_state = model.state_dict()
        compatible = {key: value for key, value in state.items() if key in model_state}

        if not compatible:
            return

        missing = [key for key in model_state if key not in compatible]
        unexpected = [key for key in state if key not in model_state]
        if missing:
            print(f"[warning] Partial checkpoint load for {checkpoint_path}: {len(missing)} weights were not matched; loaded {len(compatible)} compatible parameters.")
        if unexpected:
            print(f"[warning] Ignoring {len(unexpected)} incompatible keys from {checkpoint_path}.")

        model.load_state_dict(compatible, strict=False)

    _DETECTOR_INPUT = 640
    _FULL_IMAGE_SEGMENTER_INPUT = 256
    _ROI_SEGMENTER_INPUT = 128

    def __init__(
        self,
        detection_checkpoint: Path | None = None,
        segmentation_checkpoint: Path | None = None,
        device: str | None = None,
        threshold: float = 0.5,
        segmentation_threshold: float = 0.5,
        detection_backend: str = "custom",
        detection_model_id: str | None = None,
        segmentation_mode: str = "full_image",
        road_segmentation_model_id: str | None = None,
    ) -> None:
        if segmentation_mode not in {"full_image", "roi"}:
            raise ValueError("segmentation_mode must be 'full_image' or 'roi'.")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.threshold = threshold
        self.segmentation_threshold = segmentation_threshold
        self.segmentation_mode = segmentation_mode
        self.road_segmenter = (
            RoadSegmenter(road_segmentation_model_id, self.device)
            if road_segmentation_model_id
            else None
        )
        self._road_segmentation_fallback_reason: str | None = None
        self.detection_backend = detection_backend.lower()
        self.detection_model_id = detection_model_id or (str(detection_checkpoint) if detection_checkpoint and not Path(detection_checkpoint).exists() else None)

        self.detector = None
        self.yolo_detector = None

        self.segmenter = HARPUNet().to(self.device).eval()
        if segmentation_checkpoint and Path(segmentation_checkpoint).exists() and Path(segmentation_checkpoint).is_file():
            self._load_weights_compatibly(self.segmenter, segmentation_checkpoint)

        if self.detection_backend == "ecyolo":
            self.load_detector(self.detection_model_id or str(detection_checkpoint or "yolov8s.pt"))
        else:
            self.detector = ECYOLODetector().to(self.device).eval()
            if detection_checkpoint and Path(detection_checkpoint).exists() and Path(detection_checkpoint).is_file():
                self._load_weights_compatibly(self.detector, detection_checkpoint)

    def load_detector(self, model_id: str | None) -> None:
        
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is required.") from exc
        self.yolo_detector = YOLO(model_id or "yolov8s.pt")

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(self, image: np.ndarray) -> list[tuple[float, float, float, float, float]]:
        """Run the configured detector on *image* and return detections in original image coordinates.

        Returns
        -------
        list of ``(x1, y1, x2, y2, confidence)`` in absolute pixel coordinates.
        """
        return self.detect_stages(image)[2]

    def ecyolo_detect(self, image: np.ndarray) -> list[tuple[float, float, float, float, float]]:
        if self.yolo_detector is None:
            return []
        results = self.yolo_detector(
            image,
            conf=self.threshold,
            iou=0.45,
            verbose=False,
        )
        detections: list[tuple[float, float, float, float, float]] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                conf = float(box.conf[0].item())
                detections.append((x1, y1, x2, y2, conf))
        return detections

    def reset_video_tracker(self) -> None:
        
        if self.yolo_detector is None:
            return
        predictor = getattr(self.yolo_detector, "predictor", None)
        if predictor is None:
            return
        for tracker in getattr(predictor, "trackers", []):
            tracker.reset()
        if hasattr(predictor, "vid_path"):
            predictor.vid_path = [None] * len(predictor.vid_path)

    def track_video(
        self,
        image: np.ndarray,
    ) -> tuple[list[tuple[float, float, float, float, float]], list[int | None]]:
        """Detect and track potholes in one frame, persisting identities across calls."""
        if self.detection_backend != "ecyolo" or self.yolo_detector is None:
            raise RuntimeError("Persistent video tracking requires the YOLOv8 detection backend.")
        results = self.yolo_detector.track(
            image,
            conf=self.threshold,
            iou=0.45,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )
        detections: list[tuple[float, float, float, float, float]] = []
        track_ids: list[int | None] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            ids = getattr(boxes, "id", None)
            for index, box in enumerate(boxes):
                x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
                confidence = float(box.conf[0].item())
                track_id = int(ids[index].item()) if ids is not None else None
                detections.append((x1, y1, x2, y2, confidence))
                track_ids.append(track_id)
        return detections, track_ids

    def detect_stages(
        self,
        image: np.ndarray,
    ) -> tuple[
        np.ndarray,
        list[tuple[float, float, float, float, float]],
        list[tuple[float, float, float, float, float]],
    ]:
        """Run the configured detector on the original image and return detections in image coordinates."""
        if self.detection_backend == "ecyolo":
            detections = self.ecyolo_detect(image)
            return image, detections, detections

        s = self._DETECTOR_INPUT
        resized = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (s, s), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized.transpose(2, 0, 1).copy()).float().div(255).unsqueeze(0).to(self.device)
        with torch.no_grad():
            detections = decode_predictions(self.detector(tensor), threshold=self.threshold)[0]
        h, w = image.shape[:2]
        scaled = [(x1 * w / s, y1 * h / s, x2 * w / s, y2 * h / s, conf) for x1, y1, x2, y2, conf in detections]
        return image, scaled, scaled

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------

    def segment_image(self, image: np.ndarray) -> np.ndarray:
        """Segment a full image using the same square resize used in training."""
        original_height, original_width = image.shape[:2]
        resized = cv2.resize(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            (self._FULL_IMAGE_SEGMENTER_INPUT, self._FULL_IMAGE_SEGMENTER_INPUT),
            interpolation=cv2.INTER_LINEAR,
        )
        tensor = torch.from_numpy(resized.transpose(2, 0, 1).copy()).float().div(255).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probabilities = torch.sigmoid(self.segmenter(tensor))[0, 0].cpu().numpy()
        mask = (probabilities >= self.segmentation_threshold).astype(np.uint8)
        return cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

    def segment_roi(self, roi_image: np.ndarray) -> np.ndarray:
        """Segment a single ROI crop and return a binary mask at ROI resolution.

        The ROI is resized to the same square input used during ROI training.
        The predicted mask is resized back to the input *roi_image* dimensions.
        """
        orig_h, orig_w = roi_image.shape[:2]
        size = self._ROI_SEGMENTER_INPUT
        rgb = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized.transpose(2, 0, 1).copy()).float().div(255).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.sigmoid(self.segmenter(tensor))[0, 0].cpu().numpy()
        mask = (probs >= self.segmentation_threshold).astype(np.uint8)
        return cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    def segment_road(self, image: np.ndarray) -> np.ndarray:
        """Predict road pixels; only road-model failures use a full-frame fallback.

        Pothole segmentation is handled separately by ``segment_roi`` and
        ``segment_image`` and does not use this fallback.
        """
        road_segmenter = getattr(self, "road_segmenter", None)
        try:
            if road_segmenter is None:
                raise RuntimeError("Road segmentation is not configured.")
            road_mask = road_segmenter.predict(image)
            if road_mask.shape != image.shape[:2]:
                raise ValueError("Road segmentation mask dimensions must match the input image.")
            if not np.any(road_mask):
                raise ValueError("Road segmentation returned an empty mask.")
        except Exception as exc:
            self._road_segmentation_fallback_reason = f"{type(exc).__name__}: {exc}"
            return np.ones(image.shape[:2], dtype=np.uint8)

        self._road_segmentation_fallback_reason = None
        return (road_mask > 0).astype(np.uint8)

    # ------------------------------------------------------------------
    # Detection processing
    # ------------------------------------------------------------------

    def process_detections(
        self,
        image: np.ndarray,
        detections: list[tuple[float, float, float, float, float]],
        meters_per_pixel: float | None = None,
        road_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[dict[str, object]], dict[str, object]]:
        """Segment detected regions and compute per-detection metrics.

        ``segmentation_mode="roi"`` predicts each padded detection crop independently.
        ``segmentation_mode="full_image"`` predicts once on the full frame and gates
        that mask by the same detection crops.

        Returns
        -------
        total_mask:
            Binary union mask at full image resolution.
        results:
            Per-detection dict with ``box``, ``confidence``, ``pixel_area``,
            ``area_m2``, ``equivalent_diameter_pixels``, ``mask``.
        health:
            Road health assessment dict.
        """
        masks: list[np.ndarray] = []
        results: list[dict[str, object]] = []
        full_image_mask = (
            self.segment_image(image)
            if detections and self.segmentation_mode == "full_image"
            else None
        )

        for x1, y1, x2, y2, confidence in detections:
            roi = extract_roi(image, (x1, y1, x2, y2), confidence)
            roi_x1, roi_y1, roi_x2, roi_y2 = roi.box
            if self.segmentation_mode == "roi":
                roi_mask = self.segment_roi(roi.image)
                full_mask = paste_mask(roi_mask, roi, image.shape[:2])
            else:
                assert full_image_mask is not None
                full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                full_mask[roi_y1:roi_y2, roi_x1:roi_x2] = full_image_mask[roi_y1:roi_y2, roi_x1:roi_x2]
            area = estimate_area(full_mask, meters_per_pixel)
            masks.append(full_mask)
            results.append({
                "box": roi.box,
                "confidence": confidence,
                "pixel_area": area.pixel_area,
                "area_m2": area.area_m2,
                "equivalent_diameter_pixels": area.equivalent_diameter_pixels,
                "mask": full_mask,
            })

        total_mask = np.maximum.reduce(masks) if masks else np.zeros(image.shape[:2], dtype=np.uint8)
        total_pothole_area_m2 = estimate_area(total_mask, meters_per_pixel).area_m2
        if road_mask is None and getattr(self, "road_segmenter", None) is not None:
            road_mask = self.segment_road(image)
        road_area_pixels = None
        damaged_area_pixels = int(np.count_nonzero(total_mask))
        if road_mask is not None:
            if road_mask.shape != image.shape[:2]:
                raise ValueError("Road segmentation mask dimensions must match the input image.")
            road_binary = road_mask > 0
            pothole_binary = total_mask > 0
            predicted_road_area_pixels = int(np.count_nonzero(road_binary))
            road_mask_damage_overlap = int(np.count_nonzero(pothole_binary & road_binary))
            road_area_pixels = int(np.count_nonzero(road_binary | pothole_binary))
            if road_area_pixels == 0:
                raise ValueError("Road segmentation produced an empty road mask; cannot calculate RDP.")
            road_mask_overlap_percent = (
                100.0 * road_mask_damage_overlap / damaged_area_pixels
                if damaged_area_pixels
                else 100.0
            )
            for result in results:
                pothole_area_pixels = int(np.count_nonzero(result["mask"]))
                result["road_mask_overlap_percent"] = round(
                    100.0
                    * np.count_nonzero((result["mask"] > 0) & road_binary)
                    / pothole_area_pixels,
                    2,
                ) if pothole_area_pixels else 100.0
                result["damage_ratio"] = pothole_area_pixels / road_area_pixels
                result["damage_ratio_percent"] = round(100.0 * result["damage_ratio"], 4)
        else:
            road_mask_overlap_percent = None
            analyzed_area_pixels = image.shape[0] * image.shape[1]
            for result in results:
                result["damage_ratio"] = result["pixel_area"] / analyzed_area_pixels
                result["damage_ratio_percent"] = round(
                    100.0 * float(result["damage_ratio"]), 4
                )
        health = health_assessment(
            len(results),
            damaged_area_pixels,
            image.shape[0] * image.shape[1],
            road_area_pixels,
            total_pothole_area_m2,
        )
        health["road_mask_used"] = road_mask is not None
        fallback_reason = getattr(self, "_road_segmentation_fallback_reason", None)
        health["road_area_fallback_used"] = fallback_reason is not None
        health["road_area_fallback_reason"] = fallback_reason
        health["road_mask_damage_overlap_percent"] = (
            round(road_mask_overlap_percent, 2)
            if road_mask_overlap_percent is not None
            else None
        )
        health["predicted_road_area_pixels"] = (
            predicted_road_area_pixels if road_mask is not None else None
        )
        return total_mask, results, health

    def process_image(
        self,
        image: np.ndarray,
        meters_per_pixel: float | None = None,
    ) -> tuple[np.ndarray, list[dict[str, object]], dict[str, object]]:
        """Convenience wrapper: detect then segment the full *image*."""
        return self.process_detections(image, self.detect(image), meters_per_pixel)