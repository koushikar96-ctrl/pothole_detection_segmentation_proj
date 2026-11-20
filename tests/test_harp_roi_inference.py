import numpy as np
import pytest
import torch
from torch import nn

from src.pipeline import PotholePipeline
from src.roi import extract_roi


class ConstantLogits(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_size = None

    def forward(self, inputs):
        self.input_size = tuple(inputs.shape[-2:])
        return torch.full(
            (inputs.shape[0], 1, *self.input_size),
            8.0,
            device=inputs.device,
        )


def make_pipeline(mode):
    pipeline = object.__new__(PotholePipeline)
    pipeline.segmentation_mode = mode
    pipeline.device = torch.device("cpu")
    pipeline.segmentation_threshold = 0.5
    pipeline.segmenter = ConstantLogits()
    return pipeline


def fail_road_prediction(_):
    raise RuntimeError("inference failed")


def fail_pothole_prediction(_):
    raise RuntimeError("pothole segmentation failed")


def test_roi_inference_uses_training_size_and_restores_crop_shape():
    pipeline = make_pipeline("roi")
    crop = np.zeros((30, 60, 3), dtype=np.uint8)

    mask = pipeline.segment_roi(crop)

    assert pipeline.segmenter.input_size == (128, 128)
    assert mask.shape == crop.shape[:2]
    assert np.all(mask == 1)


def test_roi_mode_segments_and_pastes_each_detection_crop(monkeypatch):
    pipeline = make_pipeline("roi")
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    detection = (5, 4, 20, 15, 0.9)
    roi = extract_roi(image, detection[:4], detection[4])
    monkeypatch.setattr(
        pipeline,
        "segment_roi",
        lambda crop: np.ones(crop.shape[:2], dtype=np.uint8),
    )
    monkeypatch.setattr(
        pipeline,
        "segment_image",
        lambda _: pytest.fail("ROI mode must not segment the full image"),
    )

    total_mask, results, health = pipeline.process_detections(image, [detection])

    x1, y1, x2, y2 = roi.box
    assert total_mask.sum() == (x2 - x1) * (y2 - y1)
    assert np.all(total_mask[y1:y2, x1:x2] == 1)
    assert total_mask[0, 0] == 0
    assert results[0]["pixel_area"] == (x2 - x1) * (y2 - y1)
    assert results[0]["area_m2"] is None
    assert "severity" not in results[0]
    assert health["road_damage_severity"] == "Unrated"


def test_full_image_mode_keeps_existing_mask_gating(monkeypatch):
    pipeline = make_pipeline("full_image")
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    detection = (5, 4, 20, 15, 0.9)
    roi = extract_roi(image, detection[:4], detection[4])
    full_mask = np.ones(image.shape[:2], dtype=np.uint8)
    monkeypatch.setattr(pipeline, "segment_image", lambda _: full_mask)
    monkeypatch.setattr(
        pipeline,
        "segment_roi",
        lambda _: pytest.fail("Full-image mode must not segment individual ROIs"),
    )

    total_mask, _, _ = pipeline.process_detections(image, [detection])

    x1, y1, x2, y2 = roi.box
    assert total_mask.sum() == (x2 - x1) * (y2 - y1)
    assert total_mask[0, 0] == 0


def test_road_severity_uses_union_of_all_pothole_masks(monkeypatch):
    pipeline = make_pipeline("roi")
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = [
        (10, 10, 20, 20, 0.9),
        (60, 60, 70, 70, 0.9),
    ]
    monkeypatch.setattr(
        pipeline,
        "segment_roi",
        lambda crop: np.ones(crop.shape[:2], dtype=np.uint8),
    )

    total_mask, results, health = pipeline.process_detections(
        image,
        detections,
        meters_per_pixel=0.1,
    )

    expected_area_m2 = int(np.count_nonzero(total_mask)) * 0.01
    assert health["total_pothole_area_m2"] == pytest.approx(expected_area_m2)
    assert health["road_damage_severity"] == "Critical"
    assert all("severity" not in result for result in results)


def test_road_mask_does_not_discard_damage_outside_its_predicted_region(monkeypatch):
    pipeline = make_pipeline("roi")
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    detection = (5, 4, 20, 15, 0.9)
    roi = extract_roi(image, detection[:4], detection[4])
    monkeypatch.setattr(
        pipeline,
        "segment_roi",
        lambda crop: np.ones(crop.shape[:2], dtype=np.uint8),
    )
    road_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    x1, y1, x2, y2 = roi.box
    road_mask[y1:y1 + 2, x1:x1 + 2] = 1
    road_mask[0:2, 0:4] = 1

    _, results, health = pipeline.process_detections(image, [detection], road_mask=road_mask)

    pothole_area = (x2 - x1) * (y2 - y1)
    assert health["damaged_area_pixels"] == pothole_area
    assert health["road_area_pixels"] == 200
    assert health["rdp_percent"] == 96.0
    assert health["rhi"] == 4.0
    assert health["road_mask_used"] is True
    assert health["road_mask_damage_overlap_percent"] == round(400 / pothole_area, 2)
    assert health["repair_priority"] == "high"
    assert results[0]["pixel_area"] == pothole_area
    assert results[0]["damage_ratio_percent"] == 96.0
    assert results[0]["road_mask_overlap_percent"] == round(400 / pothole_area, 2)


@pytest.mark.parametrize(
    ("predict", "expected_reason"),
    [
        (fail_road_prediction, "RuntimeError: inference failed"),
        (lambda image: np.zeros(image.shape[:2], dtype=np.uint8), "ValueError: Road segmentation returned an empty mask."),
    ],
)
def test_road_segmentation_failure_falls_back_to_full_image(predict, expected_reason, monkeypatch):
    pipeline = make_pipeline("roi")
    pipeline.road_segmenter = type("RoadSegmenterStub", (), {"predict": staticmethod(predict)})()
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    detection = (5, 4, 20, 15, 0.9)
    monkeypatch.setattr(
        pipeline,
        "segment_roi",
        lambda crop: np.ones(crop.shape[:2], dtype=np.uint8),
    )

    road_mask = pipeline.segment_road(image)
    _, _, health = pipeline.process_detections(image, [detection], road_mask=road_mask)

    assert np.all(road_mask == 1)
    assert health["road_area_pixels"] == image.shape[0] * image.shape[1]
    assert health["road_area_fallback_used"] is True
    assert health["road_area_fallback_reason"] == expected_reason
    assert health["rdp_percent"] > 0


def test_pothole_segmentation_failure_is_not_replaced_by_road_fallback(monkeypatch):
    pipeline = make_pipeline("roi")
    pipeline.road_segmenter = type(
        "RoadSegmenterStub",
        (),
        {"predict": staticmethod(lambda image: np.ones(image.shape[:2], dtype=np.uint8))},
    )()
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    monkeypatch.setattr(
        pipeline,
        "segment_roi",
        fail_pothole_prediction,
    )

    road_mask = pipeline.segment_road(image)
    assert np.all(road_mask == 1)

    with pytest.raises(RuntimeError, match="pothole segmentation failed"):
        pipeline.process_detections(image, [(5, 4, 20, 15, 0.9)], road_mask=road_mask)


def test_invalid_segmentation_mode_is_rejected():
    with pytest.raises(ValueError, match="segmentation_mode"):
        PotholePipeline(segmentation_mode="letterbox")
