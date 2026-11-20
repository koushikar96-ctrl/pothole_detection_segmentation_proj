import json
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from tools.train_road_segformer import (
    IGNORE_LABEL,
    calculate_iou,
    load_idd_samples,
)


def _write_idd_sample(root, split, drive, frame, objects):
    annotation_dir = root / "gtFine" / split / drive
    image_dir = root / "leftImg8bit" / split / drive
    annotation_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    annotation_path = annotation_dir / f"{frame}_gtFine_polygons.json"
    image_path = image_dir / f"{frame}_leftImg8bit.jpg"
    annotation_path.write_text(
        json.dumps({"imgWidth": 80, "imgHeight": 40, "objects": objects}),
        encoding="utf-8",
    )
    assert cv2.imwrite(str(image_path), np.zeros((40, 80, 3), dtype=np.uint8))


def test_load_idd_samples_uses_only_visible_road_polygons(tmp_path):
    _write_idd_sample(
        tmp_path,
        "train",
        "drive1",
        "frame1",
        [
            {
                "label": "road",
                "polygon": [[0, 0], [39, 0], [39, 39], [0, 39]],
            },
            {
                "label": "drivable fallback",
                "polygon": [[40, 0], [79, 0], [79, 39], [40, 39]],
            },
            {
                "label": "road",
                "deleted": 1,
                "polygon": [[40, 0], [79, 0], [79, 39], [40, 39]],
            },
        ],
    )

    samples = load_idd_samples(tmp_path, "train", (20, 10), None, 42)

    image_path, mask = samples[0]
    assert image_path.name == "frame1_leftImg8bit.jpg"
    assert mask.shape == (10, 20)
    assert np.all(mask[:, :10] == 1)
    assert np.all(mask[:, 10:] == IGNORE_LABEL)


def test_load_idd_samples_applies_deterministic_limit(tmp_path):
    for frame in ("frame1", "frame2", "frame3"):
        _write_idd_sample(
            tmp_path,
            "train",
            "drive1",
            frame,
            [{"label": "road", "polygon": [[0, 0], [79, 0], [79, 39]]}],
        )

    first = load_idd_samples(tmp_path, "train", (20, 10), 2, 42)
    second = load_idd_samples(tmp_path, "train", (20, 10), 2, 42)

    assert len(first) == 2
    assert [sample[0] for sample in first] == [sample[0] for sample in second]


def test_calculate_iou_ignores_fallback_pixels():
    class FakeModel:
        def eval(self):
            return self

        def __call__(self, *, pixel_values):
            height, width = pixel_values.shape[-2:]
            logits = torch.zeros((1, 2, height, width))
            logits[:, 1] = 1
            return SimpleNamespace(logits=logits)

    loader = [
        (
            torch.zeros((1, 3, 1, 3)),
            torch.tensor([[[1, 0, IGNORE_LABEL]]]),
        )
    ]

    iou = calculate_iou(FakeModel(), loader, torch.device("cpu"), road_class_id=1)

    assert iou == 0.5
