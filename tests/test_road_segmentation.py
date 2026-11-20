from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from src.road_segmentation import RoadSegmenter


class FakeProcessor:
    @classmethod
    def from_pretrained(cls, _model_id):
        return cls()

    def __call__(self, *, images, return_tensors):
        assert images.shape == (4, 6, 3)
        assert return_tensors == "pt"
        return {"pixel_values": torch.zeros((1, 3, 2, 2))}


class FakeModel(nn.Module):
    config = SimpleNamespace(id2label={0: "sidewalk", 1: "road"})

    @classmethod
    def from_pretrained(cls, _model_id):
        return cls()

    def forward(self, **_inputs):
        logits = torch.tensor(
            [[
                [[0.0, 10.0], [0.0, 10.0]],
                [[10.0, 0.0], [10.0, 0.0]],
            ]]
        )
        return SimpleNamespace(logits=logits)


def test_road_segmenter_uses_road_class_label_and_restores_input_size(monkeypatch):
    import transformers

    monkeypatch.setattr(
        transformers.AutoImageProcessor,
        "from_pretrained",
        classmethod(lambda _cls, _model_id: FakeProcessor()),
    )
    monkeypatch.setattr(
        transformers.SegformerForSemanticSegmentation,
        "from_pretrained",
        classmethod(lambda _cls, _model_id: FakeModel()),
    )
    segmenter = RoadSegmenter("test-model", torch.device("cpu"))
    image = np.zeros((4, 6, 3), dtype=np.uint8)

    road_mask = segmenter.predict(image)

    assert road_mask.shape == image.shape[:2]
    assert road_mask.dtype == np.uint8
    assert segmenter.road_class_id == 1
    assert np.all(road_mask[:, :3] == 1)
    assert np.all(road_mask[:, 3:] == 0)
