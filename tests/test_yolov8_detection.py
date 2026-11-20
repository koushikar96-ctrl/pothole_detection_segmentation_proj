from types import SimpleNamespace

from src.pipeline import PotholePipeline


def test_yolov8_detection_uses_tighter_iou_suppression():
    class StubYolo:
        def __init__(self):
            self.kwargs = None

        def __call__(self, image, **kwargs):
            self.kwargs = kwargs
            return [SimpleNamespace(boxes=None)]

    pipeline = PotholePipeline.__new__(PotholePipeline)
    pipeline.yolo_detector = StubYolo()
    pipeline.threshold = 0.25

    assert pipeline.ecyolo_detect(object()) == []
    assert pipeline.yolo_detector.kwargs == {
        "conf": 0.25,
        "iou": 0.45,
        "verbose": False,
    }


def test_yolov8_video_tracking_returns_detection_boxes_and_persistent_ids():
    class StubValue:
        def __init__(self, value):
            self.value = value

        def tolist(self):
            return self.value

        def item(self):
            return self.value

    class StubBox:
        xyxy = [StubValue([10, 20, 40, 50])]
        conf = [StubValue(0.9)]

    class StubBoxes(list):
        id = [StubValue(7)]

    class StubYolo:
        def track(self, image, **kwargs):
            self.kwargs = kwargs
            return [SimpleNamespace(boxes=StubBoxes([StubBox()]))]

    pipeline = PotholePipeline.__new__(PotholePipeline)
    pipeline.detection_backend = "ecyolo"
    pipeline.yolo_detector = StubYolo()
    pipeline.threshold = 0.25

    detections, track_ids = pipeline.track_video(object())

    assert detections == [(10.0, 20.0, 40.0, 50.0, 0.9)]
    assert track_ids == [7]
    assert pipeline.yolo_detector.kwargs == {
        "conf": 0.25,
        "iou": 0.45,
        "persist": True,
        "tracker": "bytetrack.yaml",
        "verbose": False,
    }
