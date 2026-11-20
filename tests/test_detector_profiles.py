from cache.detector_profiles import DETECTOR_OPTIONS, resolve_detector_config


def test_hf_pothole_yolov8_is_the_only_detector_option():
    assert DETECTOR_OPTIONS == ["Harisanth/Pothole-Finetuned-YOLOv8"]


def test_hf_pothole_yolov8_uses_local_artifacts_folder():
    config = resolve_detector_config("Harisanth/Pothole-Finetuned-YOLOv8")
    assert config["backend"] == "ecyolo"
    display_path = str(config["checkpoint"]).replace("\\", "/")
    runtime_path = str(config["runtime_checkpoint"]).replace("\\", "/")
    assert display_path.endswith("artifacts/ec_yolo_training/best.pt")
    assert runtime_path.endswith("artifacts/hf_pothole_yolov8/best.pt")
    assert config["kind"] == "EC-YOLO"
    assert config["is_custom_detector"] is False


def test_hf_model_id_is_available_for_hub_fallback():
    config = resolve_detector_config("Harisanth/Pothole-Finetuned-YOLOv8")
    assert config["model_id"] == "Harisanth/Pothole-Finetuned-YOLOv8"
