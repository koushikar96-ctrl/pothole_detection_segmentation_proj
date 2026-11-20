import argparse
from pathlib import Path
import sys

import cv2
import json

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from pipeline import PotholePipeline
from road_segmentation import DEFAULT_ROAD_SEGMENTATION_MODEL


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ROI segmentation and road-health assessment")
    parser.add_argument("image", type=Path)
    parser.add_argument("--segmentation-checkpoint", type=Path)
    parser.add_argument("--segmentation-mode", choices=("roi", "full_image"), default="roi")
    parser.add_argument("--road-segmentation-model", default=DEFAULT_ROAD_SEGMENTATION_MODEL)
    parser.add_argument("--detection-checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--meters-per-pixel", type=float)
    args = parser.parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"Unable to read {args.image}")
    pipeline = PotholePipeline(
        args.detection_checkpoint,
        args.segmentation_checkpoint,
        segmentation_mode=args.segmentation_mode,
        road_segmentation_model_id=args.road_segmentation_model,
    )
    mask, detections, health = pipeline.process_image(image)
    display = image.copy()
    display[mask > 0] = (0, 0, 255)
    display = cv2.addWeighted(image, 0.7, display, 0.3, 0)
    for x1, y1, x2, y2, confidence in pipeline.detect(image):
        cv2.rectangle(display, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(display, f"pothole {confidence:.2f}", (int(x1), max(20, int(y1) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    output_path = args.output or args.image.with_name(f"{args.image.stem}_result.jpg")
    cv2.imwrite(str(output_path), display)
    print(json.dumps({"output": str(output_path), "health": health, "potholes": [{key: value for key, value in item.items() if key != "mask"} for item in detections]}, indent=2))