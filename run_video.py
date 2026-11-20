from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from pipeline import PotholePipeline
from road_segmentation import DEFAULT_ROAD_SEGMENTATION_MODEL
from video_tracking import PotholeTracker


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pothole detection and segmentation on recorded video")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--detection-checkpoint", type=Path)
    parser.add_argument("--segmentation-checkpoint", type=Path)
    parser.add_argument("--segmentation-mode", choices=("roi", "full_image"), default="roi")
    parser.add_argument("--road-segmentation-model", default=DEFAULT_ROAD_SEGMENTATION_MODEL)
    parser.add_argument("--threshold", type=float, default=0.25)
    args = parser.parse_args()
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise SystemExit(f"Unable to open {args.input}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    pipeline = PotholePipeline(
        args.detection_checkpoint,
        args.segmentation_checkpoint,
        threshold=args.threshold,
        segmentation_mode=args.segmentation_mode,
        road_segmentation_model_id=args.road_segmentation_model,
    )
    tracker = PotholeTracker()
    native_track_ids_seen: set[int] = set()
    if pipeline.detection_backend == "ecyolo":
        pipeline.reset_video_tracker()
    while True:
        success, frame = capture.read()
        if not success:
            break
        if pipeline.detection_backend == "ecyolo":
            frame_detections, track_ids = pipeline.track_video(frame)
            native_track_ids_seen.update(
                int(track_id) for track_id in track_ids if track_id is not None
            )
            unique_potholes = len(native_track_ids_seen)
            display_track_ids = [
                track_id if track_id is not None else "?"
                for track_id in track_ids
            ]
        else:
            frame_detections = pipeline.detect(frame)
            track_ids, _ = tracker.update(frame_detections)
            unique_potholes = tracker.unique_count
            display_track_ids = track_ids
        road_mask = pipeline.segment_road(frame)
        mask, _, health = pipeline.process_detections(
            frame,
            frame_detections,
            road_mask=road_mask,
        )
        overlay = frame.copy()
        overlay[mask > 0] = (0, 0, 255)
        output = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        for (x1, y1, x2, y2, confidence), track_id in zip(frame_detections, display_track_ids):
            cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(
                output,
                f"pothole #{track_id} {confidence:.2f}",
                (int(x1), max(20, int(y1) - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        cv2.putText(
            output,
            f"Health: {health['condition']} | Priority: {health['repair_priority']} | "
            f"Unique potholes: {unique_potholes}"
            + (" | Road-area fallback" if health.get("road_area_fallback_used") else ""),
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        writer.write(output)
    capture.release()
    writer.release()
    print(f"Saved annotated video to {args.output}")


if __name__ == "__main__":
    main()