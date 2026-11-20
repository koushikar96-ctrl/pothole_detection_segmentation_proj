from __future__ import annotations

from dataclasses import dataclass
from math import hypot


Detection = tuple[float, float, float, float, float]


@dataclass
class _Track:
    track_id: int
    box: tuple[float, float, float, float]
    missed_frames: int = 0


class PotholeTracker:
    """Associate detections across sampled video frames and count unique tracks."""

    def __init__(self, max_missed_frames: int = 2) -> None:
        if max_missed_frames < 0:
            raise ValueError("max_missed_frames must be non-negative.")
        self.max_missed_frames = max_missed_frames
        self._tracks: list[_Track] = []
        self._next_track_id = 1

    @property
    def unique_count(self) -> int:
        return self._next_track_id - 1

    @staticmethod
    def _overlap(box_a: tuple[float, float, float, float],
                 box_b: tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
            0.0, min(ay2, by2) - max(ay1, by1)
        )
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - intersection
        return intersection / union if union > 0.0 else 0.0

    @staticmethod
    def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
        x1, y1, x2, y2 = box
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @staticmethod
    def _match_distance(box_a: tuple[float, float, float, float],
                        box_b: tuple[float, float, float, float]) -> float:
        center_a = PotholeTracker._center(box_a)
        center_b = PotholeTracker._center(box_b)
        return hypot(center_a[0] - center_b[0], center_a[1] - center_b[1])

    @staticmethod
    def _can_match(box_a: tuple[float, float, float, float],
                   box_b: tuple[float, float, float, float]) -> bool:
        if PotholeTracker._overlap(box_a, box_b) >= 0.1:
            return True
        width_a, height_a = box_a[2] - box_a[0], box_a[3] - box_a[1]
        width_b, height_b = box_b[2] - box_b[0], box_b[3] - box_b[1]
        max_size = max(width_a, height_a, width_b, height_b, 0.0)
        return PotholeTracker._match_distance(box_a, box_b) <= max(32.0, 1.5 * max_size)

    def update(self, detections: list[Detection]) -> tuple[list[int], int]:
        """Return track IDs aligned with detections and the number of new tracks."""
        boxes = [(x1, y1, x2, y2) for x1, y1, x2, y2, _ in detections]
        candidates = [
            (
                self._overlap(track.box, box),
                self._match_distance(track.box, box),
                track_index,
                detection_index,
            )
            for track_index, track in enumerate(self._tracks)
            for detection_index, box in enumerate(boxes)
            if self._can_match(track.box, box)
        ]
        candidates.sort(key=lambda candidate: (-candidate[0], candidate[1]))

        track_assignments: dict[int, int] = {}
        matched_detections: set[int] = set()
        for _, _, track_index, detection_index in candidates:
            if track_index in track_assignments or detection_index in matched_detections:
                continue
            track_assignments[track_index] = detection_index
            matched_detections.add(detection_index)

        track_ids = [0] * len(detections)
        for track_index, track in enumerate(self._tracks):
            detection_index = track_assignments.get(track_index)
            if detection_index is None:
                track.missed_frames += 1
                continue
            track.box = boxes[detection_index]
            track.missed_frames = 0
            track_ids[detection_index] = track.track_id

        new_tracks = 0
        for detection_index, box in enumerate(boxes):
            if detection_index in matched_detections:
                continue
            track_id = self._next_track_id
            self._next_track_id += 1
            new_tracks += 1
            track_ids[detection_index] = track_id
            self._tracks.append(_Track(track_id, box))

        self._tracks = [
            track for track in self._tracks
            if track.missed_frames <= self.max_missed_frames
        ]
        return track_ids, new_tracks
