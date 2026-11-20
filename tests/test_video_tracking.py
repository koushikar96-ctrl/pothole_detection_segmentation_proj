import pytest

from src.video_tracking import PotholeTracker


def test_repeated_detection_in_nearby_frames_keeps_one_track():
    tracker = PotholeTracker()

    first_ids, new_count = tracker.update([(10, 10, 40, 40, 0.9)])
    second_ids, second_new_count = tracker.update([(13, 12, 43, 42, 0.8)])

    assert first_ids == second_ids
    assert new_count == 1
    assert second_new_count == 0
    assert tracker.unique_count == 1


def test_different_potholes_create_distinct_tracks():
    tracker = PotholeTracker()

    track_ids, new_count = tracker.update([
        (10, 10, 40, 40, 0.9),
        (200, 10, 230, 40, 0.8),
    ])

    assert track_ids == [1, 2]
    assert new_count == 2
    assert tracker.unique_count == 2


def test_track_survives_a_brief_missed_detection():
    tracker = PotholeTracker(max_missed_frames=1)
    first_ids, _ = tracker.update([(10, 10, 40, 40, 0.9)])

    tracker.update([])
    reappeared_ids, new_count = tracker.update([(10, 10, 40, 40, 0.85)])

    assert reappeared_ids == first_ids
    assert new_count == 0
    assert tracker.unique_count == 1


def test_track_expires_after_configured_missed_frames():
    tracker = PotholeTracker(max_missed_frames=1)
    tracker.update([(10, 10, 40, 40, 0.9)])
    tracker.update([])
    tracker.update([])

    track_ids, new_count = tracker.update([(10, 10, 40, 40, 0.9)])

    assert track_ids == [2]
    assert new_count == 1
    assert tracker.unique_count == 2


def test_negative_missed_frame_limit_is_rejected():
    with pytest.raises(ValueError, match="max_missed_frames"):
        PotholeTracker(max_missed_frames=-1)
