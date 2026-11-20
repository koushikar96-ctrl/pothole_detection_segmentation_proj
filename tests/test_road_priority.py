import pytest

from src.road_priority import inspection_priority, priority_from_rhi, select_worst_record


@pytest.mark.parametrize(
    ("rhi", "expected"),
    [(39.99, "high"), (40.0, "medium"), (70.0, "medium"), (70.01, "low")],
)
def test_priority_from_rhi_uses_current_bands(rhi, expected):
    assert priority_from_rhi(rhi) == expected


def test_saved_inspection_priority_is_recalculated_from_worst_rhi():
    inspection = {
        "average_rhi": 99.94,
        "worst_rhi": 99.01,
        "repair_priority": "medium",
    }

    assert inspection_priority(inspection) == "low"


def test_worst_record_uses_lowest_rhi_not_saved_priority():
    records = [
        {"repair_priority": "medium", "rhi": 99.96, "timestamp_seconds": 2.5},
        {"repair_priority": "low", "rhi": 99.01, "timestamp_seconds": 13.38},
    ]

    assert select_worst_record(records) is records[1]


def test_worst_record_uses_lowest_rhi_within_same_priority():
    records = [
        {"repair_priority": "high", "rhi": 20.0, "timestamp_seconds": 1.0},
        {"repair_priority": "high", "rhi": 25.0, "timestamp_seconds": 2.0},
        {"repair_priority": "high", "rhi": 29.0, "timestamp_seconds": 3.0},
    ]

    assert select_worst_record(records) is records[0]


def test_lowest_rhi_selects_high_over_medium():
    records = [
        {"repair_priority": "medium", "rhi": 30.0},
        {"repair_priority": "high", "rhi": 29.99},
    ]

    assert select_worst_record(records) is records[1]


def test_empty_records_have_the_existing_empty_fallback():
    assert select_worst_record([]) == {}
