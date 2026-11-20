import pytest

from src.area_health import (
    health_assessment,
    highest_road_damage_severity,
    road_damage_severity,
)


@pytest.mark.parametrize(
    ("area_m2", "expected_severity"),
    [
        (None, "Unrated"),
        (0.0, "Low"),
        (0.0499, "Low"),
        (0.05, "Medium"),
        (0.1999, "Medium"),
        (0.20, "High"),
        (0.9999, "High"),
        (1.0, "Critical"),
    ],
)
def test_road_severity_uses_aggregate_calibrated_area_thresholds(area_m2, expected_severity):
    assert road_damage_severity(area_m2) == expected_severity


def test_road_severity_rejects_negative_aggregate_area():
    with pytest.raises(ValueError, match="non-negative"):
        road_damage_severity(-0.01)


@pytest.mark.parametrize(
    ("severities", "expected"),
    [
        ([], "Unrated"),
        (["Unrated", "Unrated"], "Unrated"),
        (["Low", "Medium", "High"], "High"),
        (["Critical", "Low", "High"], "Critical"),
    ],
)
def test_highest_road_damage_severity(severities, expected):
    assert highest_road_damage_severity(severities) == expected


def test_road_health_uses_road_area_for_rdp_and_rhi():
    health = health_assessment(
        pothole_count=3,
        total_pixel_area=8400,
        image_area_pixels=200000,
        road_area_pixels=100000,
    )

    assert health["damage_coverage"] == 0.084
    assert health["rdp_percent"] == 8.4
    assert health["rhi"] == 91.6
    assert health["road_area_pixels"] == 100000
    assert health["repair_priority"] == "low"
    assert health["road_damage_severity"] == "Unrated"


@pytest.mark.parametrize("pothole_count", [1, 3, 9])
@pytest.mark.parametrize(
    ("rdp_percent", "expected_priority"),
    [(10.0, "low"), (50.0, "medium"), (80.0, "high")],
)
def test_pothole_count_does_not_change_rhi_priority(
    pothole_count,
    rdp_percent,
    expected_priority,
):
    health = health_assessment(
        pothole_count=pothole_count,
        total_pixel_area=round(rdp_percent * 100),
        image_area_pixels=10000,
    )

    assert health["repair_priority"] == expected_priority


@pytest.mark.parametrize(
    ("rdp_percent", "expected_priority"),
    [
        (60.01, "high"),
        (60.0, "medium"),
        (30.0, "medium"),
        (29.99, "low"),
    ],
)
def test_repair_priority_uses_presentation_rhi_bands(rdp_percent, expected_priority):
    health = health_assessment(
        pothole_count=1,
        total_pixel_area=round(rdp_percent * 100000),
        image_area_pixels=10000000,
    )

    assert health["rdp_percent"] == round(rdp_percent, 2)
    assert health["rhi"] == round(100 - rdp_percent, 2)
    assert health["repair_priority"] == expected_priority


def test_road_health_rejects_empty_road_mask_area():
    with pytest.raises(ValueError, match="at least one pixel"):
        health_assessment(0, 0, 100, road_area_pixels=0)
