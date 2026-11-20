from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

try:
    from .road_priority import priority_from_rhi
except ImportError:
    from road_priority import priority_from_rhi


ROAD_DAMAGE_SEVERITY_RANK = {
    "Unrated": 0,
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Critical": 4,
}


@dataclass
class AreaEstimate:
    pixel_area: int
    area_m2: float | None
    scale_m_per_pixel: float | None
    equivalent_diameter_pixels: float


def road_damage_severity(total_area_m2: float | None) -> str:
    """Classify aggregate road damage by calibrated total pothole area."""
    if total_area_m2 is None:
        return "Unrated"
    if total_area_m2 < 0:
        raise ValueError("Total pothole area must be non-negative.")
    if total_area_m2 < 0.05:
        return "Low"
    if total_area_m2 < 0.20:
        return "Medium"
    if total_area_m2 < 1.0:
        return "High"
    return "Critical"


def highest_road_damage_severity(severities: Iterable[str]) -> str:
    """Return the highest rated severity, or Unrated when no values are available."""
    highest = "Unrated"
    for severity in severities:
        if severity not in ROAD_DAMAGE_SEVERITY_RANK:
            raise ValueError(f"Unknown road damage severity: {severity!r}")
        if ROAD_DAMAGE_SEVERITY_RANK[severity] > ROAD_DAMAGE_SEVERITY_RANK[highest]:
            highest = severity
    return highest


def estimate_area(mask: np.ndarray, meters_per_pixel: float | None = None) -> AreaEstimate:
    binary = mask > 0
    pixel_area = int(binary.sum())
    scale = float(meters_per_pixel) if meters_per_pixel and meters_per_pixel > 0 else None
    area_m2 = pixel_area * scale * scale if scale else None
    diameter = float(2.0 * np.sqrt(pixel_area / np.pi)) if pixel_area else 0.0
    return AreaEstimate(pixel_area, area_m2, scale, diameter)


def scale_from_reference(reference_pixels: float, reference_meters: float) -> float:
    if reference_pixels <= 0 or reference_meters <= 0:
        raise ValueError("Reference dimensions must be positive")
    return reference_meters / reference_pixels


def health_assessment(
    pothole_count: int,
    total_pixel_area: int,
    image_area_pixels: int,
    road_area_pixels: int | None = None,
    total_pothole_area_m2: float | None = None,
) -> dict[str, object]:
    """Calculate RDP, RHI, condition, and repair priority from road-pixel areas."""
    analyzed_area_pixels = road_area_pixels if road_area_pixels is not None else image_area_pixels
    if analyzed_area_pixels <= 0:
        raise ValueError("Analyzed road area must contain at least one pixel")

    coverage = min(1.0, max(0.0, total_pixel_area / analyzed_area_pixels))
    rdp_percent = coverage * 100.0
    rhi_value = max(0.0, min(100.0, 100.0 - rdp_percent))
    rhi = round(rhi_value, 2)

    if rhi_value >= 80:
        condition = "good"
    elif rhi_value >= 60:
        condition = "moderate"
    elif rhi_value >= 40:
        condition = "poor"
    else:
        condition = "critical"

    priority = priority_from_rhi(rhi_value)

    return {
        "pothole_count": pothole_count,
        "damaged_area_pixels": total_pixel_area,
        "road_area_pixels": analyzed_area_pixels,
        "damage_coverage": coverage,
        "rdp_percent": round(rdp_percent, 2),
        "rhi": rhi,
        "health_score": rhi,
        "condition": condition,
        "repair_priority": priority,
        "total_pothole_area_m2": total_pothole_area_m2,
        "road_damage_severity": road_damage_severity(total_pothole_area_m2),
    }