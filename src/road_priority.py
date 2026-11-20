from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


# Lower ranks are more urgent; "urgent" is retained for legacy saved queues.
PRIORITY_RANK = {
    "urgent": 0,
    "high": 0,
    "medium": 1,
    "low": 2,
}


def priority_from_rhi(rhi: float) -> str:
    """Derive the current repair priority from RHI only."""
    if rhi < 40:
        return "high"
    if rhi <= 70:
        return "medium"
    return "low"


def inspection_priority(record: Mapping[str, Any]) -> str:
    """Recalculate a saved inspection priority from its worst observed RHI."""
    rhi = record.get("worst_rhi")
    if rhi is None:
        rhi = record.get("rhi", record.get("average_rhi"))
    if rhi is None:
        raise ValueError("Inspection record is missing RHI data.")
    return priority_from_rhi(float(rhi))


def priority_rank(priority: object) -> int:
    """Return the repair-priority rank, placing unknown values last."""
    return PRIORITY_RANK.get(str(priority).lower(), len(PRIORITY_RANK))


def select_worst_record(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Return the road frame needing the earliest repair attention.

    Since the current priority bands are monotonic in RHI, lower RHI is always
    the frame needing earliest repair attention. Empty inputs produce an empty
    mapping so callers can retain their existing fallback.
    """
    if not records:
        return {}

    return min(
        records,
        key=lambda record: (
            float(record.get("rhi", 100.0)),
            float(record.get("timestamp_seconds", 0.0)),
        ),
    )
