"""F1 — cold/warm segment label. F1 only *applies* the threshold; P0 decides its value."""
from __future__ import annotations


def segment_for(history_tids: list[str], cold_threshold: int) -> str:
    """cold if the user has <= cold_threshold history tracks, else warm."""
    return "cold" if len(history_tids) <= cold_threshold else "warm"
