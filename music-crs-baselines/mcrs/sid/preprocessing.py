"""Pure functions for SID quantizer preprocessing + collision handling."""
from __future__ import annotations

import numpy as np


def concat_modalities(text: np.ndarray, cf: np.ndarray, audio: np.ndarray) -> np.ndarray:
    """L2-normalize each modality independently, concatenate.

    Cold-start CF rows (zero vectors) are preserved as zeros — no divide-by-zero.
    Output dtype is float32 for memory efficiency downstream.
    """
    def _l2(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        if norm == 0:
            return v.astype(np.float32)
        return (v / norm).astype(np.float32)

    return np.concatenate([_l2(text), _l2(cf), _l2(audio)])


def compute_collision_buckets(
    track_ids: list[str],
    sid_assignments: list[tuple[int, int, int]],
    popularity: dict[str, float],
) -> dict[tuple[int, int, int], list[str]]:
    """Group tracks by their 3-tuple SID; sort each bucket by descending popularity.

    Tracks with missing popularity get rank 0.0 (sort to bucket bottom).
    """
    if len(track_ids) != len(sid_assignments):
        raise ValueError(
            f"track_ids ({len(track_ids)}) and sid_assignments "
            f"({len(sid_assignments)}) must be the same length"
        )
    buckets: dict[tuple[int, int, int], list[str]] = {}
    for tid, sid in zip(track_ids, sid_assignments):
        buckets.setdefault(sid, []).append(tid)
    for sid in buckets:
        buckets[sid].sort(key=lambda t: -popularity.get(t, 0.0))
    return buckets
