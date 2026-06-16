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


def dedup_with_per_bucket_cap(
    beam_outputs: list[list[str]],
    cap: int = 1,
) -> list[str]:
    """Deduplicate beam outputs across collision buckets with per-bucket cap.

    Pass 1: take cap items from each beam's bucket, in beam order.
    Pass 2: spillover — take remaining items from each bucket, in beam-then-rank order.
    Globally unique track_ids (a track appearing in multiple beam buckets
    appears once in the output, at its earliest position).
    """
    seen: set[str] = set()
    out: list[str] = []
    for bucket in beam_outputs:
        taken = 0
        for tid in bucket:
            if taken >= cap:
                break
            if tid in seen:
                continue
            out.append(tid)
            seen.add(tid)
            taken += 1
    for bucket in beam_outputs:
        for tid in bucket:
            if tid in seen:
                continue
            out.append(tid)
            seen.add(tid)
    return out
