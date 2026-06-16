"""R7 — per-segment fusion weight sweep.

Cheap re-fusion over the probe's cached per-channel ranked lists (no model re-runs): try candidate
weight vectors per cold/warm segment and report fused recall@k, so segment-aware weights are set on
evidence (plan §7.3 #3) before wiring config->segment_weights. Reuses RRFFusion.fuse_per_sub.
"""
from __future__ import annotations

from typing import Optional

from mcrs.eval.diagnostics import recall_at_k
from mcrs.retrieval.fusion import RRFFusion


def fused_recall(per_channel: dict[str, list[list[str]]], weights: dict[str, float],
                 golds: list[str], ks: list[int], idxs: Optional[list[int]] = None,
                 k: int = 60, topk: Optional[int] = None) -> dict[int, float]:
    """Re-fuse cached per-channel lists with `weights` ({label: w}, missing -> 0) and return
    {kk: recall@kk} over the query indices `idxs` (default all)."""
    idxs = list(range(len(golds))) if idxs is None else idxs
    maxk = topk or max(ks)
    # weight 0 == channel dropped (consistent with RRFFusion.fuse); fuse_per_sub itself doesn't skip,
    # so exclude 0-weight channels here or their items leak into empty slots at score 0.
    active = [l for l in per_channel if float(weights.get(l, 0.0)) != 0.0]
    if not active:
        return {kk: 0.0 for kk in ks}
    fused = RRFFusion.fuse_per_sub([per_channel[l] for l in active],
                                   [float(weights[l]) for l in active], k, maxk)
    return {kk: (sum(recall_at_k(fused[i], golds[i], kk) for i in idxs) / len(idxs)) if idxs else 0.0
            for kk in ks}


def segment_weight_sweep(per_channel: dict[str, list[list[str]]], golds: list[str],
                         segments: list[str], candidates: list[tuple[str, dict[str, float]]],
                         ks: list[int] = (100, 200), k: int = 60) -> dict[str, list[tuple]]:
    """For each segment, score each candidate (name, weights) by fused recall@ks on that segment's
    turns. Returns {segment: [(name, {kk: recall}), ...]} sorted by recall@max(ks) descending."""
    ks = list(ks)
    out: dict[str, list[tuple]] = {}
    for seg in sorted(set(segments)):
        idxs = [i for i, s in enumerate(segments) if s == seg]
        scored = [(name, fused_recall(per_channel, w, golds, ks, idxs, k)) for name, w in candidates]
        scored.sort(key=lambda nr: -nr[1][max(ks)])
        out[seg] = scored
    return out
