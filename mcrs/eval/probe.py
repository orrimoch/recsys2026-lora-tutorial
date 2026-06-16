"""P0 — recall-ceiling probe.

Pure aggregation over precomputed per-channel rankings (one list of ids per turn, pulled at
depth): per-channel + fused recall@k, optional cold/warm split, and per-channel unique-recall
(golds a channel hits that no other does — the keep/drop signal, plan §7.3 req #2). The driver
that actually runs the channels lives in the notebook; this stays pure and unit-tested.
"""
from __future__ import annotations

from typing import Optional

from mcrs.eval.diagnostics import recall_at_k
from mcrs.retrieval.fusion import RRFFusion


def _recall(lists, golds, ks, idxs) -> dict[int, float]:
    out: dict[int, float] = {}
    for k in ks:
        out[k] = (sum(recall_at_k(lists[i], golds[i], k) for i in idxs) / len(idxs)) if idxs else 0.0
    return out


def recall_ceiling(
    per_channel: dict[str, list[list[str]]],
    golds: list[str],
    ks: list[int],
    segments: Optional[list[str]] = None,
    fusion_weights: Optional[dict[str, float]] = None,
    fusion_k: int = 60,
) -> dict:
    labels = list(per_channel)
    n = len(golds)
    all_idx = list(range(n))
    maxk = max(ks)
    seg_values = sorted(set(segments)) if segments else []

    def by_seg(lists):
        return {s: _recall(lists, golds, ks, [i for i in all_idx if segments[i] == s]) for s in seg_values}

    report: dict = {"per_channel": {}, "fused": {}}

    for lab in labels:
        lists = per_channel[lab]
        entry = {"recall": _recall(lists, golds, ks, all_idx)}
        if segments:
            entry["by_segment"] = by_seg(lists)
        report["per_channel"][lab] = entry

    # unique-recall: turns where exactly one channel hits the gold within maxk
    for lab in labels:
        uniq = 0
        for i in all_idx:
            hitters = [l for l in labels if recall_at_k(per_channel[l][i], golds[i], maxk) > 0]
            if hitters == [lab]:
                uniq += 1
        report["per_channel"][lab]["unique_recall"] = uniq / n if n else 0.0

    weights = [(fusion_weights or {}).get(l, 1.0) for l in labels]
    fused_lists = RRFFusion.fuse_per_sub([per_channel[l] for l in labels], weights, fusion_k, maxk)
    fused = {"recall": _recall(fused_lists, golds, ks, all_idx)}
    if segments:
        fused["by_segment"] = by_seg(fused_lists)
    report["fused"] = fused
    return report
