"""R7 — weighted Reciprocal Rank Fusion.

score(d) = Σ_r w_r / (k + rank_r(d)); rank 1-indexed; absent => 0; dedup by track_id; stable
tie-break by first-seen. Emits F2 Candidates (rrf_score + per-channel ranks) for K1/K2. Channels
are pulled to topk_internal (>= fusion-K) before fusing. See feature doc 46_R7. F2 channel itself.
"""
from __future__ import annotations

from typing import Optional

from mcrs.contracts import Candidate


class RRFFusion:
    label = "rrf"

    def __init__(self, channels: list, weights: Optional[list[float]] = None, k: int = 60) -> None:
        self.channels = channels
        self.weights = weights if weights is not None else [1.0] * len(channels)
        self.k = k

    @staticmethod
    def fuse_per_sub(per_sub: list[list[list[str]]], weights: list[float],
                     k: int, topk: int) -> list[list[str]]:
        """Pure weighted RRF over precomputed per-channel rankings (cheap weight re-fusion)."""
        n_queries = len(per_sub[0]) if per_sub else 0
        results: list[list[str]] = []
        for qi in range(n_queries):
            fused: dict[str, float] = {}
            for si, w in enumerate(weights):
                for rank, tid in enumerate(per_sub[si][qi], start=1):
                    fused[tid] = fused.get(tid, 0.0) + w / (k + rank)
            results.append([t for t, _ in sorted(fused.items(), key=lambda kv: -kv[1])[:topk]])
        return results

    def _per_sub(self, queries, topk_internal, batch_context, user_ids):
        return [ch.batch_text_to_item_retrieval(queries, topk_internal, batch_context, user_ids)
                for ch in self.channels]

    def fuse(self, queries: list[str], topk: int, topk_internal: Optional[int] = None,
             batch_context: Optional[list[dict]] = None,
             user_ids: Optional[list[str]] = None) -> list[list[Candidate]]:
        ti = topk_internal or topk
        per_sub = self._per_sub(queries, ti, batch_context, user_ids)
        results: list[list[Candidate]] = []
        for qi in range(len(queries)):
            fused: dict[str, float] = {}
            ranks: dict[str, dict[str, int]] = {}
            for si, ch in enumerate(self.channels):
                w = self.weights[si]
                for rank, tid in enumerate(per_sub[si][qi], start=1):
                    fused[tid] = fused.get(tid, 0.0) + w / (self.k + rank)
                    ranks.setdefault(tid, {})[ch.label] = rank
            order = sorted(fused.items(), key=lambda kv: -kv[1])[:topk]
            results.append([
                Candidate(track_id=tid, rrf_score=score, channel_ranks=ranks.get(tid, {}))
                for tid, score in order
            ])
        return results

    def batch_text_to_item_retrieval(self, queries, topk, batch_context=None, user_ids=None):
        pools = self.fuse(queries, topk, batch_context=batch_context, user_ids=user_ids)
        return [[c.track_id for c in pool] for pool in pools]
