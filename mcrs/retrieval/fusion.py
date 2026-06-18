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

    def __init__(self, channels: list, weights: Optional[list[float]] = None, k: int = 60,
                 segment_weights: Optional[dict[str, list[float]]] = None) -> None:
        self.channels = channels
        self.weights = weights if weights is not None else [1.0] * len(channels)
        self.k = k
        # optional per-segment weight vectors, e.g. {"cold": [...], "warm": [...]}; a query's
        # segment is read from batch_context[i]["segment"]. Falls back to self.weights when
        # segment_weights is None, the ctx has no segment, or the segment isn't in the map.
        self.segment_weights = segment_weights

    def _weights_for(self, ctx: Optional[dict]) -> list[float]:
        if self.segment_weights and ctx:
            seg = ctx.get("segment")
            if seg in self.segment_weights:
                return self.segment_weights[seg]
        return self.weights

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

    def _queries_for(self, ch, queries, per_channel_queries):
        """A channel with a `query_key` present in `per_channel_queries` gets ITS query list (e.g.
        ColBERT's focused query); everyone else gets the main `queries` (A1 per-channel routing).

        The routed list MUST be 1:1 aligned (same length + order) with `queries`/`batch_context` —
        `fuse` reads `per_sub[si][qi]` positionally, so a length mismatch would silently fuse one
        turn's ColBERT pool into another turn's row. We hard-fail on a length mismatch rather than
        corrupt the pool."""
        key = getattr(ch, "query_key", None)
        if not per_channel_queries or not key or key not in per_channel_queries:
            return queries                        # unrouted channel, or no override supplied -> full query
        routed = per_channel_queries[key]
        if len(routed) != len(queries):
            raise ValueError(
                f"per_channel_queries['{key}'] has {len(routed)} queries but {len(queries)} main "
                f"queries — must be 1:1 aligned with the turns/batch_context (R7 per-channel routing)")
        return routed

    def _per_sub(self, queries, topk_internal, batch_context, user_ids, per_channel_queries=None):
        return [ch.batch_text_to_item_retrieval(
                    self._queries_for(ch, queries, per_channel_queries),
                    topk_internal, batch_context, user_ids)
                for ch in self.channels]

    def fuse(self, queries: list[str], topk: int, topk_internal: Optional[int] = None,
             batch_context: Optional[list[dict]] = None,
             user_ids: Optional[list[str]] = None,
             per_channel_queries: Optional[dict[str, list[str]]] = None) -> list[list[Candidate]]:
        ti = topk_internal or topk
        per_sub = self._per_sub(queries, ti, batch_context, user_ids, per_channel_queries)
        results: list[list[Candidate]] = []
        for qi in range(len(queries)):
            ctx = batch_context[qi] if batch_context and qi < len(batch_context) else None
            w_vec = self._weights_for(ctx)
            fused: dict[str, float] = {}
            ranks: dict[str, dict[str, int]] = {}
            for si, ch in enumerate(self.channels):
                w = w_vec[si]
                if w == 0:                       # zero-weighted channel = dropped (no pool pollution)
                    continue
                for rank, tid in enumerate(per_sub[si][qi], start=1):
                    fused[tid] = fused.get(tid, 0.0) + w / (self.k + rank)
                    ranks.setdefault(tid, {})[ch.label] = rank
            order = sorted(fused.items(), key=lambda kv: -kv[1])[:topk]
            results.append([
                Candidate(track_id=tid, rrf_score=score, channel_ranks=ranks.get(tid, {}))
                for tid, score in order
            ])
        return results

    def batch_text_to_item_retrieval(self, queries, topk, batch_context=None, user_ids=None,
                                     per_channel_queries=None):
        pools = self.fuse(queries, topk, batch_context=batch_context, user_ids=user_ids,
                          per_channel_queries=per_channel_queries)
        return [[c.track_id for c in pool] for pool in pools]
