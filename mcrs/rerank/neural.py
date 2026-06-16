"""K3 — neural (cross-encoder) reranker (F2 Reranker) + stacking surface.

Sharpens the top-1..3 by re-scoring ONLY the top `cross_encoder_k` of the K2-ranked pool with a
semantic model (e.g. BAAI/bge-reranker-v2-m3) over (dialogue query, enriched track doc) pairs.
Never enlarges the pool (R7 owns recall) and never reorders below the slice. The scorer is injected
(`score_fn(pairs) -> list[float]`) so logic is testable GPU-free; production wires the cross-encoder.

Two integration modes:
- `rerank()` — final-stage: reorder the top slice by neural score (simple, leak-free at inference).
- `score()`  — stacking surface: {track_id -> neural score} for the top slice, fed back as a K1
  feature. NOTE: as a TRAIN feature it must be produced OUT-OF-FOLD (K1 §4.1) or it leaks; that
  cross-fit harness is a separate step — `score()` itself just computes the scores.
See `.claude/documents/features/52_K3_neural_reranker.md`.
"""
from __future__ import annotations

from typing import Callable, Optional

from mcrs.contracts import Candidate, RankedList, TurnContext

ScoreFn = Callable[[list[tuple[str, str]]], list[float]]


class NeuralReranker:
    label = "neural"

    def __init__(self, catalog, query_builder, score_fn: ScoreFn,
                 cross_encoder_k: int = 100, enriched: bool = True,
                 max_doc_chars: int = 2000, model_revision: Optional[str] = None) -> None:
        self.catalog = catalog
        self.qb = query_builder
        self.score_fn = score_fn
        self.cross_encoder_k = cross_encoder_k
        self.enriched = enriched
        self.max_doc_chars = max_doc_chars      # coarse doc-side cap (preserve the query side); the
        #                                         cross-encoder tokenizer truncates the doc further
        # provenance only (train==serve pin recorded by D1, spec §8); the model itself lives in score_fn
        self.model_revision = model_revision

    def _doc(self, track_id: str) -> str:
        return self.catalog.id_to_metadata(track_id, enriched=self.enriched)[:self.max_doc_chars]

    def score(self, ctx: TurnContext, candidates: list[Candidate]) -> dict[str, float]:
        """{track_id -> neural score} for the top `cross_encoder_k` candidates (by incoming order)."""
        top = candidates[:self.cross_encoder_k]
        if not top:
            return {}
        query = self.qb.build(ctx).text
        scores = self.score_fn([(query, self._doc(c.track_id)) for c in top])
        if len(scores) != len(top):                 # loud fail vs silent zip truncation
            raise ValueError(f"score_fn returned {len(scores)} scores for {len(top)} candidates")
        return {c.track_id: float(s) for c, s in zip(top, scores)}

    def rerank(self, ctx: TurnContext, candidates: list[Candidate]) -> RankedList:
        if len(candidates) <= 1:
            return RankedList(turn=ctx, items=list(candidates))
        sc = self.score(ctx, candidates)
        k = self.cross_encoder_k
        top, rest = candidates[:k], candidates[k:]
        # stable sort: ties keep the incoming (K2) order; rest stays in K2 order below the slice
        top_sorted = sorted(top, key=lambda c: -sc.get(c.track_id, 0.0))
        return RankedList(turn=ctx, items=top_sorted + rest)
