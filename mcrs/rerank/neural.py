"""K3 — neural (cross-encoder) reranker (F2 Reranker) + stacking surface.

Sharpens the top-1..3 by re-scoring ONLY the top `cross_encoder_k` of the K2-ranked pool with a
semantic model (e.g. BAAI/bge-reranker-v2-m3) over (dialogue query, enriched track doc) pairs.
Never enlarges the pool (R7 owns recall) and never reorders below the slice. The scorer is injected
(`score_fn(pairs) -> list[float]`) so logic is testable GPU-free; production wires the cross-encoder.

Two integration modes:
- `rerank()` — final-stage: reorder the top slice by neural score (simple, leak-free at inference).
- `score()`  — stacking surface: {track_id -> neural score} for the top slice, fed back as a K1 feature.

OOF / leak note: an OFF-THE-SHELF (frozen) cross-encoder never sees the gold labels, so its score is
a fixed function of (query, doc) — stacking it into K1 is leak-free, exactly like the dense_cos
bi-encoder feature. Out-of-fold/cross-fit is required ONLY if the scorer is FINE-TUNED on Train
(spec §4 LoRA option), since then the model has seen the labels. So: off-the-shelf stacking = safe;
fine-tuned stacking = must be produced OOF (a separate harness, gated to the LoRA path).
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
                 max_doc_chars: int = 2000, model_revision: Optional[str] = None,
                 max_pairs_per_turn: Optional[int] = None) -> None:
        self.catalog = catalog
        self.qb = query_builder
        self.score_fn = score_fn
        self.cross_encoder_k = cross_encoder_k
        self.enriched = enriched
        self.max_doc_chars = max_doc_chars      # coarse doc-side cap (preserve the query side); the
        #                                         cross-encoder tokenizer truncates the doc further
        # provenance only (train==serve pin recorded by D1, spec §8); the model itself lives in score_fn
        self.model_revision = model_revision
        # per-turn cost budget (spec §8): raise if a turn would score more than this many pairs
        self.max_pairs_per_turn = max_pairs_per_turn

    def _doc(self, track_id: str) -> str:
        return self.catalog.id_to_metadata(track_id, enriched=self.enriched)[:self.max_doc_chars]

    def score(self, ctx: TurnContext, candidates: list[Candidate]) -> dict[str, float]:
        """{track_id -> neural score} for the top `cross_encoder_k` candidates (by incoming order)."""
        top = candidates[:self.cross_encoder_k]
        if not top:
            return {}
        if self.max_pairs_per_turn is not None and len(top) > self.max_pairs_per_turn:
            raise ValueError(f"K3 would score {len(top)} pairs/turn > budget {self.max_pairs_per_turn}")
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


def make_ce_feature_fn(lookup: dict, default: float = -1.0):
    """Adapter turning a precomputed CE-score lookup into a K1 FeatureBuilder score_fn.

    `lookup` maps (session_id, turn_number, track_id) -> CE score (typically within-pool
    normalized). Candidates the CE did not score (outside the top-`cross_encoder_k`) get `default`
    — the SAME sentinel at train and serve, so the feature is train==serve consistent and the GBDT
    can learn "not CE-scored" as its own signal. Leak-free when the lookup came from a frozen CE
    (no labels seen) or from OOF cross-fitting (see build_ce_score_lookup / oof_ce_scores)."""
    def _fn(ctx, track_id: str) -> float:
        return float(lookup.get((ctx.session_id, ctx.turn_number, track_id), default))
    return _fn


def build_ce_score_lookup(turns, pools, scorer, *, normalize: bool = True) -> dict:
    """Per-candidate CE-score lookup for K2 stacking, reusing the K3 `scorer.score` surface.

    `scorer` is anything with `.score(ctx, candidates) -> {track_id: ce_score}` (e.g. NeuralReranker,
    which scores only the top `cross_encoder_k`). `turns`/`pools` are aligned. With `normalize`, each
    turn's scores are min-max scaled within that turn's pool only (fold/scale-invariant, leak-free —
    uses no cross-turn or gold info). Returns {(session_id, turn_number, track_id): score}.

    Leak note: pass a FROZEN cross-encoder's scorer for no-OOF leak-free stacking; a FINE-TUNED CE
    must instead be cross-fit via oof_ce_scores (it has seen the labels)."""
    from mcrs.training.ce_data import normalize_within_pool

    lookup: dict = {}
    for ctx, pool in zip(turns, pools):
        d = scorer.score(ctx, pool)
        if not d:
            continue
        if normalize:
            d = normalize_within_pool(d)
        for tid, s in d.items():
            lookup[(ctx.session_id, ctx.turn_number, tid)] = float(s)
    return lookup


class ChainReranker:
    """Apply rerankers in sequence (e.g. K2 then K3), each consuming the prior's items. F2 Reranker."""
    label = "chain"

    def __init__(self, *rerankers) -> None:
        self.rerankers = rerankers

    def rerank(self, ctx: TurnContext, candidates: list[Candidate]) -> RankedList:
        items = list(candidates)
        for r in self.rerankers:
            items = r.rerank(ctx, items).items
        return RankedList(turn=ctx, items=items)
