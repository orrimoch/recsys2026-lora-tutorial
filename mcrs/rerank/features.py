"""K1 — rerank feature builder.

Fills Candidate.features from causal, pure functions of TurnContext + Candidate + catalog.
The single place features are computed (K2/K3 consume, never recompute). v1 features are all
non-model-derived, so leak-free by construction; any future model-score feature (SASRec/CE) must
be OOF/cross-fit. See `.claude/documents/features/50_K1_rerank_feature_builder.md`.
"""
from __future__ import annotations

import math
import re
from typing import Callable, Optional

from mcrs.contracts import Candidate, TurnContext


def _first(v, default=0.0):
    if isinstance(v, list):
        return v[0] if v else default
    return default if v is None else v


def _year(release_date) -> float:
    m = re.search(r"\d{4}", str(release_date or ""))
    return float(m.group()) if m else 0.0


def _artists(meta: dict) -> set:
    a = meta.get("artist_name") or []
    return set(a if isinstance(a, list) else [a])


class FeatureBuilder:
    def __init__(self, catalog, channel_labels: list[str],
                 score_fns: Optional[dict[str, Callable[[TurnContext, str], float]]] = None) -> None:
        self.catalog = catalog
        self.channel_labels = list(channel_labels)
        # injected per-candidate relevance scorers (e.g. dense query<->doc cosine, bm25 score).
        # Each fn(ctx, track_id) -> float; computed once here so K2/K3 consume, never recompute.
        # Not label-derived (fixed encoder/index) -> leak-free; a TRAINED score would need OOF.
        self.score_fns = dict(score_fns or {})
        self.score_names = sorted(self.score_fns)
        self.feature_names = (
            ["rrf_score", "n_channels_hit", "best_rank_inv"]
            + [f"rank_inv__{l}" for l in self.channel_labels]
            + ["turn_number", "history_len", "is_cold", "query_len",
               "log_popularity", "release_year", "artist_in_history"]
            + self.score_names
        )

    def build(self, ctx: TurnContext, candidates: list[Candidate]) -> list[Candidate]:
        hist_artists: set = set()
        for h in ctx.history_tids:
            if self.catalog is not None and h in self.catalog:
                hist_artists |= _artists(self.catalog.metadata(h))
        query_len = float(sum(len(u.split()) for u in ctx.utterances))
        is_cold = 1.0 if ctx.segment == "cold" else 0.0

        for c in candidates:
            ranks = c.channel_ranks
            meta = self.catalog.metadata(c.track_id) if (
                self.catalog is not None and c.track_id in self.catalog) else {}
            f = {
                "rrf_score": float(c.rrf_score),
                "n_channels_hit": float(len(ranks)),
                "best_rank_inv": max((1.0 / r for r in ranks.values()), default=0.0),
                "turn_number": float(ctx.turn_number),
                "history_len": float(len(ctx.history_tids)),
                "is_cold": is_cold,
                "query_len": query_len,
                "log_popularity": math.log1p(float(_first(meta.get("popularity"), 0.0))),
                "release_year": _year(meta.get("release_date")),
                "artist_in_history": 1.0 if (_artists(meta) & hist_artists) else 0.0,
            }
            for l in self.channel_labels:
                f[f"rank_inv__{l}"] = (1.0 / ranks[l]) if l in ranks else 0.0
            for name in self.score_names:
                f[name] = float(self.score_fns[name](ctx, c.track_id))
            c.features = f
        return candidates

    def matrix(self, candidates: list[Candidate]):
        """Stack features into a (n, n_features) array in feature_names order (for K2)."""
        import numpy as np
        return np.asarray(
            [[c.features[name] for name in self.feature_names] for c in candidates],
            dtype=np.float32,
        )
