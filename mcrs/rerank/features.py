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
        # T3.4: precompute the catalog popularity distribution once for percentile lookup (debias —
        # lets K2 down-weight popularity except when the query matches). None when no catalog (tests).
        self._pop_sorted = None
        ids = getattr(catalog, "index_to_id", None) if catalog is not None else None
        if ids:
            try:
                self._pop_sorted = sorted(
                    float(_first(catalog.metadata(t).get("popularity"), 0.0)) for t in ids)
            except Exception:
                self._pop_sorted = None
        self.feature_names = (
            ["rrf_score", "n_channels_hit", "best_rank_inv"]
            + [f"rank_inv__{l}" for l in self.channel_labels]
            + ["turn_number", "history_len", "is_cold", "query_len",
               "log_popularity", "release_year", "artist_in_history"]
            # T3.1 interaction/consensus (multi-view agreement — leak-free, built from existing ranks)
            + ["n_channels_top10", "consensus_3plus", "top5_bm25_and_dense"]
            # T3.4 popularity-debias + recency
            + ["popularity_percentile", "recency"]
            # session/behavioral: in-session replay + artist affinity (causal, leak-free)
            + ["is_replay", "artist_play_count", "last_artist_match", "artist_recency"]
            # chat-derived: does the dialogue name the candidate's artist/track (causal, leak-free)
            + ["artist_mentioned_in_chat", "track_mentioned_in_chat"]
            + self.score_names
            # T3.1 per-turn min-max calibration of each injected score (scale-comparable across turns)
            + [f"{n}_norm" for n in self.score_names]
        )

    def _pop_percentile(self, pop: float) -> float:
        """Fraction of catalog tracks no more popular than `pop` (0..1). 0.5 when no catalog dist."""
        arr = self._pop_sorted
        if not arr:
            return 0.5
        import bisect
        return bisect.bisect_right(arr, pop) / len(arr)

    @staticmethod
    def _chat_text(ctx: TurnContext) -> str:
        """Lowercased concatenation of the causal dialogue (utterances 1..t) + goal text."""
        parts = list(ctx.utterances) + ([ctx.goal] if ctx.goal else [])
        return " ".join(parts).lower()

    def build(self, ctx: TurnContext, candidates: list[Candidate]) -> list[Candidate]:
        hist_artists: set = set()
        for h in ctx.history_tids:
            if self.catalog is not None and h in self.catalog:
                hist_artists |= _artists(self.catalog.metadata(h))
        # per-position history artist sets (for play-count / recency / last-play affinity) + replay set
        hist_set = set(ctx.history_tids)
        hist_artist_sets = [
            (_artists(self.catalog.metadata(h)) if (self.catalog is not None and h in self.catalog) else set())
            for h in ctx.history_tids
        ]
        last_artists = hist_artist_sets[-1] if hist_artist_sets else set()
        chat = self._chat_text(ctx)
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
            # T3.1 interaction/consensus: agreement across channels is a strong relevance signal that
            # per-channel rank_inv alone can't express (a tree can't AND two columns without a split).
            n_top10 = sum(1 for r in ranks.values() if r <= 10)
            f["n_channels_top10"] = float(n_top10)
            f["consensus_3plus"] = 1.0 if n_top10 >= 3 else 0.0
            f["top5_bm25_and_dense"] = 1.0 if (ranks.get("bm25", 10**9) <= 5
                                               and ranks.get("dense", 10**9) <= 5) else 0.0
            # T3.4 popularity-debias + recency
            f["popularity_percentile"] = self._pop_percentile(float(_first(meta.get("popularity"), 0.0)))
            f["recency"] = min(1.0, max(0.0, 1.0 - (2026.0 - f["release_year"]) / 30.0)) if f["release_year"] > 0 else 0.0
            # session/behavioral: in-session replay + artist affinity (per-candidate, causal)
            cand_artists = _artists(meta)
            f["is_replay"] = 1.0 if c.track_id in hist_set else 0.0
            f["artist_play_count"] = float(sum(1 for hs in hist_artist_sets if hs & cand_artists))
            f["last_artist_match"] = 1.0 if (cand_artists & last_artists) else 0.0
            rec = 0.0
            for dist, hs in enumerate(reversed(hist_artist_sets)):
                if hs & cand_artists:
                    rec = 1.0 / (1.0 + dist)
                    break
            f["artist_recency"] = rec
            # chat-derived: does the dialogue name the candidate's artist / track (min-len 3 guard)
            cand_track_names = meta.get("track_name") or []
            cand_track_names = cand_track_names if isinstance(cand_track_names, list) else [cand_track_names]
            f["artist_mentioned_in_chat"] = 1.0 if any(
                len(a) >= 3 and a.lower() in chat for a in cand_artists) else 0.0
            f["track_mentioned_in_chat"] = 1.0 if any(
                len(str(t)) >= 3 and str(t).lower() in chat for t in cand_track_names) else 0.0
            for name in self.score_names:
                f[name] = float(self.score_fns[name](ctx, c.track_id))
            c.features = f
        # T3.1 per-turn calibration: min-max each injected score WITHIN this turn's pool so magnitudes
        # are comparable across turns (query length/specificity shifts the raw scale otherwise).
        for name in self.score_names:
            vals = [c.features[name] for c in candidates]
            lo, hi = (min(vals), max(vals)) if vals else (0.0, 0.0)
            rng = (hi - lo) or 1.0
            for c in candidates:
                c.features[f"{name}_norm"] = (c.features[name] - lo) / rng
        return candidates

    def matrix(self, candidates: list[Candidate]):
        """Stack features into a (n, n_features) array in feature_names order (for K2)."""
        import numpy as np
        return np.asarray(
            [[c.features[name] for name in self.feature_names] for c in candidates],
            dtype=np.float32,
        )
