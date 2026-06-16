"""L1 — filtering & top-20 assembly (F2 Filter).

Turns a reranked RankedList into <=20 unique valid canonical track_ids WITHOUT touching the
catalog universe (post-retrieval pruning only). Dedup, history-rule A/B, validity guard, backfill
(by keeping all cleaned candidates then slicing), and optional tail-MMR confined to positions 11-20
(top-10 stays relevance-ranked). See `.claude/documents/features/60_L1_filter_assembly.md`.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from mcrs.contracts import RankedList
from mcrs.data.ids import canonical_track_id


class TopKAssembler:
    def __init__(self, catalog=None, top_k: int = 20, history_rule: str = "off",
                 tail_mmr: bool = False, track_vector: Optional[Callable[[str], np.ndarray]] = None,
                 mmr_lambda: float = 0.5, tail_start: int = 10) -> None:
        self.catalog = catalog
        self.top_k = top_k
        self.history_rule = history_rule
        self.tail_mmr = tail_mmr
        self.track_vector = track_vector
        self.mmr_lambda = mmr_lambda
        self.tail_start = tail_start

    def apply(self, ranked: RankedList) -> list[str]:
        hist = set(ranked.turn.history_tids) if self.history_rule == "on" else set()
        cleaned: list[str] = []
        seen: set[str] = set()
        for c in ranked.items:
            tid = canonical_track_id(c.track_id)
            if tid in seen or tid in hist:
                continue
            if self.catalog is not None and tid not in self.catalog:
                continue
            seen.add(tid)
            cleaned.append(tid)
        if self.tail_mmr and self.track_vector is not None:
            cleaned = self._tail_mmr(cleaned)
        return cleaned[: self.top_k]

    def _tail_mmr(self, ids: list[str]) -> list[str]:
        """Greedy MMR over positions tail_start..top_k; head stays relevance-ranked."""
        head, tail = ids[: self.tail_start], ids[self.tail_start:]
        if len(tail) <= 1:
            return ids
        rel = {t: 1.0 / (i + 1) for i, t in enumerate(tail)}  # relevance proxy = pool rank
        vecs = {t: self._norm(self.track_vector(t)) for t in tail}
        selected = list(head)
        chosen: list[str] = []
        pool = list(tail)
        while pool:
            best, best_score = None, -1e9
            for t in pool:
                max_sim = max((float(vecs[t] @ vecs[s]) for s in chosen), default=0.0) \
                    if chosen else 0.0
                score = self.mmr_lambda * rel[t] - (1 - self.mmr_lambda) * max_sim
                if score > best_score:
                    best, best_score = t, score
            chosen.append(best)
            pool.remove(best)
        return head + chosen

    @staticmethod
    def _norm(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32)
        n = np.linalg.norm(v)
        return v / n if n else v
