"""K2 — LightGBM LambdaMART reranker (F2 Reranker).

One group per (session, turn); label 1 for the gold, 0 for the other fused candidates (hard
in-pool negatives); objective lambdarank, eval ndcg@20. Groups with the gold not in the pool are
skipped in training (no fabricated positive). Consumes K1's Candidate.features (never recomputes).
See `.claude/documents/features/51_K2_lgbm_lambdamart.md`.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from mcrs.contracts import Candidate, RankedList, TurnContext
from mcrs.rerank.features import FeatureBuilder


class LGBMReranker:
    label = "lgbm"

    def __init__(self, feature_builder: FeatureBuilder, n_estimators: int = 100,
                 params: Optional[dict] = None) -> None:
        self.fb = feature_builder
        self.n_estimators = n_estimators
        self.params = params or {}
        self.model = None
        self._booster = None

    def build_training_data(self, groups):
        """groups = iterable of (TurnContext, candidates, gold_id). Returns (X, y, group_sizes),
        skipping groups whose gold is not in the candidate pool (all-negative groups)."""
        X, y, gsizes = [], [], []
        for ctx, cands, gold in groups:
            ids = {c.track_id for c in cands}
            if gold is None or gold not in ids:
                continue
            self.fb.build(ctx, cands)
            for c in cands:
                X.append([c.features[n] for n in self.fb.feature_names])
                y.append(1 if c.track_id == gold else 0)
            gsizes.append(len(cands))
        return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=int), gsizes

    def fit(self, groups) -> "LGBMReranker":
        import lightgbm as lgb

        X, y, gsizes = self.build_training_data(groups)
        params = dict(objective="lambdarank", metric="ndcg", n_estimators=self.n_estimators,
                      random_state=42, verbosity=-1, num_leaves=15, min_child_samples=5)
        params.update(self.params)
        self.model = lgb.LGBMRanker(**params)
        self.model.fit(X, y, group=gsizes)
        self._booster = self.model.booster_
        return self

    def rerank(self, ctx: TurnContext, candidates: list[Candidate]) -> RankedList:
        if self._booster is None:
            raise RuntimeError("LGBMReranker.rerank called before fit()/load()")
        self.fb.build(ctx, candidates)
        scores = self._booster.predict(self.fb.matrix(candidates))
        order = np.argsort(-scores, kind="stable")
        return RankedList(turn=ctx, items=[candidates[int(i)] for i in order])

    def save(self, path: str) -> None:
        if self._booster is None:
            raise RuntimeError("nothing to save: fit() first")
        self._booster.save_model(path)

    def load(self, path: str) -> "LGBMReranker":
        import lightgbm as lgb
        self._booster = lgb.Booster(model_file=path)
        return self
