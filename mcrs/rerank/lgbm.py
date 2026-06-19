"""K2 — LightGBM LambdaMART reranker (F2 Reranker).

One group per (session, turn); label 1 for the gold, 0 for the other fused candidates (hard
in-pool negatives); objective lambdarank, eval ndcg@20. Anti-overfit: session-disjoint train/val
split + early stopping on the val set, and a configurable negative cap (keep gold + top-N pool
negatives). Groups with the gold not in the pool are skipped (no fabricated positive). Consumes
K1's Candidate.features. See `.claude/documents/features/51_K2_lgbm_lambdamart.md`.
"""
from __future__ import annotations

import json
import os
import random
from typing import Optional

import numpy as np

from mcrs.contracts import Candidate, RankedList, TurnContext
from mcrs.rerank.features import FeatureBuilder


class LGBMReranker:
    label = "lgbm"

    def __init__(self, feature_builder: FeatureBuilder, n_estimators: int = 100,
                 params: Optional[dict] = None, neg_cap: int = 0,
                 val_fraction: float = 0.1, early_stopping_rounds: int = 0,
                 min_val_groups: int = 50, seed: int = 42) -> None:
        # early_stopping_rounds=0 => OFF (plain fit on all groups). Enable only when the val
        # signal is reliable (full-scale data); on small/weak setups it underfits (best_iter≈1).
        self.fb = feature_builder
        self.n_estimators = n_estimators
        self.params = params or {}
        self.neg_cap = neg_cap
        self.val_fraction = val_fraction
        self.early_stopping_rounds = early_stopping_rounds
        self.min_val_groups = min_val_groups
        self.seed = seed
        self.model = None
        self._booster = None
        self.n_train_groups_ = 0
        self.n_val_groups_ = 0
        # feature spec the model was trained on; persisted on save, checked at rerank (serve guard)
        self.feature_names_: Optional[list[str]] = None

    # ---- group / feature assembly ----
    def _cap(self, cands: list[Candidate], gold: str, salt=None) -> list[Candidate]:
        """Keep the gold + a RANDOM sample of neg_cap negatives.

        Random (not top-N): keeping only the highest-ranked negatives biases them to higher
        rrf_score than the gold, which teaches the inverted 'high score => not gold' correlation
        and wrecks ranking. Sampling preserves the pool's score distribution.

        `salt` (per-group, e.g. (session_id, turn_number)) is mixed into the seed so each group
        draws an INDEPENDENT-but-deterministic sample. Reusing one fixed seed across all groups
        makes the sampled positions correlated with the (fusion-ordered) pool — a biased subset
        that doesn't match the full-pool eval/serve distribution.
        """
        if not self.neg_cap:
            return cands
        pos = [c for c in cands if c.track_id == gold]
        negs = [c for c in cands if c.track_id != gold]
        if len(negs) > self.neg_cap:
            negs = random.Random(f"{self.seed}|{salt}").sample(negs, self.neg_cap)
        return pos + negs

    @staticmethod
    def _has_gold(cands, gold) -> bool:
        return gold is not None and gold in {c.track_id for c in cands}

    def _xy(self, groups, cap: bool = True):
        # cap negatives for TRAINING efficiency only; eval/val uses the full pool (what serve ranks)
        X, y, gsizes = [], [], []
        for ctx, cands, gold in groups:
            cc = self._cap(cands, gold, salt=(ctx.session_id, ctx.turn_number)) if cap else list(cands)
            self.fb.build(ctx, cc)
            for c in cc:
                X.append([c.features[n] for n in self.fb.feature_names])
                y.append(1 if c.track_id == gold else 0)
            gsizes.append(len(cc))
        return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=int), gsizes

    def build_training_data(self, groups):
        """Returns (X, y, group_sizes), skipping gold-not-in-pool groups, applying neg_cap."""
        kept = [(c, cs, g) for c, cs, g in groups if self._has_gold(cs, g)]
        return self._xy(kept)

    def _session_split(self, groups):
        """Split kept groups into (train, val) so a session never spans both."""
        kept = [(c, cs, g) for c, cs, g in groups if self._has_gold(cs, g)]
        sessions = sorted({ctx.session_id for ctx, _, _ in kept})
        rng = random.Random(self.seed)
        rng.shuffle(sessions)
        n_val = int(len(sessions) * self.val_fraction)
        val_sessions = set(sessions[:n_val])
        train = [g for g in kept if g[0].session_id not in val_sessions]
        val = [g for g in kept if g[0].session_id in val_sessions]
        return train, val

    # ---- fit / predict ----
    def fit(self, groups) -> "LGBMReranker":
        import lightgbm as lgb

        params = dict(objective="lambdarank", metric="ndcg", n_estimators=self.n_estimators,
                      random_state=self.seed, verbosity=-1, num_leaves=15, min_child_samples=5)
        params.update(self.params)
        self.model = lgb.LGBMRanker(**params)
        self.feature_names_ = list(self.fb.feature_names)   # pin the trained feature spec

        # Early stopping (opt-in): session-disjoint val + stop on val ndcg@20.
        if self.early_stopping_rounds and self.val_fraction > 0:
            train_groups, val_groups = self._session_split(groups)
            if len(val_groups) >= self.min_val_groups:
                Xtr, ytr, gtr = self._xy(train_groups, cap=True)
                Xva, yva, gva = self._xy(val_groups, cap=False)  # eval on the full pool
                self.n_train_groups_, self.n_val_groups_ = len(gtr), len(gva)
                self.model.fit(Xtr, ytr, group=gtr, eval_set=[(Xva, yva)], eval_group=[gva],
                               eval_at=[20], callbacks=[
                                   lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                                   lgb.log_evaluation(0)])
                self._booster = self.model.booster_
                return self

        # Default: plain fit on all kept groups (neg_cap applied), no early stopping.
        X, y, g = self.build_training_data(groups)
        self.n_train_groups_, self.n_val_groups_ = len(g), 0
        self.model.fit(X, y, group=g)
        self._booster = self.model.booster_
        return self

    def rerank(self, ctx: TurnContext, candidates: list[Candidate]) -> RankedList:
        if self._booster is None:
            raise RuntimeError("LGBMReranker.rerank called before fit()/load()")
        if self.feature_names_ is not None and list(self.fb.feature_names) != self.feature_names_:
            raise ValueError(
                "feature spec mismatch: this FeatureBuilder differs from the trained model's. "
                f"trained on {len(self.feature_names_)} features, got {len(self.fb.feature_names)}. "
                "Reconstruct FeatureBuilder with the SAME channel_labels and score_fns (train==serve).")
        self.fb.build(ctx, candidates)
        scores = self._booster.predict(self.fb.matrix(candidates))
        order = np.argsort(-scores, kind="stable")
        return RankedList(turn=ctx, items=[candidates[int(i)] for i in order])

    def save(self, path: str) -> None:
        if self._booster is None:
            raise RuntimeError("nothing to save: fit() first")
        self._booster.save_model(path)
        if self.feature_names_ is not None:           # sidecar feature spec for the serve guard
            with open(path + ".features.json", "w") as f:
                json.dump(self.feature_names_, f)

    def load(self, path: str) -> "LGBMReranker":
        import lightgbm as lgb
        self._booster = lgb.Booster(model_file=path)
        sidecar = path + ".features.json"
        if os.path.exists(sidecar):
            with open(sidecar) as f:
                self.feature_names_ = json.load(f)
        return self
