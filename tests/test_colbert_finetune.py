"""Stage 2 — ColBERT contrastive fine-tune: pure data-shaping + dev-eval metric.

Ports `scripts/train_colbert.py` (recall-union-lgbm) pure functions onto fresh-start. The PyLate
training loop and the live-model dev-eval callback are GPU integration (run on Colab, not unit
tested). The feedback stays the original way: Contrastive loss + turn-1 dev-recall selection.
"""
from __future__ import annotations

import numpy as np
import pytest

from mcrs.training.colbert_finetune import (
    existing_artifact_blocks,
    rerank_pool,
    triples_to_contrastive_rows,
    update_best_state,
    wall_recall_at_k,
)


def _fresh():
    return {"best": -1.0, "since_improve": 0}


class TestUpdateBestState:
    def test_first_eval_is_best_no_stop(self):
        st = _fresh()
        is_best, stop = update_best_state(0.33, st, patience=2)
        assert is_best is True and stop is False
        assert st["best"] == 0.33 and st["since_improve"] == 0

    def test_improvement_resets_counter(self):
        st = _fresh()
        update_best_state(0.30, st, 2)
        update_best_state(0.25, st, 2)            # 1 since best
        is_best, stop = update_best_state(0.40, st, 2)  # new best -> reset
        assert is_best is True and stop is False and st["since_improve"] == 0

    def test_stops_after_patience_consecutive_non_improvements(self):
        # mirrors the observed curve: 0.33 peak, then 0.31, 0.30 -> stop at patience=2
        st = _fresh()
        update_best_state(0.33, st, 2)            # best
        _, s1 = update_best_state(0.31, st, 2)    # 1 since best
        _, s2 = update_best_state(0.30, st, 2)    # 2 since best -> stop
        assert s1 is False and s2 is True
        assert st["best"] == 0.33                 # best preserved

    def test_patience_zero_never_stops(self):
        st = _fresh()
        update_best_state(0.33, st, 0)
        for v in (0.2, 0.1, 0.05):
            _, stop = update_best_state(v, st, 0)
            assert stop is False

    def test_tie_is_not_best(self):
        st = _fresh()
        update_best_state(0.33, st, 5)
        is_best, _ = update_best_state(0.33, st, 5)   # equal -> not a new best (>, ties keep earlier)
        assert is_best is False and st["since_improve"] == 1


class TestTriplesToContrastiveRows:
    def test_explodes_one_row_per_negative(self):
        triples = [{"query": "q", "positive": "p", "negatives": ["n1", "n2"]}]
        assert triples_to_contrastive_rows(triples) == [
            {"query": "q", "positive": "p", "negative": "n1"},
            {"query": "q", "positive": "p", "negative": "n2"},
        ]

    def test_skips_triples_with_no_negatives(self):
        triples = [
            {"query": "q1", "positive": "p1", "negatives": []},
            {"query": "q2", "positive": "p2", "negatives": ["n"]},
        ]
        assert triples_to_contrastive_rows(triples) == [
            {"query": "q2", "positive": "p2", "negative": "n"}]

    def test_empty_input_yields_empty(self):
        assert triples_to_contrastive_rows([]) == []


class TestWallRecallAtK:
    def test_recall_over_wall_subset_only(self):
        ranked = [["g", "a"], ["a", "b"], ["c", "g2"]]
        golds = ["g", "b", "g2"]
        wall = [True, False, True]
        assert wall_recall_at_k(ranked, golds, wall, k=2) == pytest.approx(1.0)

    def test_partial_hit(self):
        ranked = [["x", "y"], ["g", "z"]]
        golds = ["g", "g"]
        wall = [True, True]
        assert wall_recall_at_k(ranked, golds, wall, k=1) == pytest.approx(0.5)

    def test_no_wall_rows_returns_zero(self):
        assert wall_recall_at_k([["a"]], ["g"], [False], k=1) == 0.0


class TestExistingArtifactBlocks:
    def test_blocks_when_exists_and_not_forced(self, tmp_path):
        d = tmp_path / "model"; d.mkdir()
        assert existing_artifact_blocks(str(d), force=False) is True

    def test_does_not_block_when_forced(self, tmp_path):
        d = tmp_path / "model"; d.mkdir()
        assert existing_artifact_blocks(str(d), force=True) is False

    def test_does_not_block_when_missing(self, tmp_path):
        assert existing_artifact_blocks(str(tmp_path / "nope"), force=False) is False

    def test_empty_path_does_not_block(self):
        assert existing_artifact_blocks("", force=False) is False


class TestRerankPool:
    def test_orders_pool_by_maxsim_desc(self):
        q = np.array([[1.0, 0.0]])
        embs = [np.array([[1.0, 0.0]]),   # A score 1.0
                np.array([[0.0, 1.0]]),   # B score 0.0
                np.array([[0.5, 0.0]])]   # C score 0.5
        assert rerank_pool(q, embs, ["A", "B", "C"], topk=2) == ["A", "C"]

    def test_topk_truncates(self):
        q = np.array([[1.0, 0.0]])
        embs = [np.array([[float(i), 0.0]]) for i in (3, 1, 2)]
        assert rerank_pool(q, embs, ["A", "B", "C"], topk=1) == ["A"]

    def test_stable_tie_break_by_input_order(self):
        q = np.array([[1.0, 0.0]])
        embs = [np.array([[1.0, 0.0]]), np.array([[1.0, 0.0]])]  # equal score
        assert rerank_pool(q, embs, ["A", "B"], topk=2) == ["A", "B"]
