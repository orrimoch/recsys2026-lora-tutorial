"""Tests for the ColBERT trainer's pure data-shaping + dev-eval metric (the
PyLate training loop + the live-model callback are GPU integration, run in the
notebook)."""
import pytest

from scripts.train_colbert import triples_to_contrastive_rows, wall_recall_at_k


class TestTriplesToContrastiveRows:
    def test_explodes_one_row_per_negative(self):
        # PyLate's Contrastive loss wants (query, positive, negative) triples with a
        # single negative per row; a builder triple carries K negatives, so explode.
        triples = [{"query": "q", "positive": "p", "negatives": ["n1", "n2"]}]
        rows = triples_to_contrastive_rows(triples)
        assert rows == [
            {"query": "q", "positive": "p", "negative": "n1"},
            {"query": "q", "positive": "p", "negative": "n2"},
        ]

    def test_skips_triples_with_no_negatives(self):
        # No negative -> can't form a contrastive pair -> drop (in-batch negs only
        # would need a different loss; v1 requires an explicit hard negative).
        triples = [
            {"query": "q1", "positive": "p1", "negatives": []},
            {"query": "q2", "positive": "p2", "negatives": ["n"]},
        ]
        rows = triples_to_contrastive_rows(triples)
        assert rows == [{"query": "q2", "positive": "p2", "negative": "n"}]

    def test_empty_input_yields_empty(self):
        assert triples_to_contrastive_rows([]) == []


class TestWallRecallAtK:
    def test_recall_over_wall_subset_only(self):
        # Row 1 is excluded (wall=False); rows 0 and 2 count. Gold in top-2 for both.
        ranked = [["g", "a"], ["a", "b"], ["c", "g2"]]
        golds = ["g", "b", "g2"]
        wall = [True, False, True]
        assert wall_recall_at_k(ranked, golds, wall, k=2) == pytest.approx(1.0)

    def test_partial_hit(self):
        ranked = [["x", "y"], ["g", "z"]]
        golds = ["g", "g"]
        wall = [True, True]
        # row0: g not in top-1 ['x'] -> 0; row1: g in top-1 ['g'] -> 1 => 0.5
        assert wall_recall_at_k(ranked, golds, wall, k=1) == pytest.approx(0.5)

    def test_no_wall_rows_returns_zero(self):
        # Safe sentinel so best-model tracking (max) never sees NaN.
        assert wall_recall_at_k([["a"]], ["g"], [False], k=1) == 0.0
