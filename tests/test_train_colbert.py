"""Tests for the ColBERT trainer's pure data-shaping (the PyLate training loop
itself is GPU integration, run in the notebook)."""
from scripts.train_colbert import triples_to_contrastive_rows


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
