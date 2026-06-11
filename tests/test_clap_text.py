"""Tests for the CLAP text->audio recall channel (clap_text.py).

The pure cosine top-k core and the retriever orchestration are tested with injected
encoders/embeddings — no transformers/laion dependency. The CLAP text tower
(laion/larger_clap_music) is loaded lazily in production only.
"""
import numpy as np

from mcrs.retrieval_modules.clap_text import ClapTextRetriever, cosine_topk


class TestCosineTopk:
    def test_ranks_nearest_track_first(self):
        q = np.array([[1.0, 0.0]], dtype=np.float32)
        tm = np.array([[0.0, 1.0], [1.0, 0.0], [0.7, 0.7]], dtype=np.float32)
        assert cosine_topk(q, tm, ["tA", "tB", "tC"], 2) == [["tB", "tC"]]

    def test_topk_larger_than_catalog(self):
        q = np.array([[1.0, 0.0]], dtype=np.float32)
        tm = np.array([[1.0, 0.0]], dtype=np.float32)
        assert cosine_topk(q, tm, ["tA"], 5) == [["tA"]]

    def test_per_query_independent(self):
        q = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        tm = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        assert cosine_topk(q, tm, ["tA", "tB"], 1) == [["tA"], ["tB"]]


class TestClapTextRetriever:
    def test_retrieves_via_injected_text_encoder(self):
        # Cold-firable: the query TEXT is encoded into the CLAP audio space and
        # matched against catalog audio — no played history needed (fires on turn-1).
        tids = ["tA", "tB"]
        mat = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        enc = lambda qs: np.array([[0.0, 1.0]], dtype=np.float32)  # query -> sounds like tB
        r = ClapTextRetriever(text_encoder=enc, track_ids=tids, track_mat=mat)
        assert r.batch_text_to_item_retrieval(["hard-hitting beat"], topk=2) == [["tB", "tA"]]

    def test_empty_catalog_returns_empty_lists(self):
        r = ClapTextRetriever(
            text_encoder=lambda qs: np.zeros((1, 2), dtype=np.float32),
            track_ids=[], track_mat=np.zeros((0, 2), dtype=np.float32),
        )
        assert r.batch_text_to_item_retrieval(["q"], topk=5) == [[]]
