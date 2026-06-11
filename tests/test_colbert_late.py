"""Tests for the ColBERT late-interaction retrieval channel (colbert_late.py).

The pure scoring core (MaxSim) is tested here without any model/pylate dependency:
token embeddings are passed in directly. Model loading + encoding is exercised
as an integration step in the nb82 probe, not unit-tested.
"""
import numpy as np
import pytest

from mcrs.retrieval_modules.colbert_late import (
    DEFAULT_Q_LEN,
    ColbertRetriever,
    maxsim_score,
    rerank_pool,
    strip_track_id_prefix,
)


class TestMaxSimScore:
    def test_orthonormal_query_fully_covered_sums_to_token_count(self):
        # Two orthonormal query tokens, doc contains both → each query token's
        # best (max) similarity is 1.0 → MaxSim = 2.0. This is the defining
        # property: every facet matched independently, no averaging.
        q = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        d = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        assert maxsim_score(q, d) == pytest.approx(2.0)

    def test_doc_missing_a_facet_scores_lower(self):
        # Doc has only the first facet token. The second query token finds no
        # match (max sim 0) → MaxSim = 1.0 < 2.0. This is why a doc that fails
        # one facet (e.g. wrong era) is correctly penalized.
        q = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        d = np.array([[1.0, 0.0]], dtype=np.float32)
        assert maxsim_score(q, d) == pytest.approx(1.0)

    def test_each_query_token_takes_its_own_max(self):
        # Three doc tokens; each query token should independently pick its best.
        q = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        d = np.array([[0.9, 0.0], [0.0, 0.8], [0.5, 0.5]], dtype=np.float32)
        # token0 best = 0.9 (vs 0, 0.5); token1 best = 0.8 (vs 0, 0.5)
        assert maxsim_score(q, d) == pytest.approx(1.7)


class TestRerankPool:
    def test_ranks_doc_that_satisfies_every_facet_first(self):
        q = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        pool_embs = [
            np.array([[1.0, 0.0]], dtype=np.float32),               # tA: 1 facet only
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),   # tB: both facets
        ]
        ranked = rerank_pool(q, pool_embs, ["tA", "tB"], topk=2)
        assert ranked == ["tB", "tA"]

    def test_respects_topk_truncation(self):
        q = np.array([[1.0, 0.0]], dtype=np.float32)
        pool_embs = [
            np.array([[1.0, 0.0]], dtype=np.float32),   # tA score 1.0
            np.array([[0.5, 0.0]], dtype=np.float32),   # tB score 0.5
            np.array([[0.1, 0.0]], dtype=np.float32),   # tC score 0.1
        ]
        ranked = rerank_pool(q, pool_embs, ["tA", "tB", "tC"], topk=2)
        assert ranked == ["tA", "tB"]


# Document token embeddings keyed by track id (what scripts/build precomputes).
_DOC_EMBS = {
    "tA": np.array([[1.0, 0.0]], dtype=np.float32),                   # facet-0 only
    "tB": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),       # both facets
    "tC": np.array([[0.0, 1.0]], dtype=np.float32),                   # facet-1 only
}


def _fake_encoder(query_token_map):
    """Return a query-encoder callable that maps each query string to its
    preset token embeddings (stands in for the PyLate ColBERT model)."""
    def encode(queries):
        return [query_token_map[q] for q in queries]
    return encode


class TestColbertRetrieverPoolRerank:
    def test_reranks_each_querys_own_pool(self):
        # q1 wants both facets → tB first; q2 wants only facet-1 → tC over tA.
        q1 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        q2 = np.array([[0.0, 1.0]], dtype=np.float32)
        retriever = ColbertRetriever(
            doc_embs=_DOC_EMBS,
            query_encoder=_fake_encoder({"both": q1, "facet1": q2}),
        )
        out = retriever.batch_rerank_pool(
            queries=["both", "facet1"],
            pools=[["tA", "tB"], ["tA", "tC"]],
            topk=2,
        )
        assert out == [["tB", "tA"], ["tC", "tA"]]

    def test_encode_docs_populates_lookup_then_reranks(self):
        # A retriever can start empty, encode catalog docs once (doc encoder),
        # and then rerank against them — the build-then-probe flow nb82 uses.
        embA = np.array([[1.0, 0.0]], dtype=np.float32)
        embB = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        q = np.array([[0.0, 1.0]], dtype=np.float32)
        retriever = ColbertRetriever(
            doc_embs={},
            query_encoder=_fake_encoder({"q": q}),
            doc_encoder=lambda texts: [embA, embB],
        )
        retriever.encode_docs(["tA", "tB"], ["text a", "text b"])
        out = retriever.batch_rerank_pool(["q"], [["tA", "tB"]], topk=2)
        assert out == [["tB", "tA"]]  # q matches facet-1 → tB scores 1.0 > tA 0.0

    def test_skips_pool_tids_missing_from_doc_embs(self):
        # 'tZ' has no precomputed doc embedding → it cannot be scored and is
        # dropped rather than crashing or being ranked arbitrarily.
        q1 = np.array([[1.0, 0.0]], dtype=np.float32)
        retriever = ColbertRetriever(
            doc_embs=_DOC_EMBS,
            query_encoder=_fake_encoder({"q": q1}),
        )
        out = retriever.batch_rerank_pool(
            queries=["q"],
            pools=[["tZ", "tA"]],
            topk=5,
        )
        assert out == [["tA"]]


class TestStripTrackIdPrefix:
    def test_strips_leading_track_id_uuid_segment(self):
        # The catalog renderer prepends 'track_id: <uuid>, '; the UUID tokenizes to
        # ~30 content-free hex subwords that dilute MaxSim. Strip it for ColBERT docs.
        assert strip_track_id_prefix(
            "track_id: 97f5eeec-1ec7-4bb9-aa00, track_name: foo, artist_name: bar"
        ) == "track_name: foo, artist_name: bar"

    def test_leaves_non_prefixed_text_unchanged(self):
        assert strip_track_id_prefix("track_name: foo, artist_name: bar") == \
            "track_name: foo, artist_name: bar"

    def test_bare_track_id_with_no_fields_unchanged(self):
        # Degenerate (no metadata) — keep the id rather than return an empty doc.
        assert strip_track_id_prefix("track_id: 97f5eeec") == "track_id: 97f5eeec"


class TestDefaults:
    def test_default_query_length_covers_the_goal_facet(self):
        # Regression guard: q_len=32 right-truncated the 'goal:' facet off ~95% of
        # turn-1 queries (mean 54.8 tok). The default must keep the goal.
        assert DEFAULT_Q_LEN >= 96
