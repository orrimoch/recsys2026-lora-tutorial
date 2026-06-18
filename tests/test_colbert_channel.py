"""R8 — ColBERT late-interaction channel (brute-force MaxSim).

Hermetic: token encoders are injected (no `pylate`, no GPU). Mirrors the R4 dense-channel
test style. Covers feature doc 47_R8 §7: MaxSim correctness, ranking, aspect independence,
topk, canonical ids, content-only (no-leak), determinism, doc-cap counter.
"""
from __future__ import annotations

import numpy as np

from mcrs.data.catalog import Catalog
from mcrs.retrieval.colbert_channel import (
    ColBERTChannel,
    colbert_doc_text,
    count_docs_over_budget,
    maxsim_score,
)

# A 3-dim "aspect" space: e0="rock", e1="jazz", e2="other".
_DOC_TOKENS = {
    "t1": np.array([[1.0, 0.0, 0.0]], dtype=np.float32),                 # rock
    "t2": np.array([[0.0, 1.0, 0.0]], dtype=np.float32),                 # jazz
    "t3": np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),  # rock + jazz
    "t4": np.array([[0.0, 0.0, 1.0]], dtype=np.float32),                 # other
}
_INDEX = ["t1", "t2", "t3", "t4"]
_MATRIX = [_DOC_TOKENS[t] for t in _INDEX]

_QTOK = {
    "rock": np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
    "jazz": np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
    "rock jazz": np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
}


def _fake_q_encoder(mapping):
    def enc(queries):
        return [mapping[q] for q in queries]
    return enc


def _channel(index=None, matrix=None, **kw):
    return ColBERTChannel(index or _INDEX, matrix or _MATRIX,
                          _fake_q_encoder(_QTOK), **kw)


# ----- pure MaxSim core -----
def test_maxsim_hand_computed():
    q = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    d_both = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
    d_one = np.array([[1.0, 0.0]], dtype=np.float32)
    # q0 best=1, q1 best=1 -> 2.0 ; vs q0=1, q1=0 -> 1.0
    assert maxsim_score(q, d_both) == 2.0
    assert maxsim_score(q, d_one) == 1.0


# ----- ranking behaviour -----
def test_single_aspect_query_ranks_matching_doc_first():
    out = _channel().batch_text_to_item_retrieval(["jazz"], topk=4)
    assert out[0][0] in ("t2", "t3")          # both match jazz with score 1
    assert out[0][-1] == "t4"                  # 'other' matches nothing -> last


def test_multi_aspect_query_surfaces_both_aspects():
    # 'rock jazz': t3 matches both (score 2) > t1/t2 (score 1) > t4 (0).
    out = _channel().batch_text_to_item_retrieval(["rock jazz"], topk=4)
    assert out[0][0] == "t3"                    # aspect independence wins
    assert "t2" in out[0][:3]                   # secondary (non-"rock") aspect surfaced
    assert out[0][-1] == "t4"


def test_topk_truncates():
    out = _channel().batch_text_to_item_retrieval(["rock jazz"], topk=2)
    assert len(out[0]) == 2 and out[0][0] == "t3"


def test_batch_of_queries():
    out = _channel().batch_text_to_item_retrieval(["rock", "jazz"], topk=1)
    assert out[0][0] in ("t1", "t3") and out[1][0] in ("t2", "t3")


# ----- canonical ids -----
def test_canonicalizes_ids():
    ch = _channel(index=["track_id: t1", "t2", "t3", "t4"])
    out = ch.batch_text_to_item_retrieval(["rock"], topk=4)
    assert out[0][0] == "t1"                    # 'track_id:' prefix stripped


# ----- content-only / no-leak -----
def test_ignores_batch_context_and_user_ids():
    ch = _channel()
    base = ch.batch_text_to_item_retrieval(["rock jazz"], topk=4)
    leaked = ch.batch_text_to_item_retrieval(
        ["rock jazz"], topk=4,
        batch_context=[{"segment": "warm", "gold": "t4"}], user_ids=["u1"],
    )
    assert base == leaked


# ----- determinism / tie-break -----
def test_ties_broken_by_ascending_index():
    # 'rock' scores t1 and t3 equally (1.0); ascending index => t1 before t3.
    out = _channel().batch_text_to_item_retrieval(["rock"], topk=4)
    assert out[0].index("t1") < out[0].index("t3")


def test_deterministic_repeated_calls():
    ch = _channel()
    a = ch.batch_text_to_item_retrieval(["rock jazz", "jazz"], topk=4)
    b = ch.batch_text_to_item_retrieval(["rock jazz", "jazz"], topk=4)
    assert a == b


def test_chunking_matches_unchunked():
    full = _channel().batch_text_to_item_retrieval(["rock jazz", "jazz", "rock"], topk=4)
    chunked = _channel(chunk_docs=1).batch_text_to_item_retrieval(
        ["rock jazz", "jazz", "rock"], topk=4)
    assert full == chunked


# ----- L2 normalisation (cosine, not raw dot) -----
def test_l2_normalizes_token_magnitude():
    # Same direction, larger magnitude must NOT win once normalised.
    index = ["small", "big"]
    matrix = [np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
              np.array([[9.0, 0.0, 0.0]], dtype=np.float32)]
    out = ColBERTChannel(index, matrix, _fake_q_encoder(_QTOK)
                         ).batch_text_to_item_retrieval(["rock"], topk=2)
    assert out[0][0] == "small"                 # tie -> ascending index, not magnitude


# ----- edges -----
def test_empty_queries_returns_empty():
    assert _channel().batch_text_to_item_retrieval([], topk=4) == []


def test_label_default():
    assert _channel().label == "colbert"


# ----- doc-text helpers -----
def test_colbert_doc_text_reads_enriched_and_strips_prefix():
    cat = Catalog(
        [{"track_id": "x", "track_name": ["Song"], "artist_name": ["A"]}],
        enriched_docs={"x": "track_id: x, upbeat 90s grunge tags: rock"},
    )
    assert colbert_doc_text(cat, "x") == "upbeat 90s grunge tags: rock"


def test_colbert_doc_text_no_prefix_is_noop():
    cat = Catalog(
        [{"track_id": "y", "track_name": ["S"], "artist_name": ["B"]}],
        enriched_docs={"y": "upbeat synthpop tags: pop"},
    )
    assert colbert_doc_text(cat, "y") == "upbeat synthpop tags: pop"


def test_colbert_doc_text_expansion_first_reorders():
    # A1 builds 'base (metadata+tags) | doc2query-expansion'; expansion (high-value, R8 §4.3) is
    # LAST, so right-truncation drops it. expansion_first swaps it to the front so it survives.
    cat = Catalog(
        [{"track_id": "x", "track_name": ["Song"], "artist_name": ["A"]}],
        enriched_docs={"x": "name: Song, tags: rock | upbeat 90s workout anthem"},
    )
    assert colbert_doc_text(cat, "x") == "name: Song, tags: rock | upbeat 90s workout anthem"
    assert colbert_doc_text(cat, "x", expansion_first=True) == \
        "upbeat 90s workout anthem | name: Song, tags: rock"


def test_colbert_doc_text_expansion_first_noop_without_separator():
    cat = Catalog(
        [{"track_id": "y", "track_name": ["S"], "artist_name": ["B"]}],
        enriched_docs={"y": "name: S, tags: pop"},   # no ' | ' expansion
    )
    assert colbert_doc_text(cat, "y", expansion_first=True) == "name: S, tags: pop"


def test_count_docs_over_budget():
    texts = ["a b c", "a b c d e", "x"]
    assert count_docs_over_budget(texts, budget=4) == 1     # only the 5-token doc
    assert count_docs_over_budget(texts, budget=10) == 0


# ----- from_catalog builder (production wiring, encoders injected) -----
def _aspect_doc_encoder(text):
    """Tiny stand-in for a ColBERT doc encoder: one token per known aspect word present."""
    aspects = {"rock": [1.0, 0.0, 0.0], "jazz": [0.0, 1.0, 0.0], "other": [0.0, 0.0, 1.0]}
    toks = [aspects[w] for w in text.split() if w in aspects] or [[0.0, 0.0, 0.0]]
    return np.array(toks, dtype=np.float32)


def _build_catalog():
    rows = [
        {"track_id": "track_id: r", "track_name": ["R"], "artist_name": ["A"]},
        {"track_id": "j", "track_name": ["J"], "artist_name": ["B"]},
    ]
    # Enriched docs carry the aspect words; 'r' doc has a track_id prefix to strip.
    enriched = {"r": "track_id: r, rock", "j": "jazz"}
    return Catalog(rows, enriched_docs=enriched)


def test_from_catalog_builds_aligned_channel_over_enriched_docs():
    cat = _build_catalog()
    ch = ColBERTChannel.from_catalog(
        cat,
        encode_query_fn=_fake_q_encoder(_QTOK),
        encode_docs_fn=lambda texts: [_aspect_doc_encoder(t) for t in texts],
    )
    out = ch.batch_text_to_item_retrieval(["rock", "jazz"], topk=2)
    assert out[0][0] == "r"          # enriched 'rock' aspect matched, prefix stripped
    assert out[1][0] == "j"
    assert ch.index_to_id == ["r", "j"]   # canonical-id-aligned to catalog order


def test_from_catalog_rejects_misaligned_doc_encoder():
    cat = _build_catalog()
    try:
        ColBERTChannel.from_catalog(
            cat, encode_query_fn=_fake_q_encoder(_QTOK),
            encode_docs_fn=lambda texts: [_aspect_doc_encoder(texts[0])],  # too few
        )
        assert False, "expected ValueError on row misalignment"
    except ValueError:
        pass


def test_from_catalog_records_doc_cap_hits():
    cat = _build_catalog()
    ch = ColBERTChannel.from_catalog(
        cat,
        encode_query_fn=_fake_q_encoder(_QTOK),
        encode_docs_fn=lambda texts: [_aspect_doc_encoder(t) for t in texts],
        doc_token_budget=1,          # 'rock' (1 tok) ok; both docs are 1 tok -> 0 over
    )
    assert ch.n_docs_over_budget == 0
