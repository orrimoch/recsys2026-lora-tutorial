"""R4 — dense-text channel (brute-force cosine; injectable encoder for hermetic tests)."""
from __future__ import annotations

import numpy as np

from mcrs.retrieval.dense_channel import DenseChannel

_INDEX = ["a", "b", "c"]
_MATRIX = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)


def _fake_encoder(mapping):
    def enc(queries):
        return np.array([mapping[q] for q in queries], dtype=np.float32)
    return enc


def test_dense_returns_nearest_by_cosine():
    ch = DenseChannel(_INDEX, _MATRIX, _fake_encoder({"q": [1.0, 0.0]}))
    out = ch.batch_text_to_item_retrieval(["q"], topk=3)
    assert out[0][0] == "a"            # exact direction match
    assert set(out[0]) == {"a", "b", "c"}


def test_dense_topk_and_batch():
    enc = _fake_encoder({"qa": [1.0, 0.05], "qb": [0.05, 1.0]})
    ch = DenseChannel(_INDEX, _MATRIX, enc)
    out = ch.batch_text_to_item_retrieval(["qa", "qb"], topk=2)
    assert len(out) == 2 and all(len(r) <= 2 for r in out)
    assert out[0][0] == "a" and out[1][0] == "b"


def test_dense_canonicalizes_ids():
    ch = DenseChannel(["track_id: a", "b"], _MATRIX[:2], _fake_encoder({"q": [1.0, 0.0]}))
    out = ch.batch_text_to_item_retrieval(["q"], topk=2)
    assert out[0][0] == "a"  # prefix stripped
