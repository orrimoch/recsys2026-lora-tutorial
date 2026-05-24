"""Regression test for DENSE_MULTIMODAL_LOCAL query encoding.

The multi-modal query encoder must encode queries in mini-batches rather than
one giant forward pass. Encoding all queries at once allocates a single FFN
activation of ``n_queries x query_max_len x intermediate_size`` floats — for an
8000-query dev/Blind eval at bge-base (384 x 3072) that is ~35 GiB, which OOMs
a 95 GB GPU. See cell 7 of colab/70_train_bi_encoder.ipynb.
"""
import numpy as np
import torch

from mcrs.retrieval_modules.dense_multimodal_local import DENSE_MULTIMODAL_LOCAL


class _FakeBatchEncoding(dict):
    def to(self, _device):
        return self


class _FakeTokenizer:
    """Returns fixed-shape token tensors; records nothing."""

    def __call__(self, queries, **kwargs):
        b = len(queries)
        return _FakeBatchEncoding(
            input_ids=torch.ones(b, 4, dtype=torch.long),
            attention_mask=torch.ones(b, 4, dtype=torch.long),
        )


class _FakeModel:
    """Echoes ``user_cf`` back as the query embedding so each row's identity is
    verifiable after chunking + concatenation. Records per-call batch sizes."""

    def __init__(self):
        self.batch_sizes = []

    def parameters(self):
        yield torch.zeros(1)  # cpu param → next(model.parameters()).device == cpu

    def forward_query(self, input_ids, attention_mask, user_cf):
        self.batch_sizes.append(int(input_ids.shape[0]))
        return user_cf


def _make_retriever(model, tokenizer, batch_size):
    r = DENSE_MULTIMODAL_LOCAL.__new__(DENSE_MULTIMODAL_LOCAL)  # bypass heavy __init__
    r.query_max_len = 8
    r.query_encode_batch_size = batch_size
    r._get_model_and_tokenizer = lambda: (model, tokenizer)
    # user_cf for uid i is a length-3 vector full of float(i)
    r._get_user_cf = lambda uid: np.full(3, float(uid), dtype=np.float32)
    return r


def test_encode_queries_chunks_and_preserves_order():
    model = _FakeModel()
    r = _make_retriever(model, _FakeTokenizer(), batch_size=4)
    queries = [f"q{i}" for i in range(10)]
    user_ids = list(range(10))

    out = r._encode_queries(queries, user_ids)

    # Encoded in mini-batches of <=4, never a single 10-wide forward.
    assert model.batch_sizes == [4, 4, 2]
    # Row order survives chunking + concatenation.
    assert out.shape == (10, 3)
    for i in range(10):
        assert np.allclose(out[i], i), f"row {i} misordered after chunking"


def test_encode_queries_single_chunk_when_input_small():
    model = _FakeModel()
    r = _make_retriever(model, _FakeTokenizer(), batch_size=64)
    queries = [f"q{i}" for i in range(3)]
    user_ids = list(range(3))

    out = r._encode_queries(queries, user_ids)

    assert model.batch_sizes == [3]
    assert out.shape == (3, 3)
