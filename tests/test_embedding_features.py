"""Session-vector cosine score-fn factory (idea 5: session vibe in audio/CF embedding spaces)."""
from __future__ import annotations

import numpy as np

from mcrs.contracts import TurnContext, UserProfile
from mcrs.rerank.embedding_features import make_session_cos_fn, l2_normalize_rows


def _ctx(history, turn=None):
    turn = turn if turn is not None else max(1, len(history))
    return TurnContext("s", "u", turn, ["q"] * turn, None,
                       UserProfile("u", 1, "f", "US", []), list(history), "warm")


def test_l2_normalize_rows_unit_and_zero_safe():
    mat = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    out = l2_normalize_rows(mat)
    assert np.allclose(out[0], [0.6, 0.8])      # 3-4-5 triangle -> unit
    assert np.allclose(out[1], [0.0, 0.0])      # zero row stays zero (no NaN)
    assert not np.isnan(out).any()
    assert np.allclose(out[2], [1.0, 0.0])


def test_session_cos_aligned_and_orthogonal():
    # rows: A=+x, B=+y (both unit). candidate C=+x, D=-x, E=+y.
    ids = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    mat = l2_normalize_rows(np.array([
        [1, 0], [0, 1], [1, 0], [-1, 0], [0, 1]], dtype=np.float32))
    fn = make_session_cos_fn(ids, mat)
    ctx = _ctx(history=["A", "B"])              # session mean direction = (x+y)/sqrt2
    inv = 1.0 / np.sqrt(2.0)
    assert abs(fn(ctx, "C") - inv) < 1e-6       # +x vs the 45deg mean -> 1/sqrt2
    assert abs(fn(ctx, "E") - inv) < 1e-6       # +y vs mean -> 1/sqrt2
    assert abs(fn(ctx, "D") - (-inv)) < 1e-6    # -x vs mean -> -1/sqrt2


def test_session_cos_zero_when_no_history_or_unknown_candidate():
    ids = {"A": 0, "C": 1}
    mat = l2_normalize_rows(np.array([[1, 0], [1, 0]], dtype=np.float32))
    fn = make_session_cos_fn(ids, mat)
    assert fn(_ctx(history=[]), "C") == 0.0                 # cold: no in-session history
    assert fn(_ctx(history=["A"]), "ZZZ") == 0.0           # candidate not in the embedding index
    assert fn(_ctx(history=["NOPE"]), "C") == 0.0          # all history tracks out of index


def test_session_cos_caches_session_vector_per_turn():
    # different candidates in the same turn must reuse one session vector (and stay correct)
    ids = {"A": 0, "C": 1, "E": 2}
    mat = l2_normalize_rows(np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32))
    fn = make_session_cos_fn(ids, mat)
    ctx = _ctx(history=["A"])
    assert abs(fn(ctx, "C") - 1.0) < 1e-6      # C aligns with the only history track A
    assert abs(fn(ctx, "E") - 0.0) < 1e-6      # E orthogonal -> cache reused, still correct
