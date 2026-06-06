"""TDD for the CLAP audio-similarity reranker feature (shared train/serve fn).

clap_session_similarity(cand_tid, played_tids, clap) = mean cosine similarity of
the candidate's (L2-normalized) CLAP vector to the session's played tracks' CLAP
vectors. "Does this candidate sound like what was already played." 0.0 when the
candidate or all played tracks lack a CLAP vector / no history.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "music-crs-baselines"))

from mcrs.retrieval_modules.clap_similarity import clap_session_similarity


def _n(v):
    v = np.array(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _lookup():
    # Toy CLAP vectors, PRE-NORMALIZED to match the contract (load_clap_lookup
    # L2-normalizes; the fn assumes normalized inputs so cosine == dot).
    return {
        "p1": _n([1.0, 0.0, 0.0]),       # played
        "p2": _n([0.0, 1.0, 0.0]),       # played
        "c_same": _n([1.0, 0.0, 0.0]),   # == p1
        "c_mid":  _n([1.0, 1.0, 0.0]),   # 45deg to both
        "c_far":  _n([0.0, 0.0, 1.0]),   # orthogonal
    }


def test_identical_to_a_played_track_high_sim():
    lk = _lookup()
    # c_same == p1; mean cos to {p1,p2} = (1 + 0)/2 = 0.5
    s = clap_session_similarity("c_same", ["p1", "p2"], lk)
    assert abs(s - 0.5) < 1e-6, s


def test_orthogonal_candidate_zero_sim():
    lk = _lookup()
    # c_far orthogonal to both played -> mean cos 0
    s = clap_session_similarity("c_far", ["p1", "p2"], lk)
    assert abs(s - 0.0) < 1e-6, s


def test_midpoint_candidate():
    lk = _lookup()
    # c_mid = (1,1,0)/sqrt2; cos to p1 = 1/sqrt2, to p2 = 1/sqrt2; mean = 1/sqrt2
    s = clap_session_similarity("c_mid", ["p1", "p2"], lk)
    assert abs(s - (1.0 / np.sqrt(2))) < 1e-6, s


def test_no_history_returns_zero():
    lk = _lookup()
    assert clap_session_similarity("c_same", [], lk) == 0.0


def test_candidate_missing_clap_returns_zero():
    lk = _lookup()
    assert clap_session_similarity("unknown_tid", ["p1"], lk) == 0.0


def test_played_missing_clap_skipped():
    lk = _lookup()
    # only p1 has a vector among the played; p_missing is ignored
    s = clap_session_similarity("c_same", ["p1", "p_missing"], lk)
    assert abs(s - 1.0) < 1e-6, s  # mean over the 1 valid played = cos(c_same,p1)=1


# --- mean-imputation + has-vector indicator (step-3 prep; fixes the 0.0 sentinel) ---

def test_mean_impute_none_preserves_zero():
    """Backward compat: with no mean_vec, a missing candidate still returns 0.0
    (identical to the original 0.0-sentinel behaviour)."""
    lk = _lookup()
    assert clap_session_similarity("unknown_tid", ["p1"], lk) == 0.0


def test_mean_impute_candidate_missing_uses_mean_vec():
    """With mean_vec supplied, a candidate that has no CLAP vector is imputed to
    the catalog mean instead of returning a hard 0.0."""
    lk = _lookup()
    mean = _n([1.0, 1.0, 1.0])  # imputed candidate
    # cos(mean, p1=(1,0,0)) = 1/sqrt(3); only p1 played
    s = clap_session_similarity("unknown_tid", ["p1"], lk, mean_vec=mean)
    assert abs(s - (1.0 / np.sqrt(3))) < 1e-6, s


def test_mean_impute_played_missing_still_skipped():
    """Missing PLAYED tracks are never imputed (even with mean_vec) — avoids the
    degenerate mean-vs-mean == 1.0 artifact."""
    lk = _lookup()
    mean = _n([1.0, 1.0, 1.0])
    s = clap_session_similarity("c_same", ["p1", "p_missing"], lk, mean_vec=mean)
    assert abs(s - 1.0) < 1e-6, s  # only p1 counted


def test_clap_has_vector_indicator():
    from mcrs.retrieval_modules.clap_similarity import clap_has_vector
    lk = _lookup()
    assert clap_has_vector("p1", lk) == 1
    assert clap_has_vector("unknown_tid", lk) == 0


def test_clap_mean_vector_normalized_and_empty():
    from mcrs.retrieval_modules.clap_similarity import clap_mean_vector
    lk = _lookup()
    m = clap_mean_vector(lk)
    assert abs(float(np.linalg.norm(m)) - 1.0) < 1e-6
    assert clap_mean_vector({}) is None


# --- clap_session_query: audio->session query for the RECALL-channel probe -----

def test_clap_session_query_mean_pools_and_normalizes():
    from mcrs.retrieval_modules.clap_similarity import clap_session_query
    lk = _lookup()
    q = clap_session_query(["p1", "p2"], lk)   # p1=[1,0,0], p2=[0,1,0]
    assert q is not None
    assert abs(float(np.linalg.norm(q)) - 1.0) < 1e-5   # L2-normalized
    assert abs(float(q[0]) - float(q[1])) < 1e-5 and abs(float(q[2])) < 1e-6


def test_clap_session_query_none_when_no_vectors():
    from mcrs.retrieval_modules.clap_similarity import clap_session_query
    lk = _lookup()
    assert clap_session_query([], lk) is None               # cold / no history
    assert clap_session_query(["unknown_tid"], lk) is None   # none have a vector


def test_clap_session_query_skips_missing_played():
    from mcrs.retrieval_modules.clap_similarity import clap_session_query
    lk = _lookup()
    q = clap_session_query(["p1", "missing"], lk)  # only p1 has a vector
    assert q is not None
    np.testing.assert_allclose(q, lk["p1"], atol=1e-5)  # == p1 (the only vector)
