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
