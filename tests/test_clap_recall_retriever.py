"""TDD for the CLAP audio->session RECALL channel (wRRF sub-retriever).

For each turn: mean-pool the session's PLAYED tracks' CLAP vectors -> return the
catalog tracks whose CLAP audio is nearest ("sounds like what they've played"),
minus already-played. Cold turns (no played track with a CLAP vector) -> [].
Distinct from the rejected clap_session_sim reranker FEATURE: this ADDS candidates.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "music-crs-baselines"))

from mcrs.retrieval_modules.clap_recall import ClapRecallRetriever


def _n(v):
    v = np.array(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _build(lookup):
    r = ClapRecallRetriever.__new__(ClapRecallRetriever)
    r._build(lookup)
    return r


def _lk():
    return {
        "p1":   _n([1.0, 0.0, 0.0]),   # played
        "near": _n([0.9, 0.1, 0.0]),   # closest to p1
        "mid":  _n([0.5, 0.5, 0.0]),
        "far":  _n([0.0, 0.0, 1.0]),   # orthogonal to p1
    }


def test_retrieves_nearest_unplayed_to_session_audio():
    r = _build(_lk())
    out = r.batch_text_to_item_retrieval(["q"], topk=10,
                                         batch_context=[{"history_tids": ["p1"]}])[0]
    assert "p1" not in out                 # played excluded
    assert out == ["near", "mid", "far"]   # ranked by cosine to the played audio


def test_no_history_returns_empty():
    r = _build(_lk())
    assert r.batch_text_to_item_retrieval(["q"], topk=10, batch_context=[{}])[0] == []


def test_played_without_clap_vector_returns_empty():
    r = _build(_lk())
    out = r.batch_text_to_item_retrieval(["q"], topk=10,
                                         batch_context=[{"history_tids": ["unknown"]}])[0]
    assert out == []


def test_mean_pools_multiple_played_and_excludes_them():
    r = _build(_lk())
    # query = mean(p1=[1,0,0], far=[0,0,1]); nearest unplayed is 'near'
    out = r.batch_text_to_item_retrieval(["q"], topk=10,
                                         batch_context=[{"history_tids": ["p1", "far"]}])[0]
    assert out[0] == "near"
    assert "p1" not in out and "far" not in out


def test_wrrf_spec_adds_clap_recall_when_opted_in():
    from mcrs.retrieval_modules import _wrrf_union_v1_specs
    assert not any(s["type"] == "clap_recall" for s in _wrrf_union_v1_specs({}))
    on = _wrrf_union_v1_specs({"use_clap_recall": True, "w_clap_recall": 0.5})
    clap = [s for s in on if s["type"] == "clap_recall"]
    assert len(clap) == 1 and clap[0]["weight"] == 0.5
