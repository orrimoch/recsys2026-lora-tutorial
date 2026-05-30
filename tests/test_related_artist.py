"""TDD for RelatedArtistRetriever (Lever 3): reaches NEW artists via cross-session
artist co-occurrence (the 95.7%-new-artist recall wall). Stage 16 probe showed
~29% of union-missed golds are reachable at top-100 co-occurring artists.

We build the retriever via __new__ + inject a tiny in-memory index (skips the
heavy catalog + train-cooc build), then assert serve-time behavior.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "music-crs-baselines"))

from collections import Counter
from mcrs.retrieval_modules.related_artist import RelatedArtistRetriever


def _stub():
    r = RelatedArtistRetriever.__new__(RelatedArtistRetriever)
    # tid -> artist
    r.tid_to_artist = {
        "p1": "alpha",         # played track, by alpha
        "a1": "alpha", "a2": "alpha",
        "b1": "beta", "b2": "beta",
        "g1": "gamma",
    }
    # artist -> popularity-sorted track ids
    r.artist_to_tids = {
        "alpha": ["a1", "a2"],
        "beta": ["b1", "b2"],
        "gamma": ["g1"],
    }
    # co-occurrence: alpha most often appears with beta, then gamma
    r.cooc = {"alpha": Counter({"beta": 10, "gamma": 3})}
    r.catalog_tids = set(r.tid_to_artist.keys())
    return r


def test_reaches_new_artists_ranked_by_cooccurrence():
    """Session played alpha (p1). Channel should surface beta's tracks (top
    co-occurring NEW artist) before gamma's, and NOT alpha's (already in session)."""
    r = _stub()
    ctx = [{"history_tids": ["p1"]}]
    out = r.batch_text_to_item_retrieval(["q"], topk=10, batch_context=ctx)[0]
    # beta tracks (cooc=10) come before gamma (cooc=3)
    assert out == ["b1", "b2", "g1"], out
    # never emits the session's own artist (alpha)
    assert "a1" not in out and "a2" not in out


def test_empty_when_no_history():
    r = _stub()
    out = r.batch_text_to_item_retrieval(["q"], topk=10, batch_context=[{}])[0]
    assert out == []


def test_respects_topk():
    r = _stub()
    ctx = [{"history_tids": ["p1"]}]
    out = r.batch_text_to_item_retrieval(["q"], topk=1, batch_context=ctx)[0]
    assert out == ["b1"]


def test_excludes_already_played_tracks():
    """A co-occurring artist's track that was already played is skipped."""
    r = _stub()
    # pretend b1 was already played; only b2 then g1 should surface
    ctx = [{"history_tids": ["p1", "b1"]}]
    out = r.batch_text_to_item_retrieval(["q"], topk=10, batch_context=ctx)[0]
    assert "b1" not in out
    assert out[0] == "b2"
