"""K1 — FeatureBuilder: behavioral/session + chat-derived rerank features."""
from __future__ import annotations

from mcrs.contracts import Candidate, TurnContext, UserProfile
from mcrs.rerank.features import FeatureBuilder
from mcrs.data.catalog import Catalog


def _ctx(history, utterances=None, goal=None, turn=None):
    turn = turn if turn is not None else (len(utterances) if utterances else 1)
    utterances = utterances if utterances is not None else ["q"] * turn
    return TurnContext("s", "u", turn, utterances, goal,
                       UserProfile("u", 1, "f", "US", []), list(history), "warm")


def _cat(rows):
    return Catalog(rows)


# ---- Task 1: behavioral history features (replay + artist affinity) ----
def test_behavioral_history_features():
    cat = _cat([
        {"track_id": "A1", "artist_name": ["A"]},
        {"track_id": "B1", "artist_name": ["B"]},
        {"track_id": "A2", "artist_name": ["A"]},
        {"track_id": "A3", "artist_name": ["A"]},   # candidate by artist A, never played
        {"track_id": "C9", "artist_name": ["C"]},   # candidate, unrelated artist
    ])
    fb = FeatureBuilder(cat, ["bm25"])
    ctx = _ctx(history=["A1", "B1", "A2"])
    cands = [Candidate("A2"), Candidate("A3"), Candidate("B1"), Candidate("C9")]
    fb.build(ctx, cands)
    f = {c.track_id: c.features for c in cands}
    # is_replay: only A2 and B1 are in history
    assert f["A2"]["is_replay"] == 1.0 and f["B1"]["is_replay"] == 1.0
    assert f["A3"]["is_replay"] == 0.0 and f["C9"]["is_replay"] == 0.0
    # artist_play_count: A-tracks share artist with 2 history plays (A1,A2); B with 1; C with 0
    assert f["A3"]["artist_play_count"] == 2.0
    assert f["B1"]["artist_play_count"] == 1.0
    assert f["C9"]["artist_play_count"] == 0.0
    # last_artist_match: last play is A2 (artist A) -> A-tracks match, others don't
    assert f["A3"]["last_artist_match"] == 1.0 and f["C9"]["last_artist_match"] == 0.0
    assert f["B1"]["last_artist_match"] == 0.0
    # artist_recency: A last at distance 0 (A2 is last) -> 1.0; B at distance 1 -> 0.5; C -> 0
    assert f["A3"]["artist_recency"] == 1.0
    assert f["B1"]["artist_recency"] == 0.5
    assert f["C9"]["artist_recency"] == 0.0


def test_behavioral_features_in_feature_names_and_empty_history_safe():
    cat = _cat([{"track_id": "x", "artist_name": ["Z"]}])
    fb = FeatureBuilder(cat, ["bm25"])
    for name in ("is_replay", "artist_play_count", "last_artist_match", "artist_recency"):
        assert name in fb.feature_names
    ctx = _ctx(history=[])                      # cold turn, no history
    cands = [Candidate("x")]
    fb.build(ctx, cands)
    f = cands[0].features
    assert f["is_replay"] == 0.0 and f["artist_play_count"] == 0.0
    assert f["last_artist_match"] == 0.0 and f["artist_recency"] == 0.0


# ---- Task 2: chat-mention features (assistant<->user dialogue) ----
def test_chat_mention_features():
    cat = _cat([
        {"track_id": "rh", "artist_name": ["Radiohead"], "track_name": ["Creep"]},
        {"track_id": "ot", "artist_name": ["Other Band"], "track_name": ["Unrelated"]},
    ])
    fb = FeatureBuilder(cat, ["bm25"])
    ctx = _ctx(history=[], utterances=["play something like radiohead please"], goal="upbeat creep vibes")
    cands = [Candidate("rh"), Candidate("ot")]
    fb.build(ctx, cands)
    f = {c.track_id: c.features for c in cands}
    assert f["rh"]["artist_mentioned_in_chat"] == 1.0     # "radiohead" appears in utterance
    assert f["ot"]["artist_mentioned_in_chat"] == 0.0
    assert f["rh"]["track_mentioned_in_chat"] == 1.0      # "creep" appears in goal text
    assert f["ot"]["track_mentioned_in_chat"] == 0.0


def test_chat_mention_short_name_not_false_matched_and_no_goal_safe():
    # short artist "U2" (len<3) must NOT substring-match random text; missing goal must not crash
    cat = _cat([{"track_id": "u2", "artist_name": ["U2"], "track_name": ["One"]}])
    fb = FeatureBuilder(cat, ["bm25"])
    ctx = _ctx(history=[], utterances=["i want a quiet untune evening"], goal=None)
    cands = [Candidate("u2")]
    fb.build(ctx, cands)
    f = cands[0].features
    assert f["artist_mentioned_in_chat"] == 0.0   # "u2" guarded by min-length, no false hit on "untune"
    assert "track_mentioned_in_chat" in f          # no crash with goal=None


# ---- Task 3: injected chat/session embedding score_fns plumb through ----
def test_injected_chat_and_session_score_fns_get_columns_and_norm():
    cat = _cat([{"track_id": "x", "artist_name": ["Z"]}])
    fb = FeatureBuilder(cat, ["bm25"], score_fns={
        "chat_doc_cos": lambda ctx, tid: 0.7,
        "session_emb_cos": lambda ctx, tid: 0.2,
    })
    for name in ("chat_doc_cos", "session_emb_cos", "chat_doc_cos_norm", "session_emb_cos_norm"):
        assert name in fb.feature_names
    ctx = _ctx(history=[])
    cands = [Candidate("x")]
    fb.build(ctx, cands)
    assert cands[0].features["chat_doc_cos"] == 0.7
    assert cands[0].features["session_emb_cos"] == 0.2
