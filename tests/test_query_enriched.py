# tests/test_query_enriched.py
from mcrs.contracts import TurnContext, UserProfile
from mcrs.retrieval.query import QueryBuilder

def _ctx(utts, hist):
    return TurnContext(session_id="s", user_id="u", turn_number=len(utts), utterances=utts,
                       goal="discover new music", user_profile=UserProfile("u", None, None, None, []),
                       history_tids=hist, segment="warm" if hist else "cold")

LABELS = {"h1": "Bon Iver – Holocene", "h2": "Tobu – Sunburst"}

def test_warm_query_has_markers_and_taste_newest_first():
    qb = QueryBuilder(markers=True, taste_items=5, track_label_fn=LABELS.get)
    txt = qb.build(_ctx(["hi want chill", "more upbeat"], ["h1", "h2"])).text
    assert txt.startswith("request: more upbeat")
    assert "context: hi want chill" in txt
    assert "goal: discover new music" in txt
    # history_tids is chronological; taste is newest-first
    assert "taste: Tobu – Sunburst; Bon Iver – Holocene" in txt

def test_cold_query_omits_taste_and_empty_lines():
    qb = QueryBuilder(markers=True, taste_items=5, track_label_fn=LABELS.get)
    txt = qb.build(_ctx(["just one turn"], [])).text
    assert "taste:" not in txt
    assert "context:" not in txt          # no older utterances
    assert txt.startswith("request: just one turn")

def test_markers_off_is_legacy_behavior():
    qb = QueryBuilder()  # defaults: markers off
    assert qb.build(_ctx(["a", "b"], [])).text == "a b discover new music"


# ── new edge-case / property / failure-mode tests ──────────────────────────

def _ctx_n(utts, hist, goal="discover new music"):
    """Helper that creates a TurnContext with an explicit goal override."""
    return TurnContext(session_id="s", user_id="u", turn_number=len(utts), utterances=utts,
                      goal=goal,
                      user_profile=UserProfile("u", None, None, None, []),
                      history_tids=hist, segment="warm" if hist else "cold")


def test_empty_utterances_no_crash():
    """Empty utterances list produces an empty Query text without crashing."""
    qb = QueryBuilder(markers=True, taste_items=5, track_label_fn=LABELS.get)
    txt = qb.build(_ctx_n([], [])).text
    # No request:, no context:, no taste: (empty history) — goal line may or may not appear
    assert "request:" not in txt
    assert "context:" not in txt


def test_goal_none_omits_goal_line():
    """When goal is None, no 'goal:' line is produced in markers mode."""
    qb = QueryBuilder(markers=True, taste_items=5, track_label_fn=LABELS.get)
    txt = qb.build(_ctx_n(["i want jazz"], [], goal=None)).text
    assert "goal:" not in txt


def test_goal_empty_string_omits_goal_line():
    """When goal is '', no 'goal:' line is produced in markers mode."""
    qb = QueryBuilder(markers=True, taste_items=5, track_label_fn=LABELS.get)
    txt = qb.build(_ctx_n(["i want jazz"], [], goal="")).text
    assert "goal:" not in txt


def test_track_label_fn_returning_none_skips_those_ids():
    """track_label_fn returning None for some ids skips them; the rest are kept newest-first."""
    # h1 -> "Bon Iver – Holocene", h3 -> None (unknown), h2 -> "Tobu – Sunburst"
    # history is [h1, h3, h2] chronological; newest-first iteration: h2, h3, h1
    # h3 is skipped; result should be [Tobu – Sunburst, Bon Iver – Holocene]
    labels = {"h1": "Bon Iver – Holocene", "h2": "Tobu – Sunburst"}  # h3 missing
    qb = QueryBuilder(markers=True, taste_items=5, track_label_fn=labels.get)
    txt = qb.build(_ctx_n(["something"], ["h1", "h3", "h2"])).text
    assert "taste: Tobu – Sunburst; Bon Iver – Holocene" in txt
    assert "h3" not in txt


def test_history_longer_than_taste_items_only_newest_appear():
    """When history is longer than taste_items, only the newest taste_items entries appear."""
    labels = {f"h{i}": f"Artist{i} – Track{i}" for i in range(10)}
    # history is h0..h9 chronological; taste_items=3 → newest 3 = h9, h8, h7
    qb = QueryBuilder(markers=True, taste_items=3, track_label_fn=labels.get)
    txt = qb.build(_ctx_n(["chill music"], [f"h{i}" for i in range(10)])).text
    assert "Artist9 – Track9" in txt
    assert "Artist8 – Track8" in txt
    assert "Artist7 – Track7" in txt
    # older entries should NOT appear
    assert "Artist6 – Track6" not in txt
    assert "Artist0 – Track0" not in txt


def test_taste_items_zero_produces_no_taste_line():
    """markers=True but taste_items=0 produces no 'taste:' line even with non-empty history."""
    qb = QueryBuilder(markers=True, taste_items=0, track_label_fn=LABELS.get)
    txt = qb.build(_ctx_n(["give me folk", "more please"], ["h1", "h2"])).text
    assert "taste:" not in txt


def test_recency_window_limits_older_utterances_in_markers_mode():
    """recency_window > 0 with markers=True: only the last N older utterances appear in context:."""
    # utterances: [u1, u2, u3, u4]; latest = u4, older = [u1, u2, u3]
    # recency_window=2 → only keep last 2 of older = [u2, u3]
    qb = QueryBuilder(markers=True, recency_window=2, taste_items=0, track_label_fn=None)
    txt = qb.build(_ctx_n(["u1", "u2", "u3", "u4"], [])).text
    assert "request: u4" in txt
    assert "u2" in txt
    assert "u3" in txt
    assert "u1" not in txt   # oldest older utterance is dropped


def test_legacy_plain_exact_output():
    """Byte-equality check for legacy (markers=False) 2-utterance + goal context."""
    qb = QueryBuilder()  # markers=False, all defaults
    txt = qb.build(_ctx_n(["i like jazz", "something mellow"], [])).text
    assert txt == "i like jazz something mellow discover new music"
