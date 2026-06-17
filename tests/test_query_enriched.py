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
