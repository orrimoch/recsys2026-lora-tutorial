"""Tests for build_retrieval_query(mode="compact_colbert").

The compact ColBERT query is the "be humble about the 512 ceiling" construction:
instead of the full multi-turn dialog (which truncates to noise — 75-88% of real
queries exceed any feasible q_len), feed ColBERT only the high-signal slice:

    goal: <listener_goal>
    culture: <preferred_musical_culture>
    <most recent user turn>

goal/culture are labelled and placed FIRST so right-truncation on a rare long user
turn drops only the query tail, never the durable intent/taste. The user turn carries
NO role prefix (same rationale as mode="last_user": role labels are retrieval noise).
goal/culture lines are omitted when absent, so the format degrades to bare last_user.
No demographics (age/country/gender) — intentionally minimal.

This one function is the SHARED builder for serve (crs_baseline) and the ColBERT
train-data builder (build_colbert_train_data), so the query is byte-identical between
training and inference. (The nb74 DEV harness is not yet wired to it.)
"""
from mcrs.crs_baseline import build_retrieval_query

_SM = [
    {"role": "user", "content": "Play upbeat pop"},
    {"role": "music", "content": "7ce1ed9d-5ec8-4584-aa71-5a67db2781ea"},
    {"role": "assistant", "content": "Here's a track you might like"},
    {"role": "user", "content": "Something more chill"},
]
_GOAL = "find relaxing evening music"
_PROFILE = {"preferred_musical_culture": "Western Pop",
            "age": 25, "country_name": "US", "gender": "f"}


def test_compact_colbert_builds_goal_culture_and_last_user_turn():
    out = build_retrieval_query(_SM, mode="compact_colbert",
                                goal_text=_GOAL, user_profile=_PROFILE)
    assert out == ("goal: find relaxing evening music\n"
                   "culture: Western Pop\n"
                   "Something more chill")


def test_compact_colbert_uses_only_the_last_user_turn():
    out = build_retrieval_query(_SM, mode="compact_colbert",
                                goal_text=_GOAL, user_profile=_PROFILE)
    assert "Something more chill" in out          # the last user turn
    assert "Play upbeat pop" not in out           # earlier user turn dropped
    assert "Here's a track" not in out            # assistant turn dropped
    assert "7ce1ed9d" not in out                  # music-turn UUID dropped


def test_compact_colbert_omits_goal_line_when_no_goal():
    out = build_retrieval_query(_SM, mode="compact_colbert",
                                goal_text=None, user_profile=_PROFILE)
    assert "goal:" not in out
    assert out == "culture: Western Pop\nSomething more chill"


def test_compact_colbert_omits_culture_line_when_no_profile():
    out = build_retrieval_query(_SM, mode="compact_colbert",
                                goal_text=_GOAL, user_profile=None)
    assert "culture:" not in out
    assert out == "goal: find relaxing evening music\nSomething more chill"


def test_compact_colbert_goal_and_culture_precede_the_user_turn():
    # Truncation-safety: durable signals first, query content last.
    out = build_retrieval_query(_SM, mode="compact_colbert",
                                goal_text=_GOAL, user_profile=_PROFILE)
    assert out.index("goal:") < out.index("culture:") < out.index("Something more chill")


def test_compact_colbert_excludes_demographics():
    # "recent query + goal + cultural" — NOT age/country/gender.
    out = build_retrieval_query(_SM, mode="compact_colbert",
                                goal_text=_GOAL, user_profile=_PROFILE)
    assert "age=" not in out and "country" not in out and "gender" not in out


def test_compact_colbert_degrades_to_bare_last_user_with_no_goal_or_culture():
    out = build_retrieval_query(_SM, mode="compact_colbert",
                                goal_text=None, user_profile=None)
    assert out == "Something more chill"


def test_compact_colbert_parses_json_string_profile():
    # I1 robustness: serve may hand a JSON-string profile; culture must still survive
    # (else serve drops 'culture:' while train keeps it -> skew on ~75% warm sessions).
    out = build_retrieval_query(_SM, mode="compact_colbert", goal_text=_GOAL,
                                user_profile='{"preferred_musical_culture": "Western Pop"}')
    assert "culture: Western Pop" in out


def test_compact_colbert_parses_python_repr_string_profile():
    out = build_retrieval_query(_SM, mode="compact_colbert", goal_text=_GOAL,
                                user_profile="{'preferred_musical_culture': 'Western Pop'}")
    assert "culture: Western Pop" in out


def test_compact_colbert_unparseable_string_profile_omits_culture_gracefully():
    out = build_retrieval_query(_SM, mode="compact_colbert", goal_text=_GOAL,
                                user_profile="not a profile")
    assert "culture:" not in out
    assert out == "goal: find relaxing evening music\nSomething more chill"


def test_compact_colbert_blank_culture_string_is_omitted():
    out = build_retrieval_query(_SM, mode="compact_colbert", goal_text=_GOAL,
                                user_profile={"preferred_musical_culture": "   "})
    assert "culture:" not in out
    assert out == "goal: find relaxing evening music\nSomething more chill"
