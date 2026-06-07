"""Tier-1 #3.2: raw_enriched retrieval-query mode appends culture + profile.

These taste signals (preferred_musical_culture + age/country/gender) are legal at
inference and help cold single-turn Blind queries surface new-artist golds. The
mode is opt-in so the shipped raw_with_goal path is unchanged.
"""
from mcrs.crs_baseline import build_retrieval_query


SM = [
    {"role": "user", "content": "play something upbeat"},
    {"role": "assistant", "content": "Sure, here's a track"},
]
PROFILE = {"preferred_musical_culture": "Western Alternative Rock",
           "age_group": "20s", "country_name": "Brazil", "gender": "female"}


def test_raw_enriched_appends_goal_culture_and_profile():
    q = build_retrieval_query(SM, mode="raw_enriched",
                              goal_text="discover new punk", user_profile=PROFILE)
    lines = q.splitlines()
    assert "goal: discover new punk" in lines
    assert "culture: Western Alternative Rock" in lines
    assert "user: age=20s country=Brazil gender=female" in lines
    # base (raw history) preserved as the leading lines
    assert lines[0] == "user: play something upbeat"


def test_raw_enriched_omits_missing_lines():
    # no goal, no profile -> identical to raw_with_goal (which == raw when no goal)
    q = build_retrieval_query(SM, mode="raw_enriched")
    base = build_retrieval_query(SM, mode="raw_with_goal")
    assert q == base
    # no appended taste lines (the history's own "user:" role prefix doesn't count)
    assert "culture:" not in q and "user: age=" not in q


def test_raw_enriched_culture_only_when_profile_absent():
    q = build_retrieval_query(SM, mode="raw_enriched", goal_text="x",
                              user_profile={"preferred_musical_culture": "K-Pop"})
    assert "culture: K-Pop" in q
    assert "user: age=" not in q  # no age/country/gender -> no profile line


def test_raw_enriched_is_an_accepted_mode():
    # CRS_BASELINE.__init__ must accept raw_enriched as a query_preprocessing_mode.
    import inspect
    from mcrs.crs_baseline import CRS_BASELINE
    src = inspect.getsource(CRS_BASELINE.__init__)
    assert "raw_enriched" in src, "raw_enriched must be in the allowed mode set"
