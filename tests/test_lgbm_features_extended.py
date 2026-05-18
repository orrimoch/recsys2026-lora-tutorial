"""Tests for new feature helpers added to scripts/build_lgbm_features.py."""
import pytest


def test_compute_release_year_cyclical_for_1969_returns_unit_circle():
    from scripts.build_lgbm_features import compute_release_year_cyclical
    sin, cos = compute_release_year_cyclical("1969-05-29")
    assert -1.0 <= sin <= 1.0
    assert -1.0 <= cos <= 1.0
    # The pair must be on the unit circle.
    assert abs((sin * sin + cos * cos) - 1.0) < 1e-9


def test_compute_release_year_cyclical_handles_missing():
    from scripts.build_lgbm_features import compute_release_year_cyclical
    assert compute_release_year_cyclical(None) == (0.0, 0.0)
    assert compute_release_year_cyclical("") == (0.0, 0.0)
    assert compute_release_year_cyclical("not-a-date") == (0.0, 0.0)


def test_compute_tag_overlap_counts_substring_matches():
    from scripts.build_lgbm_features import compute_tag_overlap
    n = compute_tag_overlap("I love folk rock from the 70s", ["folk rock", "70s", "metal"])
    assert n == 2


def test_compute_tag_overlap_empty_list_returns_zero():
    from scripts.build_lgbm_features import compute_tag_overlap
    assert compute_tag_overlap("anything", None) == 0
    assert compute_tag_overlap("anything", []) == 0


def test_last_turn_moved_toward_goal_codes():
    from scripts.build_lgbm_features import last_turn_moved_toward_goal
    assert last_turn_moved_toward_goal(["MOVES_TOWARD_GOAL"]) == 1
    assert last_turn_moved_toward_goal(["DOES_NOT_MOVE_TOWARD_GOAL"]) == 0
    assert last_turn_moved_toward_goal([]) == -1
    assert last_turn_moved_toward_goal(None) == -1
    # Uses LAST assessment (most-recent state).
    assert last_turn_moved_toward_goal(["DOES_NOT_MOVE_TOWARD_GOAL", "MOVES_TOWARD_GOAL"]) == 1


def test_query_drift_score_first_turn_returns_one():
    from scripts.build_lgbm_features import query_drift_score
    assert query_drift_score("hello", prior_queries=[], embedder=None) == 1.0


def test_pop_rank_pct_known_track_returns_in_range():
    from scripts.build_lgbm_features import build_pop_rank_pct_map
    track_meta = {"t1": {"popularity": 100.0}, "t2": {"popularity": 50.0}, "t3": {"popularity": 200.0}}
    pct = build_pop_rank_pct_map(track_meta)
    # t3 has highest popularity → smallest rank → smallest pct.
    assert pct["t3"] < pct["t1"] < pct["t2"]
    for v in pct.values():
        assert 0.0 <= v <= 1.0
