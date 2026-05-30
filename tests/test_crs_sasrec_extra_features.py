"""Unit tests for build_sasrec_extra_features (crs_baseline).

Originally TDD for sasrec_rank; extended (Lever 2) to also always emit
n_channels_hit (cross-channel agreement). Tests assert the sasrec_rank value
as a subset so the additive n_channels_hit key doesn't break them.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "music-crs-baselines"))

from mcrs.crs_baseline import build_sasrec_extra_features


def test_no_sasrec_seq_still_emits_n_channels_hit():
    # No sasrec_seq label: sasrec_rank is omitted, but n_channels_hit is still
    # emitted (Lever 2). t_a and t_b each appear in the single sub -> hit=1.
    per_sub = [[["t_a", "t_b"]]]  # one sub (bm25), one query
    labels = ["bm25"]
    batch = [["t_a", "t_b"]]
    result = build_sasrec_extra_features(per_sub, labels, batch)
    assert result is not None
    q0 = result[0]
    assert "sasrec_rank" not in q0[0], "no sasrec sub -> no sasrec_rank"
    assert q0[0]["n_channels_hit"] == 1
    assert q0[1]["n_channels_hit"] == 1


def test_returns_none_when_no_per_sub():
    assert build_sasrec_extra_features([], [], [["t_a"]]) is None


def test_n_channels_hit_counts_channels():
    # t_a in both subs -> hit=2; t_b only in sasrec -> hit=1.
    per_sub = [
        [["t_a"]],                 # bm25
        [["t_a", "t_b"]],          # sasrec_seq
    ]
    labels = ["bm25", "sasrec_seq"]
    batch = [["t_a", "t_b"]]
    result = build_sasrec_extra_features(per_sub, labels, batch)
    q0 = result[0]
    assert q0[0]["n_channels_hit"] == 2, f"t_a hit by 2 channels, got {q0[0]}"
    assert q0[1]["n_channels_hit"] == 1, f"t_b hit by 1 channel, got {q0[1]}"


def test_correct_1indexed_ranks_for_ranked_candidates():
    # sasrec sub is index 1; ranks: t_a->1, t_b->2, t_c->3
    per_sub = [
        [["x", "y"]],                  # bm25 (ignored for rank)
        [["t_a", "t_b", "t_c"]],       # sasrec_seq
    ]
    labels = ["bm25", "sasrec_seq"]
    batch = [["t_a", "t_b", "t_c"]]
    result = build_sasrec_extra_features(per_sub, labels, batch)
    q0 = result[0]
    assert q0[0]["sasrec_rank"] == 1, f"t_a is rank 1, got {q0[0]}"
    assert q0[1]["sasrec_rank"] == 2, f"t_b is rank 2, got {q0[1]}"
    assert q0[2]["sasrec_rank"] == 3, f"t_c is rank 3, got {q0[2]}"


def test_sentinel_for_candidate_absent_from_sasrec_ranking():
    per_sub = [
        [["t_a"]],            # sasrec_seq at index 0
    ]
    labels = ["sasrec_seq"]
    batch = [["t_a", "t_missing"]]
    result = build_sasrec_extra_features(per_sub, labels, batch)
    q0 = result[0]
    assert q0[0]["sasrec_rank"] == 1, f"t_a should be rank 1, got {q0[0]}"
    assert q0[1]["sasrec_rank"] == 10000, f"t_missing should be sentinel, got {q0[1]}"


def test_alignment_output_shape_matches_batch_retrieval_items():
    per_sub = [
        [["t1", "t2"], ["t3"]],        # sasrec_seq, 2 queries
    ]
    labels = ["sasrec_seq"]
    batch = [["t1", "t2"], ["t3"]]
    result = build_sasrec_extra_features(per_sub, labels, batch)
    assert len(result) == 2
    assert result[0][0]["sasrec_rank"] == 1, f"q0 t1 rank should be 1, got {result[0][0]}"
    assert result[1][0]["sasrec_rank"] == 1, f"q1 t3 rank should be 1, got {result[1][0]}"


def test_custom_sentinel_forwarded():
    per_sub = [
        [["t_a"]],
    ]
    labels = ["sasrec_seq"]
    batch = [["t_a", "t_missing"]]
    result = build_sasrec_extra_features(per_sub, labels, batch, sentinel=99999)
    q0 = result[0]
    assert q0[0]["sasrec_rank"] == 1, f"t_a rank 1, got {q0[0]}"
    assert q0[1]["sasrec_rank"] == 99999, (
        f"Expected custom sentinel 99999 for t_missing, got {q0[1]}")


def test_correct_sub_selected_when_multiple_subs():
    per_sub = [
        [["a", "b"]],                  # bm25
        [["t_sasrec_top", "t2"]],      # sasrec_seq at index 1
    ]
    labels = ["bm25", "sasrec_seq"]
    batch = [["t_sasrec_top", "t2"]]
    result = build_sasrec_extra_features(per_sub, labels, batch)
    q0 = result[0]
    assert q0[0]["sasrec_rank"] == 1, f"t_sasrec_top should be rank 1, got {q0[0]}"
