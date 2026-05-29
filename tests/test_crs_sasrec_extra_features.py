"""TDD tests for build_sasrec_extra_features in mcrs.crs_baseline.

Written BEFORE the implementation — all tests must fail on un-patched code
and pass once build_sasrec_extra_features is added.
"""
import pytest


# ---------------------------------------------------------------------------
# Test 1: returns None when no "sasrec_seq" sub is present
# ---------------------------------------------------------------------------

def test_returns_none_when_no_sasrec_seq_label():
    """If labels list doesn't contain 'sasrec_seq', return None."""
    from mcrs.crs_baseline import build_sasrec_extra_features

    per_sub = [[["t1", "t2"]], [["t2", "t1"]]]
    labels = ["bm25", "dense"]
    batch_retrieval_items = [["t1", "t2"]]

    result = build_sasrec_extra_features(per_sub, labels, batch_retrieval_items)
    assert result is None, f"Expected None when no sasrec_seq label, got {result!r}"


# ---------------------------------------------------------------------------
# Test 2: correct 1-indexed ranks for candidates in the SASRec ranking
# ---------------------------------------------------------------------------

def test_correct_1indexed_ranks_for_ranked_candidates():
    """Candidates present in the SASRec sub's ranking get correct 1-indexed sasrec_rank."""
    from mcrs.crs_baseline import build_sasrec_extra_features

    # sasrec_seq sub ranked: ["t_a", "t_b", "t_c"] for the one query
    per_sub = [[["t_a", "t_b", "t_c"]]]
    labels = ["sasrec_seq"]
    batch_retrieval_items = [["t_a", "t_b", "t_c"]]

    result = build_sasrec_extra_features(per_sub, labels, batch_retrieval_items)
    assert result is not None
    assert len(result) == 1, "One list per query"
    q0 = result[0]
    assert len(q0) == 3, "One dict per candidate"
    assert q0[0] == {"sasrec_rank": 1}, f"t_a is rank 1, got {q0[0]}"
    assert q0[1] == {"sasrec_rank": 2}, f"t_b is rank 2, got {q0[1]}"
    assert q0[2] == {"sasrec_rank": 3}, f"t_c is rank 3, got {q0[2]}"


# ---------------------------------------------------------------------------
# Test 3: sentinel 10000 for a candidate absent from the SASRec sub's ranking
# ---------------------------------------------------------------------------

def test_sentinel_for_candidate_absent_from_sasrec_ranking():
    """A candidate in batch_retrieval_items but absent from the SASRec ranking
    gets sasrec_rank == 10000 (the default sentinel)."""
    from mcrs.crs_baseline import build_sasrec_extra_features

    # sasrec_seq ranked only t_a; t_bm25_only is NOT in its list
    per_sub = [[["t_a"]]]
    labels = ["sasrec_seq"]
    batch_retrieval_items = [["t_a", "t_bm25_only"]]

    result = build_sasrec_extra_features(per_sub, labels, batch_retrieval_items)
    assert result is not None
    q0 = result[0]
    assert q0[0] == {"sasrec_rank": 1}, f"t_a should be rank 1, got {q0[0]}"
    assert q0[1] == {"sasrec_rank": 10000}, (
        f"t_bm25_only is absent from sasrec — expected sentinel 10000, got {q0[1]}"
    )


# ---------------------------------------------------------------------------
# Test 4: alignment — output shape matches batch_retrieval_items exactly
# ---------------------------------------------------------------------------

def test_alignment_output_shape_matches_batch_retrieval_items():
    """Output has one list per query, each with one dict per candidate,
    in the same order as batch_retrieval_items."""
    from mcrs.crs_baseline import build_sasrec_extra_features

    # Two queries; sasrec_seq sub may rank different items for each
    per_sub = [
        # sasrec_seq per-query rankings (index 0 since it's the only sub)
        [
            ["t1", "t2", "t3"],   # query 0
            ["t3", "t1"],          # query 1
        ]
    ]
    labels = ["sasrec_seq"]
    batch_retrieval_items = [
        ["t1", "t3", "t_outside"],  # query 0 — t_outside not in sasrec
        ["t3", "t2", "t1"],         # query 1 — t2 not in sasrec for q1
    ]

    result = build_sasrec_extra_features(per_sub, labels, batch_retrieval_items)
    assert result is not None
    assert len(result) == 2, "Two lists — one per query"
    assert len(result[0]) == 3, "Query 0 has 3 candidates"
    assert len(result[1]) == 3, "Query 1 has 3 candidates"

    # Query 0: t1 rank=1, t3 rank=3, t_outside sentinel
    assert result[0][0] == {"sasrec_rank": 1},     f"q0 t1 rank should be 1, got {result[0][0]}"
    assert result[0][1] == {"sasrec_rank": 3},     f"q0 t3 rank should be 3, got {result[0][1]}"
    assert result[0][2] == {"sasrec_rank": 10000}, f"q0 t_outside should be sentinel, got {result[0][2]}"

    # Query 1: t3 rank=1, t2 not in sasrec for q1 -> sentinel, t1 rank=2
    assert result[1][0] == {"sasrec_rank": 1},     f"q1 t3 rank should be 1, got {result[1][0]}"
    assert result[1][1] == {"sasrec_rank": 10000}, f"q1 t2 not ranked -> sentinel, got {result[1][1]}"
    assert result[1][2] == {"sasrec_rank": 2},     f"q1 t1 rank should be 2, got {result[1][2]}"


# ---------------------------------------------------------------------------
# Test 5: custom sentinel value is forwarded correctly
# ---------------------------------------------------------------------------

def test_custom_sentinel_forwarded():
    """Callers can override the default sentinel (10000) with any int."""
    from mcrs.crs_baseline import build_sasrec_extra_features

    per_sub = [[["t_a"]]]
    labels = ["sasrec_seq"]
    batch_retrieval_items = [["t_a", "t_missing"]]

    result = build_sasrec_extra_features(
        per_sub, labels, batch_retrieval_items, sentinel=99999)
    assert result is not None
    assert result[0][1] == {"sasrec_rank": 99999}, (
        f"Expected custom sentinel 99999 for t_missing, got {result[0][1]}"
    )


# ---------------------------------------------------------------------------
# Test 6: multi-sub per_sub — correct sasrec_seq sub is selected by label
# ---------------------------------------------------------------------------

def test_correct_sub_selected_when_multiple_subs():
    """When per_sub has multiple subs, the 'sasrec_seq' sub is identified by
    its position in labels, not assumed to be first."""
    from mcrs.crs_baseline import build_sasrec_extra_features

    # Three subs: bm25 (idx 0), dense (idx 1), sasrec_seq (idx 2)
    bm25_rankings = [["t_bm25_top", "t_common"]]
    dense_rankings = [["t_common", "t_dense_top"]]
    sasrec_rankings = [["t_sasrec_top", "t_common"]]
    per_sub = [bm25_rankings, dense_rankings, sasrec_rankings]
    labels = ["bm25", "dense", "sasrec_seq"]
    batch_retrieval_items = [["t_sasrec_top", "t_common", "t_bm25_only"]]

    result = build_sasrec_extra_features(per_sub, labels, batch_retrieval_items)
    assert result is not None
    q0 = result[0]
    # t_sasrec_top is rank 1 in sasrec_seq sub
    assert q0[0] == {"sasrec_rank": 1}, f"t_sasrec_top should be rank 1, got {q0[0]}"
    # t_common is rank 2 in sasrec_seq sub
    assert q0[1] == {"sasrec_rank": 2}, f"t_common should be rank 2, got {q0[1]}"
    # t_bm25_only not in sasrec_seq -> sentinel
    assert q0[2] == {"sasrec_rank": 10000}, f"t_bm25_only should be sentinel, got {q0[2]}"
