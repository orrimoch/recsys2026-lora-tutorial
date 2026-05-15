"""Tests for the Phase 0 retrieval-diagnostic pure-function units.

Phase 0 measures where the current retrieval pipeline (BM25 + dense + fused/reranked)
fails on the dev split. These tests cover the categorization and slicing helpers that
turn raw per-component top-K lists into a failure-mode breakdown.
"""
from tests.conftest import _tid


def test_categorize_miss_returns_hit_when_gold_in_fused_top_k():
    """If gold is in the fused/reranked top-K, the turn is a hit, not a miss."""
    from scripts.phase0_retrieval_diagnostic import categorize_miss

    gold = _tid("g", 1)
    fused = [_tid("x", i) for i in range(20)]
    fused[5] = gold  # gold sits at rank 5 in fused top-20

    bm25_top100 = [_tid("b", i) for i in range(100)]
    dense_top100 = [_tid("d", i) for i in range(100)]

    result = categorize_miss(
        gold_id=gold,
        bm25_top_k=bm25_top100,
        dense_top_k=dense_top100,
        fused_top_k=fused,
    )

    assert result == "hit_in_top_k"


def test_categorize_miss_returns_not_in_bm25_when_only_dense_has_gold():
    """Gold absent from BM25 top-100 but present in dense top-100, missed by fused → BM25 is the weak link."""
    from scripts.phase0_retrieval_diagnostic import categorize_miss

    gold = _tid("g", 1)
    bm25_top100 = [_tid("b", i) for i in range(100)]  # no gold
    dense_top100 = [_tid("d", i) for i in range(100)]
    dense_top100[42] = gold
    fused = [_tid("x", i) for i in range(20)]  # no gold

    result = categorize_miss(
        gold_id=gold,
        bm25_top_k=bm25_top100,
        dense_top_k=dense_top100,
        fused_top_k=fused,
    )

    assert result == "not_in_bm25_only"


def test_categorize_miss_returns_not_in_dense_when_only_bm25_has_gold():
    """Gold present in BM25 top-100 but absent from dense top-100, missed by fused → dense is the weak link."""
    from scripts.phase0_retrieval_diagnostic import categorize_miss

    gold = _tid("g", 2)
    bm25_top100 = [_tid("b", i) for i in range(100)]
    bm25_top100[17] = gold
    dense_top100 = [_tid("d", i) for i in range(100)]  # no gold
    fused = [_tid("x", i) for i in range(20)]  # no gold

    result = categorize_miss(
        gold_id=gold,
        bm25_top_k=bm25_top100,
        dense_top_k=dense_top100,
        fused_top_k=fused,
    )

    assert result == "not_in_dense_only"


def test_categorize_miss_returns_in_both_low_rank_when_both_have_gold_but_fused_doesnt():
    """Gold in BOTH top-100s but missed by fused → fusion/rerank is the weak link, not retrieval."""
    from scripts.phase0_retrieval_diagnostic import categorize_miss

    gold = _tid("g", 3)
    bm25_top100 = [_tid("b", i) for i in range(100)]
    bm25_top100[80] = gold
    dense_top100 = [_tid("d", i) for i in range(100)]
    dense_top100[75] = gold
    fused = [_tid("x", i) for i in range(20)]  # no gold

    result = categorize_miss(
        gold_id=gold,
        bm25_top_k=bm25_top100,
        dense_top_k=dense_top100,
        fused_top_k=fused,
    )

    assert result == "in_both_low_rank"


def test_categorize_miss_returns_not_in_either_when_gold_absent_from_both_top_100():
    """Gold absent from BM25 AND dense top-100 → both retrievers are too weak; embedder/preprocessing needed."""
    from scripts.phase0_retrieval_diagnostic import categorize_miss

    gold = _tid("g", 4)
    bm25_top100 = [_tid("b", i) for i in range(100)]  # no gold
    dense_top100 = [_tid("d", i) for i in range(100)]  # no gold
    fused = [_tid("x", i) for i in range(20)]  # no gold

    result = categorize_miss(
        gold_id=gold,
        bm25_top_k=bm25_top100,
        dense_top_k=dense_top100,
        fused_top_k=fused,
    )

    assert result == "not_in_either"


# ---- bucket_query_length: short (<10 words), medium (10-30), long (>30) ----

def test_bucket_query_length_short_under_ten_words():
    from scripts.phase0_retrieval_diagnostic import bucket_query_length
    assert bucket_query_length("play me something upbeat") == "short"


def test_bucket_query_length_medium_at_ten_words():
    from scripts.phase0_retrieval_diagnostic import bucket_query_length
    q = " ".join(["word"] * 10)
    assert bucket_query_length(q) == "medium"


def test_bucket_query_length_medium_at_thirty_words():
    from scripts.phase0_retrieval_diagnostic import bucket_query_length
    q = " ".join(["word"] * 30)
    assert bucket_query_length(q) == "medium"


def test_bucket_query_length_long_over_thirty_words():
    from scripts.phase0_retrieval_diagnostic import bucket_query_length
    q = " ".join(["word"] * 31)
    assert bucket_query_length(q) == "long"


# ---- bucket_depth: "1" (first turn), "2-4" (mid), "5+" (deep) ----

def test_bucket_depth_first_turn_returns_1():
    from scripts.phase0_retrieval_diagnostic import bucket_depth
    assert bucket_depth(1) == "1"


def test_bucket_depth_turn_two_returns_mid_band():
    from scripts.phase0_retrieval_diagnostic import bucket_depth
    assert bucket_depth(2) == "2-4"


def test_bucket_depth_turn_four_returns_mid_band():
    from scripts.phase0_retrieval_diagnostic import bucket_depth
    assert bucket_depth(4) == "2-4"


def test_bucket_depth_turn_five_returns_deep_band():
    from scripts.phase0_retrieval_diagnostic import bucket_depth
    assert bucket_depth(5) == "5+"


# ---- has_artist_mention: heuristic — "by <Capitalized>" or quoted-string pattern ----

def test_has_artist_mention_true_for_by_capitalized():
    from scripts.phase0_retrieval_diagnostic import has_artist_mention
    assert has_artist_mention("play something by Taylor Swift") is True


def test_has_artist_mention_true_for_quoted_string():
    from scripts.phase0_retrieval_diagnostic import has_artist_mention
    assert has_artist_mention('I want "Bohemian Rhapsody" please') is True


def test_has_artist_mention_false_for_abstract_query():
    from scripts.phase0_retrieval_diagnostic import has_artist_mention
    assert has_artist_mention("something upbeat for a rainy sunday morning") is False


def test_has_artist_mention_false_for_lowercase_by():
    """'by' with lowercase noun shouldn't trigger — that's prepositional, not attribution."""
    from scripts.phase0_retrieval_diagnostic import has_artist_mention
    assert has_artist_mention("walk by the river") is False


# ---- summarize_diagnostic: aggregate per-turn records into final report ----

def _make_record(*, gold, bm25, dense, fused, query="play me something", turn=1, sid="s1"):
    return {
        "session_id": sid,
        "turn_number": turn,
        "user_query": query,
        "gold_id": gold,
        "bm25_top_100": bm25,
        "dense_top_100": dense,
        "fused_top_k": fused,
    }


def test_summarize_diagnostic_counts_n_turns_and_top_level_shape():
    from scripts.phase0_retrieval_diagnostic import summarize_diagnostic

    gold = _tid("g", 0)
    fused = [gold] + [_tid("x", i) for i in range(19)]
    rec = _make_record(gold=gold, bm25=[gold], dense=[gold], fused=fused)

    result = summarize_diagnostic([rec])

    assert result["n_turns"] == 1
    assert "per_component_metrics" in result
    assert "failure_breakdown" in result
    assert "slice_breakdown" in result


def test_summarize_diagnostic_failure_breakdown_counts_hits():
    """Two hits + one miss-not-in-either → counts split correctly."""
    from scripts.phase0_retrieval_diagnostic import summarize_diagnostic

    g_hit_a, g_hit_b, g_miss = _tid("g", 0), _tid("g", 1), _tid("g", 2)
    other = [_tid("x", i) for i in range(19)]
    fused_hit_a = [g_hit_a] + other
    fused_hit_b = [g_hit_b] + other
    fused_miss = other + [_tid("x", 19)]  # 20 distractors, no gold

    recs = [
        _make_record(gold=g_hit_a, bm25=[g_hit_a], dense=[g_hit_a], fused=fused_hit_a, sid="s1"),
        _make_record(gold=g_hit_b, bm25=[g_hit_b], dense=[g_hit_b], fused=fused_hit_b, sid="s2"),
        _make_record(gold=g_miss, bm25=[_tid("o", 0)], dense=[_tid("o", 1)], fused=fused_miss, sid="s3"),
    ]

    result = summarize_diagnostic(recs)

    assert result["failure_breakdown"]["hit_in_top_k"] == 2
    assert result["failure_breakdown"]["not_in_either"] == 1


def test_summarize_diagnostic_per_component_recall_at_20_for_fused():
    """3 records, 2 hits in fused top-20 → fused recall@20 = 2/3."""
    from scripts.phase0_retrieval_diagnostic import summarize_diagnostic

    g0, g1, g2 = _tid("g", 0), _tid("g", 1), _tid("g", 2)
    other = [_tid("x", i) for i in range(19)]
    rec_hit_a = _make_record(gold=g0, bm25=[g0], dense=[g0], fused=[g0] + other, sid="s1")
    rec_hit_b = _make_record(gold=g1, bm25=[g1], dense=[g1], fused=[g1] + other, sid="s2")
    rec_miss = _make_record(gold=g2, bm25=[_tid("o", 0)], dense=[_tid("o", 1)], fused=other + [_tid("x", 19)], sid="s3")

    result = summarize_diagnostic([rec_hit_a, rec_hit_b, rec_miss])

    assert abs(result["per_component_metrics"]["fused"]["recall@20"] - 2 / 3) < 1e-9


def test_summarize_diagnostic_slice_breakdown_groups_by_artist_mention():
    """Two records, one with 'by Capitalized' query → slice breakdown should have yes/no buckets with counts."""
    from scripts.phase0_retrieval_diagnostic import summarize_diagnostic

    g0, g1 = _tid("g", 0), _tid("g", 1)
    other = [_tid("x", i) for i in range(19)]
    rec_artist = _make_record(
        gold=g0, bm25=[g0], dense=[g0], fused=[g0] + other,
        query="play something by Taylor Swift", sid="s1",
    )
    rec_abstract = _make_record(
        gold=g1, bm25=[g1], dense=[g1], fused=[g1] + other,
        query="something upbeat for sunday morning", sid="s2",
    )

    result = summarize_diagnostic([rec_artist, rec_abstract])

    artist_slice = result["slice_breakdown"]["artist_mention"]
    assert artist_slice["yes"]["n"] == 1
    assert artist_slice["no"]["n"] == 1
    assert abs(artist_slice["yes"]["recall@20"] - 1.0) < 1e-9
    assert abs(artist_slice["no"]["recall@20"] - 1.0) < 1e-9


# ---- per-turn records JSONL I/O (consumed by compare_diagnostic_runs) ----

def test_write_then_read_records_jsonl_roundtrip(tmp_path):
    """Records survive write→read unchanged so the comparison script can consume them."""
    from scripts.phase0_retrieval_diagnostic import (
        read_records_jsonl,
        write_records_jsonl,
    )

    records = [
        {
            "session_id": "s1",
            "turn_number": 1,
            "user_query": "play me something",
            "gold_id": _tid("g", 0),
            "bm25_top_100": [_tid("b", i) for i in range(3)],
            "dense_top_100": [_tid("d", i) for i in range(3)],
            "fused_top_k": [_tid("f", i) for i in range(3)],
        },
        {
            "session_id": "s2",
            "turn_number": 4,
            "user_query": "by Taylor Swift",
            "gold_id": _tid("g", 1),
            "bm25_top_100": [_tid("b", i) for i in range(3)],
            "dense_top_100": [_tid("d", i) for i in range(3)],
            "fused_top_k": [_tid("f", i) for i in range(3)],
        },
    ]

    path = tmp_path / "records.jsonl"
    write_records_jsonl(records, path)
    loaded = list(read_records_jsonl(path))

    assert loaded == records
