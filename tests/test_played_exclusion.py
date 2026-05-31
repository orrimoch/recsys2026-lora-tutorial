"""Bug #2 fix: exclude already-played tracks from the served top-20.

The gold is ALWAYS a brand-new track (0/8000 dev golds were previously played),
so a played track occupying a top-20 slot is a guaranteed miss. Dropping it can
only promote real candidates -> provably non-decreasing for nDCG@20. The fix
lives in CRS_BASELINE._finalize_topk (dedupe + catalog-filter + played-exclusion
+ backfill), now that bug #1 makes history_tids non-empty at serve.
"""
from mcrs.crs_baseline import CRS_BASELINE


def _bare(valid_catalog):
    obj = CRS_BASELINE.__new__(CRS_BASELINE)  # skip __init__ (no model load)
    obj._valid_catalog = valid_catalog
    return obj


def test_excludes_played_and_backfills_from_pool():
    obj = _bare({"t1", "t2", "t3", "t4"})
    out = obj._finalize_topk(["t1", "t2", "t3"], ["t4"], played_set={"t2"}, k=20)
    assert "t2" not in out
    assert out == ["t1", "t3", "t4"]  # t2 dropped, t4 backfilled from pool


def test_dedupes_keep_first():
    obj = _bare({"t1", "t2"})
    assert obj._finalize_topk(["t1", "t1", "t2"], [], set(), k=20) == ["t1", "t2"]


def test_catalog_filter_drops_unknown_ids():
    obj = _bare({"t1"})
    assert obj._finalize_topk(["t1", "tX"], [], set(), k=20) == ["t1"]


def test_backfill_respects_k():
    obj = _bare({"a", "b", "c"})
    assert obj._finalize_topk(["a"], ["b", "c"], set(), k=2) == ["a", "b"]


def test_played_track_in_pool_also_excluded_on_backfill():
    obj = _bare({"a", "p", "b"})
    out = obj._finalize_topk(["a"], ["p", "b"], {"p"}, k=20)
    assert out == ["a", "b"]  # 'p' played -> skipped even during backfill


def test_none_catalog_still_dedupes_and_excludes_played():
    obj = _bare(None)  # db lacked metadata_dict
    assert obj._finalize_topk(["t1", "t1", "t2"], [], {"t2"}, k=20) == ["t1"]


def test_non_played_gold_is_never_dropped():
    obj = _bare({"gold", "p1", "p2"})
    # gold sits between two played tracks; exclusion moves it to the top.
    out = obj._finalize_topk(["p1", "gold", "p2"], [], {"p1", "p2"}, k=20)
    assert out == ["gold"]


def test_empty_played_set_is_a_noop_beyond_dedupe_and_catalog():
    obj = _bare({"a", "b", "c"})
    assert obj._finalize_topk(["a", "b", "c"], [], set(), k=20) == ["a", "b", "c"]
