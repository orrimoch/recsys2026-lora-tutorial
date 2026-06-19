# tests/test_ce_sampler.py
from mcrs.training.ce_data import sample_negatives

# pool: list of (track_id, rank) already sorted by rank ascending (1=best)
POOL = [(f"t{i}", i) for i in range(1, 41)]   # t1..t40
ARTIST = {f"t{i}": ("same" if i == 5 else f"art{i}") for i in range(1, 41)}
TITLE = {f"t{i}": f"title{i}" for i in range(1, 41)}

def _kw(**o):
    base = dict(gold_tid="g", gold_title="gold", gold_artist="same",
                artist_fn=ARTIST.get, title_fn=TITLE.get, n=10, k_min=4, seed=0)
    base.update(o); return base

def test_deterministic_given_seed():
    a = sample_negatives(POOL, **_kw())
    b = sample_negatives(POOL, **_kw())
    assert a == b

def test_skip_top_rank_off_keeps_rank1():
    negs = sample_negatives(POOL, **_kw(n=40, skip_top_rank=False))
    assert "t1" in negs

def test_skip_top_rank_on_drops_rank1():
    negs = sample_negatives(POOL, **_kw(n=40, skip_top_rank=True))
    assert "t1" not in negs

def test_same_artist_downweighted_not_absent_over_runs():
    # 't5' is same-artist; with soft down-weight it should be selectable but rarer than a normal mid negative.
    seen = sum("t5" in sample_negatives(POOL, **_kw(seed=s, same_artist="soft_downweight")) for s in range(50))
    assert 0 < seen < 50

def test_same_artist_drop_removes_it():
    negs = sample_negatives(POOL, **_kw(n=40, same_artist="drop"))
    assert "t5" not in negs

def test_returns_at_least_k_min_when_pool_allows():
    negs = sample_negatives(POOL, **_kw(n=10))
    assert len(negs) == 10


# ---------------------------------------------------------------------------
# Hardening tests
# ---------------------------------------------------------------------------

def test_pool_smaller_than_n_returns_all_eligible_no_crash():
    """When pool has fewer eligible items than n, return all eligible (no crash, len < n)."""
    small_pool = [("t1", 1), ("t2", 2)]
    negs = sample_negatives(small_pool, gold_tid="g", gold_title="gold", gold_artist="other",
                            artist_fn=lambda tid: "diff", title_fn=lambda tid: f"title_{tid}",
                            n=10, k_min=1, seed=0)
    assert len(negs) == 2                     # only 2 available
    assert set(negs) == {"t1", "t2"}


def test_all_same_artist_soft_downweight_returns_some():
    """Pool of all same-artist items with soft_downweight must still return some negatives."""
    pool = [(f"t{i}", i) for i in range(1, 6)]
    negs = sample_negatives(pool, gold_tid="g", gold_title="gold", gold_artist="same_art",
                            artist_fn=lambda tid: "same_art",   # everything is same-artist
                            title_fn=lambda tid: f"title_{tid}",
                            n=3, k_min=1, seed=0, same_artist="soft_downweight")
    assert len(negs) > 0


def test_all_same_artist_drop_returns_empty():
    """Pool of all same-artist items with same_artist='drop' must return []."""
    pool = [(f"t{i}", i) for i in range(1, 6)]
    negs = sample_negatives(pool, gold_tid="g", gold_title="gold", gold_artist="same_art",
                            artist_fn=lambda tid: "same_art",   # everything is same-artist
                            title_fn=lambda tid: f"title_{tid}",
                            n=3, k_min=1, seed=0, same_artist="drop")
    assert negs == []


def test_no_duplicate_track_ids_in_output():
    """No track_id must appear twice in the returned negatives."""
    negs = sample_negatives(POOL, **_kw(n=30))
    assert len(negs) == len(set(negs))        # set collapses duplicates if any exist


def test_different_seeds_give_different_selections():
    """Two different seeds on a large pool must (almost always) yield different selections."""
    r1 = sample_negatives(POOL, **_kw(n=5, seed=0))
    r2 = sample_negatives(POOL, **_kw(n=5, seed=999))
    assert r1 != r2


def test_skip_top_rank_removes_rank1_only():
    """skip_top_rank=True must remove exactly rank-1 from consideration, not rank-2+."""
    negs = sample_negatives(POOL, **_kw(n=40, skip_top_rank=True))
    assert "t1" not in negs
    assert "t2" in negs                       # rank-2 must still be eligible


def test_empty_pool_returns_empty():
    """An empty pool must return [] without error."""
    negs = sample_negatives([], gold_tid="g", gold_title="gold", gold_artist="a",
                            artist_fn=lambda tid: None, title_fn=lambda tid: None,
                            n=5, k_min=1, seed=0)
    assert negs == []


def test_pool_all_gold_returns_empty():
    """A pool consisting solely of the gold track_id must return [] (no eligible negatives)."""
    negs = sample_negatives([("g", 1), ("g", 2)], gold_tid="g", gold_title="gold",
                            gold_artist="a", artist_fn=lambda tid: "a",
                            title_fn=lambda tid: "gold",
                            n=5, k_min=1, seed=0)
    assert negs == []


# ----- T1.3: false-negative (teacher) denoise -----
def test_false_negative_drop_set_drops_top_quantile_by_teacher_score():
    from mcrs.training.ce_data import false_negative_drop_set
    tids = ["a", "b", "c", "d", "e"]
    scores = {"a": 0.1, "b": 0.2, "c": 0.9, "d": 0.95, "e": 0.0}   # c,d are teacher-high
    assert false_negative_drop_set(tids, scores, 0.4) == {"c", "d"}   # floor(5*0.4)=2 highest
    assert false_negative_drop_set(tids, scores, 0.0) == set()        # off by default (no-op)
    assert false_negative_drop_set([], scores, 0.5) == set()


def test_sample_negatives_drops_teacher_false_negatives():
    # the rank-1 negative t1 is a likely unlabeled positive (teacher scores it ~gold); with the
    # filter on it must never be sampled, even at n>=pool (so it can't be a sampling fluke).
    neg_scores = {f"t{i}": (10.0 if i == 1 else 0.0) for i in range(1, 41)}
    with_filter = sample_negatives(POOL, **_kw(n=40, neg_scores=neg_scores, fp_quantile=0.1))
    assert "t1" not in with_filter
    # default (no scores / quantile 0) keeps it -> proves the filter, not some other rule, removed it
    assert "t1" in sample_negatives(POOL, **_kw(n=40))
