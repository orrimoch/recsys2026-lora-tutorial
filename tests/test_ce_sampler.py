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
