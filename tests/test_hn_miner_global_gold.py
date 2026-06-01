"""Clean negatives: no query's gold may be mined as ANOTHER query's hard
negative. v1 excluded only a row's own gold, so a true positive elsewhere could
be trained as a false negative (unmasked across batches). global_gold_ids fixes
it by removing the whole gold set from every negative pool.
"""
import numpy as np
from mcrs.retrieval_modules.hn_miner import batch_mine_negatives


def _unit(rows):
    a = np.asarray(rows, dtype=np.float32)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


# query ~ t1 (gold); t2 is ANOTHER query's gold sitting high in similarity; t3/t4 clean.
_Q = _unit([[1.0, 0.0, 0.0]])
_TRACKS = _unit([[1.0, 0.0, 0.0], [0.95, 0.05, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
_TIDS = ["t1", "t2", "t3", "t4"]


def test_global_gold_excluded_from_negatives():
    out = batch_mine_negatives(_Q, _TRACKS, _TIDS, ["t1"], strategy="simans",
                               k_negs=2, pool_size=10, use_gpu=False,
                               global_gold_ids={"t1", "t2"})
    negs = out[0]
    assert "t2" not in negs   # another query's gold -> not a negative here
    assert "t1" not in negs   # own gold always excluded
    assert set(negs) <= {"t3", "t4"}


def test_none_preserves_v1_behavior_t2_minable():
    # Without global exclusion, t2 (high-sim non-own-gold) is a legal hard neg.
    out = batch_mine_negatives(_Q, _TRACKS, _TIDS, ["t1"], strategy="simans",
                               k_negs=3, pool_size=10, use_gpu=False,
                               global_gold_ids=None)
    negs = out[0]
    assert "t1" not in negs
    assert "t2" in set(negs)  # t2 is now a (false) negative — the v1 bug
