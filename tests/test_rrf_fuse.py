"""wRRF re-fusion helpers — let the fusion-weight sweep re-fuse cached
per-sub rankings without re-running the sub-retrievers."""
from mcrs.retrieval_modules.rrf import RRF_MODEL


class _Stub:
    """Minimal sub-retriever returning a fixed ranked list, truncated to topk."""
    def __init__(self, ranked):
        self.ranked = ranked

    def batch_text_to_item_retrieval(self, queries, topk, **kw):
        return [self.ranked[:topk] for _ in queries]


def _rrf(subs, k=60):
    r = RRF_MODEL.__new__(RRF_MODEL)  # bypass __init__ (it loads real models)
    r.k = k
    r.subs = subs
    return r


def test_fuse_per_sub_equal_weights_is_symmetric():
    per_sub = [
        [["a", "b", "c"]],   # sub 0, query 0
        [["b", "a", "d"]],   # sub 1, query 0
    ]
    out = RRF_MODEL.fuse_per_sub(per_sub, weights=[1.0, 1.0], k=60, topk=4)
    assert set(out[0][:2]) == {"a", "b"}   # a and b tie above c/d
    assert len(out[0]) == 4


def test_fuse_per_sub_weight_promotes_that_subs_top():
    per_sub = [
        [["x", "y", "z"]],   # sub 0 favors x
        [["q", "r", "s"]],   # sub 1 favors q
    ]
    out_b = RRF_MODEL.fuse_per_sub(per_sub, weights=[1.0, 100.0], k=60, topk=1)
    assert out_b[0][0] == "q"
    out_a = RRF_MODEL.fuse_per_sub(per_sub, weights=[100.0, 1.0], k=60, topk=1)
    assert out_a[0][0] == "x"


def test_fuse_per_sub_truncates_to_topk():
    per_sub = [[["a", "b", "c", "d", "e"]]]
    out = RRF_MODEL.fuse_per_sub(per_sub, weights=[1.0], k=60, topk=3)
    assert out[0] == ["a", "b", "c"]


def test_batch_per_sub_rankings_honors_per_sub_topk_and_labels():
    r = _rrf([
        {"retriever": _Stub(["a", "b", "c"]), "topk": 2, "weight": 1.0, "label": "s0"},
        {"retriever": _Stub(["d", "e", "f"]), "topk": 3, "weight": 0.5, "label": "s1"},
    ])
    per_sub, labels = r.batch_per_sub_rankings(["q1", "q2"])
    assert labels == ["s0", "s1"]
    assert per_sub[0] == [["a", "b"], ["a", "b"]]
    assert per_sub[1] == [["d", "e", "f"], ["d", "e", "f"]]


def test_refactored_batch_retrieval_matches_manual_fuse():
    r = _rrf([
        {"retriever": _Stub(["a", "b", "c"]), "topk": 3, "weight": 1.0, "label": "s0"},
        {"retriever": _Stub(["b", "c", "d"]), "topk": 3, "weight": 2.0, "label": "s1"},
    ])
    out = r.batch_text_to_item_retrieval(["q"], topk=4)
    per_sub, _ = r.batch_per_sub_rankings(["q"])
    fused = RRF_MODEL.fuse_per_sub(per_sub, [1.0, 2.0], 60, 4)
    assert out == fused
