"""Segment-aware routing: cold queries (no played history) lean toward content
channels; warm queries (with history) lean toward session channels.

The union runs the same sub-retrievers but fuses with per-query weights chosen by
segment — implementing "if no history do A else do B" at the RRF layer. Opt-in
via use_segment_routing; default off keeps the shipped fixed-weight fusion.
"""
from mcrs.retrieval_modules.rrf import RRF_MODEL
from mcrs.retrieval_modules import _wrrf_union_v1_specs


# ---- pure segmented fusion ----

def test_fuse_segmented_picks_weights_by_segment():
    # sub0 ranks a>b (a "content" channel); sub1 ranks b>a (a "session" channel).
    per_sub = [[["a", "b"]], [["b", "a"]]]
    # cold weights favor sub0 (content); warm weights favor sub1 (session).
    cold_w, warm_w = [1.0, 0.0], [0.0, 1.0]

    cold = RRF_MODEL.fuse_per_sub_segmented(per_sub, cold_w, warm_w, k=60, topk=2,
                                            is_warm=[False])
    assert cold[0][0] == "a"   # cold query -> content channel wins

    warm = RRF_MODEL.fuse_per_sub_segmented(per_sub, cold_w, warm_w, k=60, topk=2,
                                            is_warm=[True])
    assert warm[0][0] == "b"   # warm query -> session channel wins


def test_fuse_segmented_equals_plain_when_weights_equal():
    per_sub = [[["a", "b"]], [["b", "a"]]]
    w = [1.0, 1.0]
    seg = RRF_MODEL.fuse_per_sub_segmented(per_sub, w, w, k=60, topk=2, is_warm=[True])
    plain = RRF_MODEL.fuse_per_sub(per_sub, w, k=60, topk=2)
    assert seg == plain


# ---- RRF_MODEL routes by history presence ----

class _FakeSub:
    def __init__(self, ranked):
        self.ranked = ranked

    def batch_text_to_item_retrieval(self, queries, topk, batch_context=None, user_ids=None):
        return [list(self.ranked) for _ in queries]


def _routed_model():
    r = RRF_MODEL.__new__(RRF_MODEL)
    r.k = 60
    r.subs = [
        {"retriever": _FakeSub(["a", "b"]), "topk": 10, "weight": 1.0,
         "cold_weight": 1.0, "warm_weight": 0.0, "label": "content"},
        {"retriever": _FakeSub(["b", "a"]), "topk": 10, "weight": 1.0,
         "cold_weight": 0.0, "warm_weight": 1.0, "label": "session"},
    ]
    return r


def test_rrf_routes_cold_query_to_content():
    r = _routed_model()
    out = r.batch_text_to_item_retrieval(["q"], topk=2, batch_context=[{"history_tids": []}])
    assert out[0][0] == "a"   # no history -> content channel


def test_rrf_routes_warm_query_to_session():
    r = _routed_model()
    out = r.batch_text_to_item_retrieval(["q"], topk=2, batch_context=[{"history_tids": ["t1"]}])
    assert out[0][0] == "b"   # has history -> session channel


# ---- union spec gating ----

def test_segment_routing_off_by_default():
    specs = _wrrf_union_v1_specs({})
    assert all("cold_weight" not in s for s in specs)


def test_segment_routing_sets_cold_warm_weights():
    specs = {s["type"]: s for s in _wrrf_union_v1_specs({"use_segment_routing": True})}
    # content (dense) leans cold; session (same_artist) leans warm.
    dense = specs["dense_metadata_qwen3_instruct"]
    art = specs["same_artist"]
    assert dense["cold_weight"] >= dense["warm_weight"]
    assert art["warm_weight"] > art["cold_weight"]
