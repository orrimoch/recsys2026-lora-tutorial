"""EXP-012 — rescue-aware / channel-quota fusion.

Pure-function + dispatcher tests for RRF_MODEL.fuse_per_sub_quota: reserve a few
top slots for orthogonal wall-cracker channels so their single-channel rescues
are guaranteed into the reranker window WITHOUT down-weighting (the failure mode
of config 207's up-weighting). Design corrections from the RecSys design-review
are encoded here: insert reserved at the window TAIL (minimal eviction), only
inject candidates not already in the window, round-robin fairness, dedup.
"""
from mcrs.retrieval_modules.rrf import RRF_MODEL


class _Stub:
    """Minimal sub-retriever returning a fixed ranked list, truncated to topk."""
    def __init__(self, ranked):
        self.ranked = ranked

    def batch_text_to_item_retrieval(self, queries, topk, **kw):
        return [self.ranked[:topk] for _ in queries]


def _rrf(subs, k=60, **attrs):
    r = RRF_MODEL.__new__(RRF_MODEL)  # bypass __init__ (it loads real models)
    r.k = k
    r.subs = subs
    # quota attrs default to the symmetric (no-op) configuration
    r.fusion_strategy = attrs.get("fusion_strategy", "symmetric")
    r.channel_quota = attrs.get("channel_quota", 0)
    r.quota_labels = attrs.get("quota_labels", None)
    r.quota_window = attrs.get("quota_window", 50)
    return r


# ---- pure fuse_per_sub_quota -------------------------------------------------

def test_quota_promotes_a_dropped_single_channel_rescue_into_the_window():
    # ch0, ch1 identical -> a,b,c,x,y,z are all 2-channel and dominate RRF.
    # ch2 (quota) contributes "rescue" as a rank-1 SINGLE-channel candidate that
    # pure RRF ranks below the window (rank 7), i.e. the reranker never sees it.
    per_sub = [
        [["a", "b", "c", "x", "y", "z"]],
        [["a", "b", "c", "x", "y", "z"]],
        [["rescue", "h", "i"]],
    ]
    weights = [1.0, 1.0, 1.0]
    pure = RRF_MODEL.fuse_per_sub(per_sub, weights, k=60, topk=7)
    assert "rescue" not in pure[0][:3]  # baseline: dropped from the window

    out = RRF_MODEL.fuse_per_sub_quota(
        per_sub, weights, k=60, topk=7, quota_by_idx={2: 1}, window=3)
    assert "rescue" in out[0][:3]            # promoted into the window
    assert out[0][0] == "a" and out[0][1] == "b"  # strongest RRF preserved at front
    assert out[0][2] == "rescue"             # inserted at the window TAIL


def test_quota_displaces_only_the_lowest_rrf_window_item():
    per_sub = [
        [["a", "b", "c", "x", "y", "z"]],
        [["a", "b", "c", "x", "y", "z"]],
        [["rescue"]],
    ]
    weights = [1.0, 1.0, 1.0]
    out = RRF_MODEL.fuse_per_sub_quota(
        per_sub, weights, k=60, topk=7, quota_by_idx={2: 1}, window=3)
    # "c" was the lowest-RRF item in the top-3 window -> pushed to just past it.
    assert out[0][3] == "c"
    # nothing is lost: the displaced item and the rest follow.
    assert set(out[0]) == {"a", "b", "c", "x", "y", "z", "rescue"}


def test_quota_is_noop_when_rescue_already_in_window():
    per_sub = [
        [["a", "b", "c", "x"]],
        [["a", "b", "c", "x"]],
        [["a"]],  # "a" is already rank-1 in the fused window
    ]
    weights = [1.0, 1.0, 1.0]
    pure = RRF_MODEL.fuse_per_sub(per_sub, weights, k=60, topk=4)
    out = RRF_MODEL.fuse_per_sub_quota(
        per_sub, weights, k=60, topk=4, quota_by_idx={2: 1}, window=3)
    assert out == pure  # no spurious reordering


def test_quota_dedupes_a_rescue_shared_across_two_quota_channels():
    # window=1 keeps r1 (a single-rank-1 rescue shared by two quota channels) out
    # of the fused window, so the reserve path genuinely runs for both channels.
    per_sub = [
        [["a", "b", "c", "x", "y", "z"]],
        [["a", "b", "c", "x", "y", "z"]],
        [["r1", "r2"]],   # quota
        [["r1", "r3"]],   # quota, shares r1
    ]
    weights = [1.0, 1.0, 1.0, 1.0]
    out = RRF_MODEL.fuse_per_sub_quota(
        per_sub, weights, k=60, topk=8, quota_by_idx={2: 1, 3: 1}, window=1)
    assert out[0].count("r1") == 1   # reserved once despite two channels
    assert out[0][0] == "r1"         # promoted into the (size-1) window


def test_quota_round_robin_gives_each_channel_a_slot():
    # two quota channels, q=1 each, both rescues dropped by RRF -> BOTH must land
    # in the window (round-robin), not just the first channel.
    per_sub = [
        [["a", "b", "c", "d", "e", "f"]],
        [["a", "b", "c", "d", "e", "f"]],
        [["R2"]],
        [["R3"]],
    ]
    weights = [1.0, 1.0, 1.0, 1.0]
    out = RRF_MODEL.fuse_per_sub_quota(
        per_sub, weights, k=60, topk=8, quota_by_idx={2: 1, 3: 1}, window=4)
    assert "R2" in out[0][:4] and "R3" in out[0][:4]


def test_quota_empty_is_identical_to_pure_rrf():
    per_sub = [
        [["a", "b", "c"]],
        [["b", "a", "d"]],
    ]
    weights = [1.0, 1.0]
    pure = RRF_MODEL.fuse_per_sub(per_sub, weights, k=60, topk=4)
    out = RRF_MODEL.fuse_per_sub_quota(
        per_sub, weights, k=60, topk=4, quota_by_idx={}, window=3)
    assert out == pure


def test_quota_respects_topk_cap():
    per_sub = [
        [["a", "b", "c", "x", "y", "z"]],
        [["a", "b", "c", "x", "y", "z"]],
        [["rescue"]],
    ]
    weights = [1.0, 1.0, 1.0]
    out = RRF_MODEL.fuse_per_sub_quota(
        per_sub, weights, k=60, topk=3, quota_by_idx={2: 1}, window=3)
    assert len(out[0]) == 3
    assert "rescue" in out[0]  # rescue still makes the (now =topk) window


# ---- dispatcher integration --------------------------------------------------

def test_dispatcher_uses_quota_when_strategy_set():
    subs = [
        {"retriever": _Stub(["a", "b", "c", "x", "y", "z"]), "topk": 6,
         "weight": 1.0, "label": "bm25"},
        {"retriever": _Stub(["a", "b", "c", "x", "y", "z"]), "topk": 6,
         "weight": 1.0, "label": "same_artist"},
        {"retriever": _Stub(["rescue", "h", "i"]), "topk": 6,
         "weight": 1.0, "label": "dense_metadata_e5_instruct_local"},
    ]
    r = _rrf(subs, fusion_strategy="channel_quota", channel_quota=1,
             quota_labels=["dense_metadata_e5_instruct_local"], quota_window=3)
    out = r.batch_text_to_item_retrieval(["q"], topk=7)
    assert "rescue" in out[0][:3]  # quota path active


def test_dispatcher_defaults_to_symmetric():
    subs = [
        {"retriever": _Stub(["a", "b", "c", "x", "y", "z"]), "topk": 6,
         "weight": 1.0, "label": "bm25"},
        {"retriever": _Stub(["a", "b", "c", "x", "y", "z"]), "topk": 6,
         "weight": 1.0, "label": "same_artist"},
        {"retriever": _Stub(["rescue", "h", "i"]), "topk": 6,
         "weight": 1.0, "label": "dense_metadata_e5_instruct_local"},
    ]
    r = _rrf(subs)  # no fusion_strategy -> symmetric
    out = r.batch_text_to_item_retrieval(["q"], topk=7)
    assert "rescue" not in out[0][:3]  # unchanged from pure RRF


def test_dispatcher_quota_only_targets_listed_labels():
    # channel_quota set, but the e5 label is NOT in quota_labels -> no promotion.
    subs = [
        {"retriever": _Stub(["a", "b", "c", "x", "y", "z"]), "topk": 6,
         "weight": 1.0, "label": "bm25"},
        {"retriever": _Stub(["a", "b", "c", "x", "y", "z"]), "topk": 6,
         "weight": 1.0, "label": "same_artist"},
        {"retriever": _Stub(["rescue", "h", "i"]), "topk": 6,
         "weight": 1.0, "label": "dense_metadata_e5_instruct_local"},
    ]
    r = _rrf(subs, fusion_strategy="channel_quota", channel_quota=1,
             quota_labels=["colbert_index"], quota_window=3)  # e5 not listed
    out = r.batch_text_to_item_retrieval(["q"], topk=7)
    assert "rescue" not in out[0][:3]
