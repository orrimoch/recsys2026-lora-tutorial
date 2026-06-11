"""W3.a: ColBERT full-catalog recall channel union gating (regression pin).

Stage B (2026-06-11) showed ColbertIndexRetriever (PLAID over the 47k catalog)
rescues ~10% of new-artist WALL golds the union missed (+0.05 recall@100 ceiling) —
the FIRST content channel to crack the wall. This wires it as an opt-in union
channel (`use_colbert`, default off) so it can be RRF-fused + gated on nDCG. The
index is query-independent and cold-firable on Blind (needs a prebuilt PLAID index).
"""
from mcrs.retrieval_modules import _wrrf_union_v1_specs


def test_colbert_channel_off_by_default():
    types = [s["type"] for s in _wrrf_union_v1_specs({})]
    assert "colbert_index" not in types


def test_use_colbert_appends_channel_with_default_weight():
    specs = _wrrf_union_v1_specs({"use_colbert": True})
    cb = [s for s in specs if s["type"] == "colbert_index"]
    assert len(cb) == 1
    assert cb[0]["weight"] == 1.0
    assert cb[0]["topk_internal"] == 100


def test_w_colbert_overrides_weight():
    specs = _wrrf_union_v1_specs({"use_colbert": True, "w_colbert": 0.5})
    cb = [s for s in specs if s["type"] == "colbert_index"]
    assert cb[0]["weight"] == 0.5


def test_colbert_channel_threads_index_config():
    specs = _wrrf_union_v1_specs({
        "use_colbert": True,
        "colbert_index_folder": "/x/plaid",
        "colbert_index_name": "my-index",
        "colbert_model": "/x/music-colbert-v1",
    })
    cb = [s for s in specs if s["type"] == "colbert_index"][0]
    ec = cb["extra_config"]
    assert ec["colbert_index_folder"] == "/x/plaid"
    assert ec["colbert_index_name"] == "my-index"
    assert ec["colbert_model"] == "/x/music-colbert-v1"

    # default index name when not provided
    specs2 = _wrrf_union_v1_specs({"use_colbert": True})
    cb2 = [s for s in specs2 if s["type"] == "colbert_index"][0]
    assert cb2["extra_config"]["colbert_index_name"] == "colbert-music-v1"
