"""Tier-1 #3.4: cf-bpr user x item channel gating (regression pin).

The cf_bpr recall channel + its use_cfbpr union gate already exist but were
untested. cf_bpr is cold-firable on Blind (query-independent, uses user_id;
fires for warm users, empty -> 0 RRF contribution for cold users, so no
regression). Frozen precomputed embeddings -> leak-free as a recall channel
(distinct from the dropped LGBM cfbpr_score feature). This pins the opt-in
default-off gating so it can't silently break.
"""
from mcrs.retrieval_modules import _wrrf_union_v1_specs


def test_cfbpr_channel_off_by_default():
    types = [s["type"] for s in _wrrf_union_v1_specs({})]
    assert "cf_bpr" not in types


def test_use_cfbpr_appends_channel_with_default_weight():
    specs = _wrrf_union_v1_specs({"use_cfbpr": True})
    cf = [s for s in specs if s["type"] == "cf_bpr"]
    assert len(cf) == 1
    assert cf[0]["weight"] == 0.25  # low default: only ~43% of users are warm
    assert cf[0]["topk_internal"] == 100


def test_w_cfbpr_overrides_weight():
    specs = _wrrf_union_v1_specs({"use_cfbpr": True, "w_cfbpr": 0.5})
    cf = [s for s in specs if s["type"] == "cf_bpr"]
    assert cf[0]["weight"] == 0.5
