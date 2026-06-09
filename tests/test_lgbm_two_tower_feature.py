"""--use-two-tower wiring in build_lgbm_features.

The reranker must train on the SAME union it serves, so enabling the two-tower
channel at feature-build time has to inject `use_two_tower` into the union's
extra_config. The channel's union gating itself (the appended sub-spec) is tested
in mcrs (_wrrf_union_v1_specs); here we test the build-side assembly.
"""
from scripts.build_lgbm_features import _union_extra_config


def test_two_tower_off_by_default():
    assert "use_two_tower" not in _union_extra_config()


def test_use_two_tower_sets_channel_keys():
    extra = _union_extra_config(use_two_tower=True, w_two_tower=0.7,
                                two_tower_model_dir="two_tower_v1")
    assert extra["use_two_tower"] is True
    assert extra["w_two_tower"] == 0.7
    assert extra["two_tower_model_dir"] == "two_tower_v1"


def test_two_tower_composes_with_sasrec_and_routing():
    extra = _union_extra_config(use_sasrec=True, w_sasrec=1.0,
                                use_segment_routing=True, use_two_tower=True)
    assert extra["use_sasrec"] is True
    assert extra["use_segment_routing"] is True
    assert extra["use_two_tower"] is True
