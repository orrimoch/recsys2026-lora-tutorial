import yaml


def test_config_190_union_lgbm_no_crossencoder():
    cfg = yaml.safe_load(open(
        "music-crs-baselines/config/190-union-lgbm-v5kto-blindA.yaml"))
    assert cfg["retrieval_type"] == "wrrf_union_v1"
    assert cfg["reranker_type"] == "lgbm_rerank"       # real registered key (not "lgbm")
    assert "multimodal_cross_encoder" not in str(cfg)  # cross-encoder dropped
    assert cfg["retrieval_topk"] >= 100
