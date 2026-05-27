import inspect
from mcrs.retrieval_modules import load_retrieval_module, _wrrf_union_v1_specs


def test_factory_source_registers_new_types():
    src = inspect.getsource(load_retrieval_module)
    for key in ('"same_artist"', '"session_cf"', '"wrrf_union_v1"', '"sasrec_seq"'):
        assert key in src, f"missing factory branch {key}"


def test_union_v1_is_three_channels_without_session_cf():
    types = [s["type"] for s in _wrrf_union_v1_specs({})]
    for t in ("bm25", "dense_metadata_qwen3", "same_artist"):
        assert t in types, f"union missing kept channel {t}"
    # Dropped after G1 ablation (marginal recall@100 +0.002 < 0.01 keep rule).
    assert "session_cf" not in types, "session_cf should be dropped from wrrf_union_v1"
    # Opt-in channels are off by default.
    assert "sasrec_seq" not in types and "hyde_qwen3" not in types
