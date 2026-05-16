"""W5 tests: sid_stream_weight + sid_hub_repo YAML override in factory."""
import pytest


def test_factory_sid_generator_respects_hub_repo_override(monkeypatch, tmp_path):
    """When extra_config contains sid_hub_repo, factory passes it to SID_GENERATOR."""
    import pandas as pd

    sid_dir = tmp_path / "sid"
    sid_dir.mkdir()
    pd.DataFrame([
        {"track_id": "t1", "code_1": 0, "code_2": 0, "code_3": 0,
         "popularity": 1.0, "bucket_rank": 0},
    ]).to_parquet(sid_dir / "track_to_sid.parquet")

    captured = {}
    class _StubSidGen:
        def __init__(self, hub_repo, sid_lookup_path, **kwargs):
            captured["hub_repo"] = hub_repo
            captured["sid_lookup_path"] = str(sid_lookup_path)
            captured["kwargs"] = kwargs

    monkeypatch.setattr(
        "mcrs.retrieval_modules.sid_generator.SID_GENERATOR", _StubSidGen,
    )

    from mcrs.retrieval_modules import load_retrieval_module
    load_retrieval_module(
        retrieval_type="sid_generator",
        dataset_name="dummy", track_split_types=[], corpus_types=[],
        cache_dir=str(tmp_path),
        extra_config={"sid_hub_repo": "OrRim123/recsys2026-sid-generator-qwen15b-v2-merged"},
    )

    assert captured["hub_repo"] == "OrRim123/recsys2026-sid-generator-qwen15b-v2-merged"


def test_factory_wrrf_sid_respects_stream_weight_override(monkeypatch, tmp_path):
    """When extra_config contains sid_stream_weight, the wrrf_bm25_dense_sid_v1
    sub_specs has the SID entry with the overridden weight (not the default 0.5)."""
    captured_sub_specs = {}

    class _StubRrf:
        def __init__(self, dataset_name, track_split_types, corpus_types, cache_dir,
                     sub_specs, k):
            captured_sub_specs["specs"] = sub_specs
            captured_sub_specs["k"] = k

    monkeypatch.setattr("mcrs.retrieval_modules.RRF_MODEL", _StubRrf)

    from mcrs.retrieval_modules import load_retrieval_module
    load_retrieval_module(
        retrieval_type="wrrf_bm25_dense_sid_v1",
        dataset_name="dummy", track_split_types=[], corpus_types=[],
        cache_dir=str(tmp_path),
        extra_config={"sid_stream_weight": 0.7},
    )

    sid_specs = [s for s in captured_sub_specs["specs"] if s["type"] == "sid_generator"]
    assert len(sid_specs) == 1
    assert sid_specs[0]["weight"] == 0.7


def test_factory_wrrf_sid_defaults_to_0_5_when_override_absent(monkeypatch, tmp_path):
    """When extra_config lacks sid_stream_weight, default of 0.5 is preserved."""
    captured_sub_specs = {}

    class _StubRrf:
        def __init__(self, dataset_name, track_split_types, corpus_types, cache_dir,
                     sub_specs, k):
            captured_sub_specs["specs"] = sub_specs

    monkeypatch.setattr("mcrs.retrieval_modules.RRF_MODEL", _StubRrf)

    from mcrs.retrieval_modules import load_retrieval_module
    load_retrieval_module(
        retrieval_type="wrrf_bm25_dense_sid_v1",
        dataset_name="dummy", track_split_types=[], corpus_types=[],
        cache_dir=str(tmp_path),
    )

    sid_specs = [s for s in captured_sub_specs["specs"] if s["type"] == "sid_generator"]
    assert sid_specs[0]["weight"] == 0.5


def test_factory_sid_hub_repo_defaults_to_v1_when_override_absent(monkeypatch, tmp_path):
    """When extra_config lacks sid_hub_repo, default to v1 merged."""
    import pandas as pd
    sid_dir = tmp_path / "sid"
    sid_dir.mkdir()
    pd.DataFrame([
        {"track_id": "t1", "code_1": 0, "code_2": 0, "code_3": 0,
         "popularity": 1.0, "bucket_rank": 0},
    ]).to_parquet(sid_dir / "track_to_sid.parquet")

    captured = {}
    class _StubSidGen:
        def __init__(self, hub_repo, sid_lookup_path, **kwargs):
            captured["hub_repo"] = hub_repo

    monkeypatch.setattr(
        "mcrs.retrieval_modules.sid_generator.SID_GENERATOR", _StubSidGen,
    )

    from mcrs.retrieval_modules import load_retrieval_module
    load_retrieval_module(
        retrieval_type="sid_generator",
        dataset_name="dummy", track_split_types=[], corpus_types=[],
        cache_dir=str(tmp_path),
    )

    assert captured["hub_repo"] == "OrRim123/recsys2026-sid-generator-qwen15b-v1-merged"
