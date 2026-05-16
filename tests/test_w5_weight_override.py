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
