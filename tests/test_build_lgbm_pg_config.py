"""build_lgbm_features must be able to put propose-ground in the reranker's
training pool (--use-propose-ground), mirroring --use-two-tower. Without this the
reranker can never train on the pg channel it serves on -> the same train/serve
pool mismatch that sank config 198x. Tests the pure config assembler so no heavy
retrieval deps are exercised.
"""
import importlib.util
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    spec = importlib.util.spec_from_file_location(
        "build_lgbm_features", os.path.join(REPO, "scripts/build_lgbm_features.py"))
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:  # heavy import deps may be missing locally
        pytest.skip(f"could not import build_lgbm_features: {e!r}")
    return mod


def test_pg_absent_from_config_by_default():
    extra = _load()._union_extra_config()
    assert "use_propose_ground" not in extra


def test_pg_wired_into_union_config_when_enabled():
    extra = _load()._union_extra_config(
        use_propose_ground=True, w_propose_ground=0.5, pg_model="gemini-2.5-flash")
    assert extra["use_propose_ground"] is True
    assert extra["w_propose_ground"] == 0.5
    assert extra["pg_model"] == "gemini-2.5-flash"
    # union spec keys must match mcrs.retrieval_modules.__init__ (pg_* prefix)
    assert extra["pg_inner_dense"] == "dense_metadata_qwen3_instruct"
    assert extra["pg_n_proposals"] == 20
    assert extra["pg_batch_size"] == 16


def test_pg_coexists_with_sasrec_and_two_tower():
    extra = _load()._union_extra_config(
        use_sasrec=True, use_two_tower=True, use_propose_ground=True)
    assert extra["use_sasrec"] is True
    assert extra["use_two_tower"] is True
    assert extra["use_propose_ground"] is True
