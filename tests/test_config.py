"""F2 — config loader tests (defaults, validation, dotted access).

Per .claude/documents/features/11_F2_interfaces_contracts_config.md §7, §9.
"""
from __future__ import annotations

import pytest

from mcrs.config import load_config


def test_defaults_applied_when_key_absent():
    cfg = load_config({})
    assert cfg.seed == 42
    assert cfg.segment.cold_threshold == 1
    assert cfg.retrieval.topk == 300


def test_provided_value_overrides_default():
    cfg = load_config({"seed": 7, "retrieval": {"topk": 500}})
    assert cfg.seed == 7
    assert cfg.retrieval.topk == 500
    assert cfg.segment.cold_threshold == 1  # untouched default


def test_colbert_expansion_first_default_and_override():
    # C1: serve-side ColBERT doc recipe must match training (expansion_first=True by default).
    assert load_config({}).retrieval.colbert.expansion_first is True
    assert load_config(
        {"retrieval": {"colbert": {"expansion_first": False}}}
    ).retrieval.colbert.expansion_first is False


def test_unknown_top_level_key_rejected():
    with pytest.raises(ValueError):
        load_config({"nonsense": 1})


def test_unknown_nested_key_rejected():
    with pytest.raises(ValueError):
        load_config({"retrieval": {"bogus": 1}})


def test_type_mismatch_rejected():
    with pytest.raises((ValueError, TypeError)):
        load_config({"seed": "not-an-int"})


def test_loads_from_yaml_file(tmp_path):
    p = tmp_path / "exp.yaml"
    p.write_text("seed: 5\nretrieval:\n  topk: 400\n")
    cfg = load_config(str(p))
    assert cfg.seed == 5 and cfg.retrieval.topk == 400
