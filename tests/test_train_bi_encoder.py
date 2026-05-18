"""Tests for the custom PEFT-LoRA training loop in scripts/train_bi_encoder.py."""
import json

import pytest


def test_triple_jsonl_dataset_yields_query_pos_neg(tmp_path):
    """TripleJsonlDataset returns dicts with query, positive, negatives keys."""
    from scripts.train_bi_encoder import TripleJsonlDataset

    path = tmp_path / "triples.jsonl"
    with open(path, "w") as f:
        for i in range(3):
            f.write(json.dumps({
                "query": f"q{i}", "pos": [f"p{i}"],
                "neg": [f"n{i}_{j}" for j in range(15)],
            }) + "\n")
    ds = TripleJsonlDataset(str(path))
    assert len(ds) == 3
    row = ds[0]
    assert row["query"] == "q0"
    assert row["positive"] == "p0"
    assert len(row["negatives"]) == 15


def test_triple_jsonl_dataset_pads_short_neg_list(tmp_path):
    """When a row has fewer than `n_negatives` negs, it's repeated (don't drop the row)."""
    from scripts.train_bi_encoder import TripleJsonlDataset

    path = tmp_path / "triples.jsonl"
    with open(path, "w") as f:
        f.write(json.dumps({"query": "q", "pos": ["p"], "neg": ["n1", "n2"]}) + "\n")
    ds = TripleJsonlDataset(str(path), n_negatives=15)
    row = ds[0]
    assert len(row["negatives"]) == 15
    # First two are the actual negs; rest are random samples from the same pool.
    assert "n1" in row["negatives"]
    assert "n2" in row["negatives"]


def test_build_lora_targets_returns_attention_module_names():
    """target_modules covers BGE-M3 XLMRoberta attention + FFN projections."""
    from scripts.train_bi_encoder import _BGE_M3_LORA_TARGETS
    # BGE-M3 is XLMRoberta-based; attention layers are .query/.key/.value/.dense
    assert "query" in _BGE_M3_LORA_TARGETS
    assert "key" in _BGE_M3_LORA_TARGETS
    assert "value" in _BGE_M3_LORA_TARGETS
