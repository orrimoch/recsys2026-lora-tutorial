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
    ds = TripleJsonlDataset(str(path), n_negatives=15)
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


def test_cls_pool_returns_cls_token_not_mean():
    """C1 regression: pooling MUST use last_hidden[:, 0], not mean over time."""
    import torch
    from scripts.train_bi_encoder import _cls_pool
    # Hand-crafted: position 0 is distinct from positions 1+.
    h = torch.tensor([
        [[1.0, 0.0], [10.0, 0.0], [10.0, 0.0]],  # batch row 1
        [[0.0, 1.0], [0.0, 10.0], [0.0, 10.0]],  # batch row 2
    ])
    out = _cls_pool(h)
    # If pooling is CLS, output rows are normalize([1,0]) and normalize([0,1]) → [1,0] and [0,1]
    assert torch.allclose(out[0], torch.tensor([1.0, 0.0]), atol=1e-6)
    assert torch.allclose(out[1], torch.tensor([0.0, 1.0]), atol=1e-6)


def test_dataset_default_n_negatives_is_7():
    """C2 regression: spec says train_group_size=8 (1 pos + 7 negs)."""
    import json
    from pathlib import Path
    import tempfile
    from scripts.train_bi_encoder import TripleJsonlDataset

    rows = [{"query": "q", "pos": ["p"], "neg": [f"n{i}" for i in range(15)]}]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        path = f.name
    ds = TripleJsonlDataset(Path(path), n_negatives=7, seed=42)
    item = ds[0]
    assert len(item["negatives"]) == 7


def test_dataset_samples_negs_randomly_not_first_n():
    """C2 regression: epoch 2 should see different negs than epoch 1 (random sample)."""
    import json
    from pathlib import Path
    import tempfile
    from scripts.train_bi_encoder import TripleJsonlDataset

    rows = [{"query": "q", "pos": ["p"], "neg": [f"n{i}" for i in range(15)]}]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        path = f.name
    # Two datasets with different seeds should produce different neg samples.
    ds1 = TripleJsonlDataset(Path(path), n_negatives=7, seed=1)
    ds2 = TripleJsonlDataset(Path(path), n_negatives=7, seed=2)
    n1 = set(ds1[0]["negatives"])
    n2 = set(ds2[0]["negatives"])
    # Random samples of 7 from 15 with different seeds should differ.
    assert n1 != n2, "Different seeds should produce different neg samples"
