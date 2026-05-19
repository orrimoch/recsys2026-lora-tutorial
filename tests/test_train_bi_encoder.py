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


def test_dataset_default_n_negatives_is_15():
    """ML-reviewer I1: spec §6 line 1042 says n_negatives_per_query=15. The bigger
    per-row contrastive denominator (no cross-row in-batch negs in our InfoNCE)
    materially improves loss quality vs 7."""
    import json
    from pathlib import Path
    import tempfile
    from scripts.train_bi_encoder import TripleJsonlDataset

    rows = [{"query": "q", "pos": ["p"], "neg": [f"n{i}" for i in range(15)]}]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        path = f.name
    # Default n_negatives constructor arg should now be 15 (was 7 pre-ML-review).
    ds = TripleJsonlDataset(Path(path))
    item = ds[0]
    assert len(item["negatives"]) == 15


def test_collate_batch_truncates_queries_from_left():
    """Long queries truncate from the LEFT to preserve [QUERY]: at the end —
    that block carries the current user turn (most informative). Right
    truncation would silently cut [QUERY]: for ~10-20% of long queries."""
    import inspect
    from scripts import train_bi_encoder as mod
    src = inspect.getsource(mod._collate_batch)
    assert 'truncation_side = "left"' in src or "truncation_side = 'left'" in src, \
        "_collate_batch should set tokenizer.truncation_side = 'left' for queries"
    # Doc tokenization should remain right-truncate (default)
    assert 'truncation_side = "right"' in src or "truncation_side = 'right'" in src, \
        "_collate_batch should restore truncation_side = 'right' for docs"


def test_info_nce_loss_in_batch_uses_full_batch_denominator():
    """I-1: in-batch InfoNCE contrasts each query against ALL B*n_per docs.
    With B=2 queries and n_per=3 (1 pos + 2 negs each), the score matrix
    must be shape (2, 6) and labels must point to columns 0 and 3 (positions
    of each query's positive in the flat doc layout)."""
    import torch
    from scripts.train_bi_encoder import _info_nce_loss_in_batch

    # B=2, n_per=3, D=4.
    q_emb = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0, 0.0]])
    # docs: [pos_0, neg_0_0, neg_0_1, pos_1, neg_1_0, neg_1_1]
    # Each query's positive aligned with its query embedding → high score.
    d_emb = torch.tensor([[1.0, 0.0, 0.0, 0.0],  # pos_0 — matches q_0
                          [0.0, 0.0, 1.0, 0.0],  # neg_0_0
                          [0.0, 0.0, 0.0, 1.0],  # neg_0_1
                          [0.0, 1.0, 0.0, 0.0],  # pos_1 — matches q_1
                          [0.5, 0.0, 0.5, 0.0],  # neg_1_0
                          [0.0, 0.5, 0.0, 0.5]]) # neg_1_1
    loss = _info_nce_loss_in_batch(q_emb, d_emb, n_per=3, temperature=0.05)
    # With pos perfectly aligned and negs orthogonal/half-aligned at temp 0.05,
    # the cross-entropy should be near zero — the positive's logit dominates.
    assert float(loss.item()) < 0.5, \
        f"in-batch InfoNCE on perfectly aligned positives is too high: {loss.item():.4f}"


def test_info_nce_loss_in_batch_penalizes_wrong_positive():
    """I-1: when a NEGATIVE (column 1, which is one of query 0's own negs)
    has a HIGHER score than query 0's positive (column 0), the loss must
    be large (>1.0 in log-space). Sanity check that the label index is right."""
    import torch
    from scripts.train_bi_encoder import _info_nce_loss_in_batch

    q_emb = torch.tensor([[1.0, 0.0]])
    # B=1, n_per=2 → 2 docs. Positive is at col 0, but neg has higher dot.
    d_emb = torch.tensor([[0.0, 0.0],   # pos (zero vector → score 0)
                          [10.0, 0.0]]) # neg (high score)
    loss = _info_nce_loss_in_batch(q_emb, d_emb, n_per=2, temperature=1.0)
    assert float(loss.item()) > 5.0, \
        f"in-batch InfoNCE didn't penalize wrong-positive scoring: {loss.item():.4f}"


def test_info_nce_loss_in_batch_treats_other_queries_pos_as_neg():
    """I-1: false-negative behavior — confirm that for query 0, the docs
    associated with query 1 (including query 1's positive at col n_per)
    appear in the denominator. We test this by checking that the score at
    column n_per influences the loss."""
    import torch
    from scripts.train_bi_encoder import _info_nce_loss_in_batch

    # B=2, n_per=2: cols 0,1 belong to q0; cols 2,3 belong to q1.
    # q0's pos is at col 0; q1's pos is at col 2.
    q_emb = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    # q1's positive (col 2) is highly aligned with q0 — should hurt q0's loss.
    d_emb_bad = torch.tensor([[1.0, 0.0],   # q0 pos
                              [0.0, 0.0],   # q0 neg
                              [10.0, 0.0],  # q1 pos — but aligned with q0!
                              [0.0, 1.0]])  # q1 neg
    d_emb_good = torch.tensor([[1.0, 0.0],
                                [0.0, 0.0],
                                [0.0, 10.0],  # q1 pos aligned with q1, not q0
                                [0.0, 1.0]])
    loss_bad = _info_nce_loss_in_batch(q_emb, d_emb_bad, n_per=2, temperature=1.0)
    loss_good = _info_nce_loss_in_batch(q_emb, d_emb_good, n_per=2, temperature=1.0)
    assert float(loss_bad.item()) > float(loss_good.item()), \
        f"in-batch loss isn't sensitive to other-query positives: bad={loss_bad.item():.4f} vs good={loss_good.item():.4f}"


def test_cli_has_gradient_checkpointing_toggle():
    """Speed knob: --gradient-checkpointing / --no-gradient-checkpointing.
    Default ON for back-compat at small per_device_batch_size; at bs=8 on
    Blackwell-95GB we want it OFF for ~30% speedup."""
    import inspect
    from scripts import train_bi_encoder as mod
    src_main = inspect.getsource(mod.main)
    src_train = inspect.getsource(mod._train)
    assert "--gradient-checkpointing" in src_main, "missing --gradient-checkpointing CLI flag"
    assert "--no-gradient-checkpointing" in src_main, "missing --no-gradient-checkpointing CLI flag"
    # _train must gate the call on the flag, not call unconditionally.
    assert "args.gradient_checkpointing" in src_train, \
        "_train does not honor args.gradient_checkpointing"


def test_cli_has_in_batch_negs_and_full_catalog_args():
    """I-1 + I-3: new CLI flags exist."""
    import inspect
    from scripts import train_bi_encoder as mod
    src = inspect.getsource(mod.main)
    assert "--in-batch-negs" in src, "missing --in-batch-negs flag"
    assert "--no-in-batch-negs" in src, "missing --no-in-batch-negs flag"
    assert "--val-full-catalog-every-n-steps" in src, \
        "missing --val-full-catalog-every-n-steps flag"
    assert "--track-meta-hf" in src, "missing --track-meta-hf flag"


def test_dataset_session_disjoint_split_no_session_overlap():
    """CRITICAL: Sub 1 had 95% session-level leak in val split (row-shuffle
    placed multiple turns from same session in both train and val). With
    session_disjoint=True, NO session can appear in both train and val."""
    import json
    import tempfile
    from scripts.train_bi_encoder import TripleJsonlDataset

    # 5 sessions, ~5 rows each = 25 total. Row-shuffle would put rows from
    # the SAME session in both train and val. Session-disjoint must not.
    rows = []
    for sid in range(5):
        for turn in range(5):
            rows.append({
                "query": f"session_{sid}_turn_{turn}_query",
                "pos": [f"track_{sid}_{turn}"],
                "neg": [f"n{i}" for i in range(15)],
                "pos_tid": f"track_{sid}_{turn}",
                "session_id": f"sess_{sid}",
            })
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        path = f.name

    train = TripleJsonlDataset(path, split="train", val_fraction=0.4, seed=42,
                                session_disjoint=True)
    val = TripleJsonlDataset(path, split="val", val_fraction=0.4, seed=42,
                              session_disjoint=True)

    train_sids = {r["session_id"] for r in train.rows}
    val_sids = {r["session_id"] for r in val.rows}
    assert train_sids.isdisjoint(val_sids), \
        f"session leak detected: {train_sids & val_sids} appears in both train and val"
    # All rows from each session land entirely on one side.
    assert len(train) + len(val) == 25, \
        f"expected 25 total rows across splits, got {len(train) + len(val)}"


def test_dataset_session_disjoint_falls_back_to_row_shuffle_without_session_id():
    """Back-compat: older triples files (no session_id) silently fall back to
    row-level shuffle so existing callers keep working."""
    import json
    import tempfile
    from scripts.train_bi_encoder import TripleJsonlDataset

    rows = [{"query": f"q{i}", "pos": [f"p{i}"], "neg": [f"n{j}" for j in range(15)]}
            for i in range(20)]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        path = f.name
    # session_disjoint=True but no session_id field → falls back to row shuffle.
    train = TripleJsonlDataset(path, split="train", val_fraction=0.2, seed=42,
                                session_disjoint=True)
    val = TripleJsonlDataset(path, split="val", val_fraction=0.2, seed=42,
                              session_disjoint=True)
    assert len(train) + len(val) == 20


def test_dataset_carries_pos_tid_when_present():
    """I-3: TripleJsonlDataset surfaces pos_tid via .pos_tids() and __getitem__
    when the on-disk triples carry it. Older triples without pos_tid: pos_tids()
    returns [] and full-catalog val is silently skipped (back-compat)."""
    import json
    import tempfile
    from scripts.train_bi_encoder import TripleJsonlDataset

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for i in range(3):
            row = {
                "query": f"q{i}", "pos": [f"p{i}"],
                "neg": [f"n{j}" for j in range(15)],
                "pos_tid": f"track_{i}",
            }
            f.write(json.dumps(row) + "\n")
        with_tid_path = f.name
    ds = TripleJsonlDataset(with_tid_path)
    assert ds.pos_tids() == ["track_0", "track_1", "track_2"]
    assert ds[0]["pos_tid"] == "track_0"

    # Back-compat: older triples file with no pos_tid.
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for i in range(3):
            row = {"query": f"q{i}", "pos": [f"p{i}"],
                   "neg": [f"n{j}" for j in range(15)]}
            f.write(json.dumps(row) + "\n")
        no_tid_path = f.name
    ds_old = TripleJsonlDataset(no_tid_path)
    assert ds_old.pos_tids() == []
    assert "pos_tid" not in ds_old[0]


def test_val_metrics_perfect_ranking():
    """If positive (col 0) has the highest score for every row → top1=1, ndcg=1."""
    import torch
    from scripts.train_bi_encoder import _val_metrics_from_scores

    # Each row: positive at index 0 has score 10, others have 0.
    scores = torch.zeros(4, 16)
    scores[:, 0] = 10.0
    top1, ndcg = _val_metrics_from_scores(scores)
    assert abs(top1 - 1.0) < 1e-6
    assert abs(ndcg - 1.0) < 1e-6


def test_val_metrics_positive_at_rank_2_drops_ndcg():
    """Positive ranked 2 → ndcg = 1/log2(3) ≈ 0.6309."""
    import math
    import torch
    from scripts.train_bi_encoder import _val_metrics_from_scores

    scores = torch.zeros(1, 16)
    scores[0, 0] = 5.0   # positive
    scores[0, 1] = 10.0  # one negative beats it → positive ranks 2
    top1, ndcg = _val_metrics_from_scores(scores)
    assert top1 == 0.0
    assert abs(ndcg - (1.0 / math.log2(3))) < 1e-6


def test_val_metrics_positive_at_worst_rank():
    """Positive ranked last (rank 16) → ndcg = 1/log2(17)."""
    import math
    import torch
    from scripts.train_bi_encoder import _val_metrics_from_scores

    scores = torch.zeros(1, 16)
    scores[0, 0] = -100.0   # positive lowest
    scores[0, 1:] = torch.arange(1, 16).float()  # all negs higher
    top1, ndcg = _val_metrics_from_scores(scores)
    assert top1 == 0.0
    assert abs(ndcg - (1.0 / math.log2(17))) < 1e-6


def test_val_metrics_mean_over_batch():
    """Half rows perfect, half worst → top1 = 0.5, ndcg = mean of (1.0, 1/log2(17))."""
    import math
    import torch
    from scripts.train_bi_encoder import _val_metrics_from_scores

    scores = torch.zeros(4, 16)
    scores[0:2, 0] = 10.0  # rows 0,1: positive has highest score
    scores[2:4, 0] = -100.0
    scores[2:4, 1:] = torch.arange(1, 16).float().unsqueeze(0).expand(2, -1)
    top1, ndcg = _val_metrics_from_scores(scores)
    assert abs(top1 - 0.5) < 1e-6
    expected_ndcg = (1.0 + 1.0 + 1.0/math.log2(17) + 1.0/math.log2(17)) / 4
    assert abs(ndcg - expected_ndcg) < 1e-6


def test_cli_has_checkpoint_and_resume_args():
    """Checkpointing + warm-start CLI flags exist with documented defaults."""
    import inspect
    from scripts import train_bi_encoder as mod

    src = inspect.getsource(mod.main)
    assert "--checkpoint-every-n-epochs" in src, \
        "missing --checkpoint-every-n-epochs CLI arg"
    assert "--resume-from" in src, "missing --resume-from CLI arg"
    assert "--val-every-n-steps" in src, "missing --val-every-n-steps CLI arg"
    assert "--val-fraction" in src, "missing --val-fraction CLI arg"


def test_train_loop_uses_resume_from_when_set():
    """The _train function branches on args.resume_from to load an existing
    adapter instead of creating a fresh LoRA. Source-level smoke check so we
    don't accidentally regress the warm-start path."""
    import inspect
    from scripts import train_bi_encoder as mod

    src = inspect.getsource(mod._train)
    assert "resume_from" in src, "_train does not reference resume_from"
    assert "PeftModel.from_pretrained" in src, \
        "_train does not call PeftModel.from_pretrained for warm-start"
    assert "checkpoint_epoch_" in src, \
        "_train does not emit per-epoch checkpoints"


def test_dataset_train_val_split_sizes_match_fraction():
    """ML-reviewer N1: split='train'/'val' partitions rows by fraction.
    Train + val sizes should equal total; val ~= round(N * val_fraction)."""
    import json
    import tempfile
    from scripts.train_bi_encoder import TripleJsonlDataset

    rows = [{"query": f"q{i}", "pos": [f"p{i}"], "neg": [f"n{i}_{j}" for j in range(15)]}
            for i in range(100)]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        path = f.name
    train = TripleJsonlDataset(path, split="train", val_fraction=0.1)
    val = TripleJsonlDataset(path, split="val", val_fraction=0.1)
    assert len(train) + len(val) == 100
    assert len(val) == 10  # round(100 * 0.1)


def test_dataset_train_val_split_is_disjoint():
    """No query should appear in both train and val with the same seed."""
    import json
    import tempfile
    from scripts.train_bi_encoder import TripleJsonlDataset

    rows = [{"query": f"q{i}", "pos": [f"p{i}"], "neg": [f"n{i}_{j}" for j in range(15)]}
            for i in range(50)]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        path = f.name
    train = TripleJsonlDataset(path, split="train", val_fraction=0.2, seed=42)
    val = TripleJsonlDataset(path, split="val", val_fraction=0.2, seed=42)
    train_qs = {train[i]["query"] for i in range(len(train))}
    val_qs = {val[i]["query"] for i in range(len(val))}
    assert train_qs.isdisjoint(val_qs)
    assert len(train_qs | val_qs) == 50


def test_dataset_default_split_loads_everything():
    """Back-compat: omitting split + val_fraction loads the full file."""
    import json
    import tempfile
    from scripts.train_bi_encoder import TripleJsonlDataset

    rows = [{"query": f"q{i}", "pos": [f"p{i}"], "neg": [f"n{j}" for j in range(15)]}
            for i in range(20)]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        path = f.name
    ds = TripleJsonlDataset(path)
    assert len(ds) == 20


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
