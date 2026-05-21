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


def test_rows_below_n_negatives_are_dropped(tmp_path):
    """Issue E (no upsampling with replacement): rows whose mined neg-list is
    shorter than `n_negatives` are DROPPED at load time, not padded with
    repeated negs. Repeating negs duplicated their gradient signal."""
    from scripts.train_bi_encoder import TripleJsonlDataset

    path = tmp_path / "triples.jsonl"
    with open(path, "w") as f:
        f.write(json.dumps({"query": "q_short", "pos": ["p"], "neg": ["n1", "n2"]}) + "\n")
        f.write(json.dumps({"query": "q_full", "pos": ["p"],
                            "neg": [f"n{i}" for i in range(15)]}) + "\n")
    ds = TripleJsonlDataset(str(path), n_negatives=15)
    # Short row (2 negs < 15) must be dropped; only the full one remains.
    assert len(ds) == 1, f"expected 1 row to survive, got {len(ds)}"
    assert ds[0]["query"] == "q_full"


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


def test_collate_uses_two_tokenizers_no_runtime_mutation():
    """Issue G: _collate_batch must take TWO tokenizer instances (one
    left-truncate for queries, one right-truncate for docs) and MUST NOT
    mutate `truncation_side` at call time. Removes fragility under
    `persistent_workers=True` and any future multi-thread DataLoader path."""
    import inspect
    from scripts import train_bi_encoder as mod
    sig = inspect.signature(mod._collate_batch)
    params = list(sig.parameters.keys())
    # Expect signature like (batch, q_tokenizer, d_tokenizer, max_q_len, max_p_len)
    assert "q_tokenizer" in params and "d_tokenizer" in params, \
        f"_collate_batch must accept q_tokenizer + d_tokenizer; got {params}"
    src = inspect.getsource(mod._collate_batch)
    # No runtime mutation of truncation_side.
    assert "tokenizer.truncation_side =" not in src, \
        "_collate_batch must not mutate tokenizer.truncation_side at call time"
    assert "_original_side" not in src, \
        "_collate_batch should not need to snapshot/restore truncation_side"


def test_collate_passes_left_and_right_tokenizers_through():
    """Functional behavior: _collate_batch calls q_tokenizer for queries
    (which is configured left-truncate) and d_tokenizer for docs
    (right-truncate). We verify by stub tokenizers that record which tokenizer
    received which texts."""
    from scripts.train_bi_encoder import _collate_batch
    import torch

    class StubTok:
        def __init__(self, side):
            self.truncation_side = side
            self.received = []
        def __call__(self, texts, **kwargs):
            self.received.append(list(texts))
            n = len(texts)
            return {"input_ids": torch.zeros(n, 4, dtype=torch.long),
                    "attention_mask": torch.ones(n, 4, dtype=torch.long)}

    q_tok = StubTok("left")
    d_tok = StubTok("right")
    batch = [{"query": "the question", "positive": "track_A",
              "negatives": ["track_B", "track_C", "track_D"]}]
    _collate_batch(batch, q_tok, d_tok, max_q_len=32, max_p_len=32)
    # q_tokenizer should have been called with exactly the query.
    assert q_tok.received == [["the question"]]
    # d_tokenizer should have been called with positive + negatives in order.
    assert d_tok.received == [["track_A", "track_B", "track_C", "track_D"]]
    # Neither side mutated post-construction.
    assert q_tok.truncation_side == "left"
    assert d_tok.truncation_side == "right"


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


def test_user_disjoint_split_no_user_in_both_folds(tmp_path):
    """Δ2 CRITICAL contract: every session of a given user_id lives in exactly
    one partition (train OR val). No user_id may appear in both folds."""
    import json
    from scripts.train_bi_encoder import TripleJsonlDataset

    # 5 users × 3 sessions × 5 rows = 75 rows. With user-disjoint, no user
    # can straddle the train/val boundary.
    rows = []
    for u in range(5):
        for s in range(3):
            for t in range(5):
                rows.append({
                    "query": f"u{u}_s{s}_t{t}",
                    "pos": [f"p"], "neg": [f"n{i}" for i in range(15)],
                    "pos_tid": f"track_{u}_{s}_{t}",
                    "user_id": f"user_{u}",
                    "session_id": f"sess_{u}_{s}",
                })
    path = tmp_path / "triples.jsonl"
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    train = TripleJsonlDataset(str(path), split="train", val_fraction=0.4,
                                seed=42, split_key="user_id")
    val = TripleJsonlDataset(str(path), split="val", val_fraction=0.4,
                              seed=42, split_key="user_id")

    train_users = {r["user_id"] for r in train.rows}
    val_users = {r["user_id"] for r in val.rows}
    assert train_users.isdisjoint(val_users), \
        f"user_id leak detected: {train_users & val_users}"
    # All 75 rows accounted for; user's sessions all on one side.
    assert len(train) + len(val) == 75


def test_user_disjoint_split_also_implies_session_disjoint(tmp_path):
    """Δ2: user-disjoint is strictly stronger than session-disjoint. If train
    and val are user-disjoint, no session_id can appear in both either."""
    import json
    from scripts.train_bi_encoder import TripleJsonlDataset

    rows = []
    for u in range(4):
        for s in range(3):
            for t in range(4):
                rows.append({
                    "query": f"u{u}_s{s}_t{t}",
                    "pos": ["p"], "neg": [f"n{i}" for i in range(15)],
                    "user_id": f"user_{u}",
                    "session_id": f"sess_{u}_{s}",
                })
    path = tmp_path / "triples.jsonl"
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    train = TripleJsonlDataset(str(path), split="train", val_fraction=0.25,
                                seed=42, split_key="user_id")
    val = TripleJsonlDataset(str(path), split="val", val_fraction=0.25,
                              seed=42, split_key="user_id")
    train_sids = {r["session_id"] for r in train.rows}
    val_sids = {r["session_id"] for r in val.rows}
    assert train_sids.isdisjoint(val_sids)


def test_user_disjoint_split_fails_loud_when_user_id_missing(tmp_path):
    """Δ2 issue A fix: when split_key='user_id' but any row is missing
    user_id, FAIL LOUDLY. No silent fallback to row-shuffle or session_id —
    that's how the original Sub-1 leak happened."""
    import json
    import pytest
    from scripts.train_bi_encoder import TripleJsonlDataset

    rows = [
        {"query": "q1", "pos": ["p"], "neg": [f"n{i}" for i in range(15)],
         "user_id": "u1", "session_id": "s1"},
        # Row 2 is missing user_id — should trigger loud failure.
        {"query": "q2", "pos": ["p"], "neg": [f"n{i}" for i in range(15)],
         "session_id": "s2"},
    ]
    path = tmp_path / "triples.jsonl"
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    with pytest.raises((ValueError, RuntimeError)) as excinfo:
        TripleJsonlDataset(str(path), split="train", val_fraction=0.1,
                           seed=42, split_key="user_id")
    assert "user_id" in str(excinfo.value).lower()


def test_split_key_inspects_all_rows_not_just_first(tmp_path):
    """Issue A regression: previously `has_session` checked only all_rows[0].
    A row-0 with empty session_id would silently demote to row-shuffle even
    if every other row had session_id. Fix: ALL rows must carry the field."""
    import json
    import pytest
    from scripts.train_bi_encoder import TripleJsonlDataset

    # Row 0 has user_id="u1"; row 1 has user_id=None (the "first row OK,
    # others broken" failure mode).
    rows = [
        {"query": "q1", "pos": ["p"], "neg": [f"n{i}" for i in range(15)],
         "user_id": "u1", "session_id": "s1"},
        {"query": "q2", "pos": ["p"], "neg": [f"n{i}" for i in range(15)],
         "user_id": None, "session_id": "s2"},
    ]
    path = tmp_path / "triples.jsonl"
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    with pytest.raises((ValueError, RuntimeError)):
        TripleJsonlDataset(str(path), split="train", val_fraction=0.1,
                           seed=42, split_key="user_id")


def test_split_key_session_id_legacy_path_still_works(tmp_path):
    """Back-compat: callers can opt to split by session_id explicitly when
    triples lack user_id (legacy mined files)."""
    import json
    from scripts.train_bi_encoder import TripleJsonlDataset

    rows = []
    for s in range(5):
        for t in range(5):
            rows.append({
                "query": f"s{s}_t{t}", "pos": ["p"],
                "neg": [f"n{i}" for i in range(15)],
                "session_id": f"sess_{s}",
            })
    path = tmp_path / "triples.jsonl"
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    train = TripleJsonlDataset(str(path), split="train", val_fraction=0.4,
                                seed=42, split_key="session_id")
    val = TripleJsonlDataset(str(path), split="val", val_fraction=0.4,
                              seed=42, split_key="session_id")
    train_sids = {r["session_id"] for r in train.rows}
    val_sids = {r["session_id"] for r in val.rows}
    assert train_sids.isdisjoint(val_sids)


def test_split_key_row_is_explicit_opt_in(tmp_path):
    """Back-compat: callers with no session_id/user_id MUST explicitly pass
    split_key='row' — no implicit fallback (per issue A)."""
    import json
    from scripts.train_bi_encoder import TripleJsonlDataset

    rows = [{"query": f"q{i}", "pos": [f"p{i}"],
             "neg": [f"n{j}" for j in range(15)]} for i in range(20)]
    path = tmp_path / "triples.jsonl"
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    train = TripleJsonlDataset(str(path), split="train", val_fraction=0.2,
                                seed=42, split_key="row")
    val = TripleJsonlDataset(str(path), split="val", val_fraction=0.2,
                              seed=42, split_key="row")
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
    Train + val sizes should equal total; val ~= round(N * val_fraction).
    Uses split_key='row' explicitly because these rows lack user_id."""
    import json
    import tempfile
    from scripts.train_bi_encoder import TripleJsonlDataset

    rows = [{"query": f"q{i}", "pos": [f"p{i}"], "neg": [f"n{i}_{j}" for j in range(15)]}
            for i in range(100)]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        path = f.name
    train = TripleJsonlDataset(path, split="train", val_fraction=0.1, split_key="row")
    val = TripleJsonlDataset(path, split="val", val_fraction=0.1, split_key="row")
    assert len(train) + len(val) == 100
    assert len(val) == 10  # round(100 * 0.1)


def test_dataset_train_val_split_is_disjoint():
    """No query should appear in both train and val with the same seed.
    Uses split_key='row' explicitly because these rows lack user_id."""
    import json
    import tempfile
    from scripts.train_bi_encoder import TripleJsonlDataset

    rows = [{"query": f"q{i}", "pos": [f"p{i}"], "neg": [f"n{i}_{j}" for j in range(15)]}
            for i in range(50)]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        path = f.name
    train = TripleJsonlDataset(path, split="train", val_fraction=0.2, seed=42, split_key="row")
    val = TripleJsonlDataset(path, split="val", val_fraction=0.2, seed=42, split_key="row")
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


# ---------------------------------------------------------------------------
# §6.5 amendment: UserDisjointBatchSampler + in-batch false-positive mask +
# CLI plumbing + default-fraction bump (issue D).
# ---------------------------------------------------------------------------


def test_user_disjoint_batch_sampler_unique_users_per_batch():
    """Δ3 issue B fix: every batch the sampler emits contains rows whose
    user_ids are pairwise distinct. Same-user collisions cause false negatives
    in in-batch InfoNCE."""
    from scripts.train_bi_encoder import UserDisjointBatchSampler

    # 10 users × 3 rows = 30 rows. batch_size=5 → trivially achievable.
    row_user_ids = [f"u{i // 3}" for i in range(30)]
    sampler = UserDisjointBatchSampler(row_user_ids, batch_size=5, seed=42)
    seen_any_batch = False
    for batch in sampler:
        seen_any_batch = True
        users = [row_user_ids[i] for i in batch]
        assert len(set(users)) == len(users), \
            f"duplicate users in batch: {users}"
    assert seen_any_batch, "sampler emitted no batches"


def test_user_disjoint_batch_sampler_each_index_at_most_once_per_epoch():
    """Δ3 without-replacement guarantee: across one full iteration of the
    sampler (= one epoch), every row index is yielded AT MOST ONCE."""
    from scripts.train_bi_encoder import UserDisjointBatchSampler

    row_user_ids = [f"u{i // 3}" for i in range(30)]
    sampler = UserDisjointBatchSampler(row_user_ids, batch_size=5, seed=42)
    all_indices = []
    for batch in sampler:
        all_indices.extend(batch)
    assert len(all_indices) == len(set(all_indices)), \
        f"index emitted twice in same epoch: " \
        f"{[i for i in all_indices if all_indices.count(i) > 1][:5]}"
    # And every emitted index is in valid range.
    assert all(0 <= i < 30 for i in all_indices)


def test_user_disjoint_batch_sampler_yields_full_batches_when_possible():
    """Δ3: the sampler should produce batches of `batch_size` whenever the
    remaining row pool supports it. The final batch may be ragged but no
    interior batch should be short."""
    from scripts.train_bi_encoder import UserDisjointBatchSampler

    # 10 users × 4 rows = 40 rows. batch_size=8 → 5 batches of 8 fit perfectly.
    row_user_ids = [f"u{i // 4}" for i in range(40)]
    sampler = UserDisjointBatchSampler(row_user_ids, batch_size=8, seed=42)
    batches = list(sampler)
    # All batches except possibly the last must have batch_size rows.
    for batch in batches[:-1]:
        assert len(batch) == 8, f"interior batch is short: {len(batch)}"


def test_user_disjoint_batch_sampler_deterministic_by_seed():
    """Δ3: same seed + same row_user_ids → same emission order."""
    from scripts.train_bi_encoder import UserDisjointBatchSampler

    row_user_ids = [f"u{i // 3}" for i in range(30)]
    s1 = UserDisjointBatchSampler(row_user_ids, batch_size=5, seed=7)
    s2 = UserDisjointBatchSampler(row_user_ids, batch_size=5, seed=7)
    assert list(s1) == list(s2)
    s3 = UserDisjointBatchSampler(row_user_ids, batch_size=5, seed=99)
    assert list(s1) != list(s3)


def test_user_disjoint_batch_sampler_fixed_seed_yields_same_batches_across_iters():
    """When `fixed_seed=True`, the sampler does NOT advance its epoch counter
    on iteration, so two consecutive `list(sampler)` calls produce identical
    batches. This is the contract the val_loader relies on so val_loss is
    apples-to-apples across opt-steps."""
    from scripts.train_bi_encoder import UserDisjointBatchSampler

    row_user_ids = [f"u{i // 3}" for i in range(30)]
    sampler = UserDisjointBatchSampler(row_user_ids, batch_size=5, seed=42,
                                        fixed_seed=True)
    first = list(sampler)
    second = list(sampler)
    assert first == second, \
        "fixed_seed=True must produce identical batches across iterations"


def test_user_disjoint_batch_sampler_advances_when_not_fixed():
    """Default behavior (fixed_seed=False): epoch counter advances on each
    iteration so train batches change across epochs."""
    from scripts.train_bi_encoder import UserDisjointBatchSampler

    row_user_ids = [f"u{i // 3}" for i in range(30)]
    sampler = UserDisjointBatchSampler(row_user_ids, batch_size=5, seed=42)
    first = list(sampler)
    second = list(sampler)
    assert first != second, \
        "default sampler must advance epoch on each iteration"


def test_train_loop_constructs_user_disjoint_val_sampler():
    """When split_key='user_id' AND val has user_ids, the val_loader must use
    UserDisjointBatchSampler with fixed_seed=True so val_loss is comparable
    to train_loss on the TB curve."""
    import inspect
    from scripts import train_bi_encoder as mod
    src = inspect.getsource(mod._train)
    # Must construct a val_sampler from val_ds.user_ids().
    assert "val_sampler" in src, \
        "_train must construct a val_sampler"
    assert "UserDisjointBatchSampler" in src, \
        "_train must reference UserDisjointBatchSampler"
    assert "fixed_seed=True" in src, \
        "val_sampler must be constructed with fixed_seed=True for reproducibility"


def test_user_disjoint_batch_sampler_len_matches_emitted():
    """Δ3: `len(sampler)` is required by PyTorch DataLoader for progress
    reporting. It must equal the number of batches actually yielded."""
    from scripts.train_bi_encoder import UserDisjointBatchSampler

    row_user_ids = [f"u{i // 3}" for i in range(30)]
    sampler = UserDisjointBatchSampler(row_user_ids, batch_size=5, seed=42)
    emitted = list(sampler)
    assert len(sampler) == len(emitted)


def test_in_batch_loss_masks_duplicate_pos_tid_collisions():
    """Issue C: when two queries in the batch share the same pos_tid (popular
    track), the OTHER query's positive column must be masked off this query's
    denominator. Otherwise the loss says 'push apart' on a track this query
    actually likes — pure label noise."""
    import torch
    from scripts.train_bi_encoder import _info_nce_loss_in_batch_masked

    # B=2, n_per=2. Both queries' positives are the SAME track.
    q_emb = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    # docs: [q0_pos, q0_neg, q1_pos, q1_neg]. q0_pos and q1_pos are identical.
    d_emb = torch.tensor([[1.0, 0.0],   # q0_pos
                          [0.0, 0.5],   # q0_neg
                          [1.0, 0.0],   # q1_pos — same track as q0_pos!
                          [0.0, 0.5]])  # q1_neg
    pos_tids = ["track_X", "track_X"]   # collision

    # Unmasked baseline: q1_pos at col 2 has identical score as q0_pos at col 0,
    # which inflates q0's denominator → loss far from zero.
    from scripts.train_bi_encoder import _info_nce_loss_in_batch
    loss_unmasked = _info_nce_loss_in_batch(q_emb, d_emb, n_per=2, temperature=1.0)
    loss_masked = _info_nce_loss_in_batch_masked(
        q_emb, d_emb, n_per=2, temperature=1.0, pos_tids=pos_tids,
    )
    # Masking the collision column should strictly reduce the loss.
    assert float(loss_masked.item()) < float(loss_unmasked.item()), \
        f"mask had no effect: masked={loss_masked.item():.4f} vs " \
        f"unmasked={loss_unmasked.item():.4f}"


def test_in_batch_loss_mask_is_noop_when_all_pos_tids_distinct():
    """Issue C: when no two queries share a pos_tid AND no cross-pos-into-neg
    collisions, the masked loss is numerically equal to the unmasked in-batch
    loss."""
    import torch
    from scripts.train_bi_encoder import (
        _info_nce_loss_in_batch, _info_nce_loss_in_batch_masked,
    )
    torch.manual_seed(0)
    q_emb = torch.randn(3, 8)
    q_emb = q_emb / q_emb.norm(dim=-1, keepdim=True)
    d_emb = torch.randn(3 * 4, 8)
    d_emb = d_emb / d_emb.norm(dim=-1, keepdim=True)
    pos_tids = ["A", "B", "C"]
    neg_tids_per_row = [["nA1", "nA2", "nA3"],
                       ["nB1", "nB2", "nB3"],
                       ["nC1", "nC2", "nC3"]]
    unmasked = float(_info_nce_loss_in_batch(q_emb, d_emb, n_per=4, temperature=0.1).item())
    masked = float(_info_nce_loss_in_batch_masked(
        q_emb, d_emb, n_per=4, temperature=0.1, pos_tids=pos_tids,
        neg_tids_per_row=neg_tids_per_row,
    ).item())
    assert abs(masked - unmasked) < 1e-5, \
        f"mask perturbed loss when no collisions exist: {masked} vs {unmasked}"


def test_in_batch_loss_masks_cross_positive_in_negative_slots():
    """Issue C (RocketQAv2 / BGE-M3 §3.3 extension): when query i's gold
    appears in query j's mined neg list (j ≠ i), that negative *slot* in
    the score matrix must be masked off query i's denominator. Otherwise
    we're penalizing query i for retrieving its own gold from the wrong
    row's slot."""
    import torch
    from scripts.train_bi_encoder import (
        _info_nce_loss_in_batch, _info_nce_loss_in_batch_masked,
    )

    # B=2, n_per=3 (1 pos + 2 negs each).
    # Doc layout: [q0_pos, q0_neg_0, q0_neg_1, q1_pos, q1_neg_0, q1_neg_1]
    # Set up so q0's gold = "T0" appears as q1's neg_0 (slot col 4).
    q_emb = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    d_emb = torch.tensor([[1.0, 0.0],   # q0_pos (= T0)
                          [0.0, 0.5],   # q0_neg_0 = some-other
                          [0.0, 0.3],   # q0_neg_1 = some-other
                          [0.0, 1.0],   # q1_pos (= T1)
                          [1.0, 0.0],   # q1_neg_0 — SAME embedding as q0_pos!
                          [0.0, 0.3]])  # q1_neg_1
    pos_tids = ["T0", "T1"]
    neg_tids = [["other1", "other2"], ["T0", "other3"]]  # q1's neg_0 = T0

    unmasked = _info_nce_loss_in_batch(q_emb, d_emb, n_per=3, temperature=1.0)
    masked = _info_nce_loss_in_batch_masked(
        q_emb, d_emb, n_per=3, temperature=1.0,
        pos_tids=pos_tids, neg_tids_per_row=neg_tids,
    )
    # Without the cross-pos-into-neg mask, query 0 sees col 4 with high score
    # (it's q0's own gold!) in the denominator, hurting q0's loss.
    # With the mask, col 4 → -inf for row 0, denominator shrinks, loss drops.
    assert float(masked.item()) < float(unmasked.item()), \
        f"cross-pos-into-neg mask had no effect: " \
        f"masked={masked.item():.4f} vs unmasked={unmasked.item():.4f}"


def test_in_batch_loss_mask_never_touches_own_positive_or_own_negs():
    """Mask invariant: cells (i, c) where c is one of row i's OWN slots
    (positive at i*n_per OR negatives at i*n_per+1..i*n_per+K) must NEVER
    be masked, even if a tid collision exists."""
    import torch
    from scripts.train_bi_encoder import _info_nce_loss_in_batch_masked

    # Worst case: q0 lists its own gold as a neg of itself (shouldn't happen
    # in practice, but the mask must not touch own-row slots regardless).
    q_emb = torch.tensor([[1.0, 0.0]])
    d_emb = torch.tensor([[1.0, 0.0],   # q0_pos = T0
                          [0.0, 1.0],   # q0_neg_0
                          [0.5, 0.5]])  # q0_neg_1
    pos_tids = ["T0"]
    neg_tids = [["T0", "other"]]  # row 0's own neg_0 == own gold (pathological)
    # Loss should be finite (no -inf in the label column or in the row's own negs).
    loss = _info_nce_loss_in_batch_masked(
        q_emb, d_emb, n_per=3, temperature=1.0,
        pos_tids=pos_tids, neg_tids_per_row=neg_tids,
    )
    assert torch.isfinite(loss).item(), f"loss is not finite: {loss.item()}"


def test_cli_has_split_key_arg():
    """Δ2: CLI exposes --split-key {user_id,session_id,row}. Default user_id."""
    import inspect
    from scripts import train_bi_encoder as mod
    src = inspect.getsource(mod.main)
    assert "--split-key" in src, "missing --split-key CLI arg"


def test_cli_default_val_fraction_is_one_tenth():
    """Issue D: the default --val-fraction should be 0.10 (was 0.05 — too few
    val users at typical mine size to give a stable metric)."""
    import inspect
    from scripts import train_bi_encoder as mod
    src = inspect.getsource(mod.main)
    # Look for the val-fraction argparse default. Pattern: default=0.10 / 0.1
    assert "--val-fraction" in src, "missing --val-fraction"
    # We look for either "0.1" or "0.10" in the same surrounding 200-char window
    # as the --val-fraction flag.
    idx = src.find("--val-fraction")
    window = src[idx:idx + 400]
    assert "default=0.1" in window, \
        f"--val-fraction default should be 0.1, not 0.05; saw: {window[:200]!r}"


def test_dataset_carries_neg_tids_when_present(tmp_path):
    """Issue C extension: when triples carry per-neg track_ids, the dataset
    surfaces `neg_tids` on the row, kept index-aligned with `negatives`
    after subsampling."""
    import json
    from scripts.train_bi_encoder import TripleJsonlDataset

    rows = [{
        "query": "q", "pos": ["p"],
        "neg": [f"text_{i}" for i in range(15)],
        "neg_tids": [f"tid_{i}" for i in range(15)],
        "pos_tid": "gold", "user_id": "u", "session_id": "s",
    }]
    path = tmp_path / "t.jsonl"
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    ds = TripleJsonlDataset(str(path), n_negatives=15, split_key="row")
    item = ds[0]
    assert "neg_tids" in item
    assert len(item["neg_tids"]) == 15
    # Order matches negatives.
    for k, neg_text in enumerate(item["negatives"]):
        # Find the original index that emitted this text.
        original_idx = int(neg_text.split("_")[1])
        assert item["neg_tids"][k] == f"tid_{original_idx}"


def test_dataset_neg_tids_stay_aligned_under_subsampling(tmp_path):
    """When n_negatives < len(row.neg), the dataset subsamples WITHOUT
    replacement — neg_tids must be subsampled with the SAME indices so they
    remain aligned with the kept neg texts."""
    import json
    from scripts.train_bi_encoder import TripleJsonlDataset

    rows = [{
        "query": "q", "pos": ["p"],
        "neg": [f"text_{i}" for i in range(20)],
        "neg_tids": [f"tid_{i}" for i in range(20)],
        "pos_tid": "gold", "user_id": "u", "session_id": "s",
    }]
    path = tmp_path / "t.jsonl"
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    ds = TripleJsonlDataset(str(path), n_negatives=10, split_key="row", seed=1)
    item = ds[0]
    assert len(item["negatives"]) == 10
    assert len(item["neg_tids"]) == 10
    for k in range(10):
        original_idx = int(item["negatives"][k].split("_")[1])
        assert item["neg_tids"][k] == f"tid_{original_idx}", \
            f"slot {k}: text {item['negatives'][k]} but tid {item['neg_tids'][k]}"


def test_train_loop_emits_train_inbatch_ndcg_metric():
    """Quick-iter diagnostic (user request): train loop logs `train/ndcg_inbatch`
    every --logging-steps using the SAME (B, n_per) per-row score matrix that
    val/ndcg uses. Directly comparable curves on TB; gap = leak signature."""
    import inspect
    from scripts import train_bi_encoder as mod
    src = inspect.getsource(mod._train)
    assert "train/ndcg_inbatch" in src, \
        "_train must log train/ndcg_inbatch for train-vs-val alignment check"
    assert "train/top1_inbatch" in src, \
        "_train must log train/top1_inbatch alongside ndcg"
    assert "_val_metrics_from_scores" in src, \
        "_train must reuse _val_metrics_from_scores for an apples-to-apples curve"


def test_dataset_default_split_key_is_user_id():
    """Δ2: when caller passes split='train' without specifying split_key,
    the default is 'user_id' (NOT session_id or row)."""
    import inspect
    from scripts.train_bi_encoder import TripleJsonlDataset
    sig = inspect.signature(TripleJsonlDataset.__init__)
    default = sig.parameters["split_key"].default
    assert default == "user_id", f"expected split_key default 'user_id', got {default!r}"
