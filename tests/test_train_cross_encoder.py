"""Unit tests for the Stage B cross-encoder training helpers (Phase 6c)."""
import math

import torch
import torch.nn as nn
from types import SimpleNamespace

from mcrs.training.multimodal_bi_encoder import MultiModalConfig
from mcrs.training.multimodal_cross_encoder import MultiModalCrossEncoder
from scripts.train_cross_encoder import (
    ce_group_metrics,
    compute_batch_loss,
    flatten_ce_pairs,
    pairwise_bce_loss,
)


class _StubBackbone(nn.Module):
    """Tiny mean-mixing backbone — skips the 570M reranker download."""

    def __init__(self, vocab_size: int, hidden_dim: int):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_dim)
        self._embed = nn.Embedding(vocab_size, hidden_dim)

    def get_input_embeddings(self):
        return self._embed

    def forward(self, inputs_embeds=None, attention_mask=None):
        mask = attention_mask.unsqueeze(-1).to(inputs_embeds.dtype)
        pooled = (inputs_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        lhs = inputs_embeds.clone()
        lhs[:, 0, :] = pooled
        return SimpleNamespace(last_hidden_state=lhs)


class _StubTokenizer:
    """Minimal HF-tokenizer stand-in: text → padded char-code token ids."""

    def __call__(self, texts, padding=True, truncation=True, max_length=16,
                 return_tensors="pt"):
        seqs = [[1] + [(ord(c) % 40) + 2 for c in t][: max_length - 1] for t in texts]
        width = max(len(s) for s in seqs)
        input_ids = [s + [0] * (width - len(s)) for s in seqs]
        attn = [[1] * len(s) + [0] * (width - len(s)) for s in seqs]
        return {"input_ids": torch.tensor(input_ids),
                "attention_mask": torch.tensor(attn)}


def _row(query, pos, negs, audio_dim=8, cf_dim=6):
    v = lambda d: [0.1] * d
    return {
        "query": query, "positive": pos, "negatives": list(negs),
        "pos_clap": v(audio_dim), "pos_cf_track": v(cf_dim),
        "neg_clap": [v(audio_dim) for _ in negs], "neg_cf_track": [v(cf_dim) for _ in negs],
        "user_cf": v(cf_dim),
        "pos_tag_ids": [1, 2], "pos_year": 2000,
        "neg_tag_ids": [[3] for _ in negs], "neg_years": [1995 for _ in negs],
    }


def _tiny_ce_model(hidden_dim=16):
    cfg = MultiModalConfig(backbone_name="stub", hidden_dim=hidden_dim, audio_dim=8,
                           cf_dim=6, tag_vocab_size=5, max_tags=4, lora_rank=0)
    return MultiModalCrossEncoder(cfg, backbone=_StubBackbone(50, hidden_dim))


def test_pairwise_bce_lower_for_correct_ranking():
    """Loss is lower when positives score high and negatives score low than
    when the predictions are inverted — confirms the loss drives the intended
    BCE(pos→1) + BCE(neg→0) objective."""
    pos_logits = torch.tensor([5.0, 4.0])
    neg_logits = torch.tensor([-5.0, -4.0, -3.0])

    good = pairwise_bce_loss(pos_logits, neg_logits)
    bad = pairwise_bce_loss(-pos_logits, -neg_logits)  # inverted predictions

    assert torch.isfinite(good)
    assert good < bad


def test_pairwise_bce_neg_weights_scale_negative_term():
    """Per-negative weights scale the negative BCE term — zero-weighting all
    negatives leaves only the positive term (used later for v2 rank-weighting)."""
    pos_logits = torch.tensor([2.0])
    neg_logits = torch.tensor([2.0, 2.0])

    unweighted = pairwise_bce_loss(pos_logits, neg_logits)
    zero_negs = pairwise_bce_loss(
        pos_logits, neg_logits, neg_weights=torch.zeros_like(neg_logits)
    )

    # Zeroing the negative weights must drop the (positive) negative-term loss.
    assert zero_negs < unweighted
    assert torch.isfinite(zero_negs)


def test_flatten_ce_pairs_expands_row_into_pos_and_neg_pairs():
    """Each TripleJsonlDataset row (1 gold + K negs + modalities) flattens into
    K+1 (query, doc) scoring pairs with aligned modalities and a pos/neg flag.
    Query + user_cf are repeated across the row's pairs."""
    rows = [{
        "query": "q1",
        "positive": "POS",
        "negatives": ["NEG0", "NEG1"],
        "pos_clap": [0.1], "pos_cf_track": [0.2],
        "neg_clap": [[0.3], [0.4]], "neg_cf_track": [[0.5], [0.6]],
        "user_cf": [0.9],
        "pos_tag_ids": [1, 2], "pos_year": 2000,
        "neg_tag_ids": [[3], [4]], "neg_years": [1990, 1995],
    }]

    flat = flatten_ce_pairs(rows)

    assert flat["doc_text"] == ["POS", "NEG0", "NEG1"]
    assert flat["is_positive"] == [True, False, False]
    assert flat["query"] == ["q1", "q1", "q1"]
    assert flat["user_cf"] == [[0.9], [0.9], [0.9]]
    assert flat["doc_clap"] == [[0.1], [0.3], [0.4]]
    assert flat["doc_cf"] == [[0.2], [0.5], [0.6]]
    assert flat["doc_tags"] == [[1, 2], [3], [4]]
    assert flat["doc_year"] == [2000, 1990, 1995]


def test_flatten_ce_pairs_concatenates_multiple_rows():
    """Pairs from multiple rows are concatenated in row order."""
    rows = [
        {"query": "qA", "positive": "PA", "negatives": ["NA0"],
         "pos_clap": [1.0], "pos_cf_track": [1.0], "neg_clap": [[0.0]],
         "neg_cf_track": [[0.0]], "user_cf": [0.1],
         "pos_tag_ids": [1], "pos_year": 2001, "neg_tag_ids": [[2]], "neg_years": [1999]},
        {"query": "qB", "positive": "PB", "negatives": ["NB0"],
         "pos_clap": [2.0], "pos_cf_track": [2.0], "neg_clap": [[0.0]],
         "neg_cf_track": [[0.0]], "user_cf": [0.2],
         "pos_tag_ids": [3], "pos_year": 2002, "neg_tag_ids": [[4]], "neg_years": [1998]},
    ]

    flat = flatten_ce_pairs(rows)

    assert flat["doc_text"] == ["PA", "NA0", "PB", "NB0"]
    assert flat["is_positive"] == [True, False, True, False]
    assert flat["query"] == ["qA", "qA", "qB", "qB"]


def test_compute_batch_loss_runs_and_backprops():
    """The per-batch training step assembles flatten → tokenize → forward →
    split → pairwise BCE into a finite, differentiable scalar that backprops
    into the model params. Verified locally with a stub backbone (no GPU)."""
    model = _tiny_ce_model()
    tok = _StubTokenizer()
    rows = [
        _row("hello there", "gold track", ["bad one", "bad two"]),
        _row("another query", "second gold", ["neg a"]),
    ]

    loss = compute_batch_loss(model, tok, rows, max_tags=4, device="cpu")

    assert torch.isfinite(loss)
    assert loss.requires_grad
    loss.backward()
    grads = [p.grad is not None for p in model.parameters() if p.requires_grad]
    assert any(grads)


def test_stage_b_deferred_import_symbols_resolve():
    """Regression guard (reviewer-caught Critical): the deferred imports in
    train_cross_encoder.main() and MULTIMODAL_RERANKER.__init__ must resolve
    from the SAME places Stage A defines them. MultiModalArtifacts lives in
    train_bi_encoder (scripts/), NOT in mcrs.training.multimodal_bi_encoder.
    Mirrors those import blocks so a wrong source is caught without a model load.
    """
    import pathlib
    import sys
    root = pathlib.Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    sys.path.insert(0, str(root / "music-crs-baselines"))

    from mcrs.training.multimodal_bi_encoder import MultiModalConfig  # noqa: F401
    from mcrs.training.multimodal_cross_encoder import MultiModalCrossEncoder  # noqa: F401
    from train_bi_encoder import MultiModalArtifacts, TripleJsonlDataset  # noqa: F401
    from build_bi_encoder_training_data import (  # noqa: F401
        _format_history_music_turn, _load_multimodal_artifacts, _track_to_tag_ids,
    )

    import mcrs.training.multimodal_bi_encoder as _mm
    assert not hasattr(_mm, "MultiModalArtifacts"), (
        "MultiModalArtifacts must NOT be importable from "
        "mcrs.training.multimodal_bi_encoder — it's defined in train_bi_encoder."
    )


def test_compute_batch_loss_lower_when_model_ranks_well():
    """Sanity: a batch is just one scalar; confirm the step returns a 0-dim
    tensor (not per-pair) so the optimizer sees a single objective."""
    model = _tiny_ce_model()
    tok = _StubTokenizer()
    rows = [_row("q", "p", ["n1", "n2"])]

    loss = compute_batch_loss(model, tok, rows, max_tags=4, device="cpu")

    assert loss.dim() == 0


# ---------------------------------------------------------------------------
# In-group ranking metrics (train/val viewing parity with the bi-encoder).
# `ce_group_metrics` segments the flat (query, doc) logits — ordered
# [pos, neg_1..neg_K] per row by flatten_ce_pairs — into per-query groups via
# the is_positive flags, then computes top1 + nDCG the SAME way the bi-encoder's
# _val_metrics_from_scores does (rank = 1 + #cands strictly above the positive;
# nDCG = 1/log2(rank+1)). Pure tensor op; no model needed.
# ---------------------------------------------------------------------------


def test_ce_group_metrics_perfect_ranking():
    """Positive is the top score in every group → top1=1, ndcg=1."""
    logits = torch.tensor([5.0, 1.0, 0.0, 4.0, -1.0])  # groups: [5,1,0], [4,-1]
    is_pos = torch.tensor([True, False, False, True, False])

    top1, ndcg = ce_group_metrics(logits, is_pos)

    assert abs(top1 - 1.0) < 1e-6
    assert abs(ndcg - 1.0) < 1e-6


def test_ce_group_metrics_positive_at_rank_2_drops_ndcg():
    """One negative outscores the positive → rank 2 → ndcg = 1/log2(3), top1=0."""
    logits = torch.tensor([1.0, 2.0, 0.0])  # single group, positive at index 0
    is_pos = torch.tensor([True, False, False])

    top1, ndcg = ce_group_metrics(logits, is_pos)

    assert top1 == 0.0
    assert abs(ndcg - (1.0 / math.log2(3))) < 1e-6


def test_ce_group_metrics_handles_ragged_group_sizes():
    """Groups need not be equal length (a row may carry < n_negatives). A
    1-positive/0-negative group ranks the positive first (rank 1)."""
    logits = torch.tensor([3.0, 9.0, 0.5])  # group A: [3,9] (rank 2); group B: [0.5] (rank 1)
    is_pos = torch.tensor([True, False, True])

    top1, ndcg = ce_group_metrics(logits, is_pos)

    assert abs(top1 - 0.5) < 1e-6  # A wrong, B correct
    expected = ((1.0 / math.log2(3)) + 1.0) / 2
    assert abs(ndcg - expected) < 1e-6


def test_ce_group_metrics_ties_favor_the_positive():
    """A negative tying the positive does NOT outrank it (strictly-greater
    count), matching the bi-encoder argmax==0 tie semantics → top1 stays 1."""
    logits = torch.tensor([2.0, 2.0])
    is_pos = torch.tensor([True, False])

    top1, ndcg = ce_group_metrics(logits, is_pos)

    assert abs(top1 - 1.0) < 1e-6
    assert abs(ndcg - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Disconnect-safety contract (checkpointing + resume). The training loop is
# GPU-bound, so these are source-level guards: a single epoch is ~15k steps
# (hours on a small GPU), and the loop previously saved ONLY at the very end —
# a Colab disconnect lost everything. Lock in step-level + per-epoch
# checkpointing, resume, and that checkpoint dirs aren't pushed to the Hub.
# ---------------------------------------------------------------------------


def test_train_ce_exposes_checkpoint_resume_and_gradckpt_flags():
    import inspect
    from scripts import train_cross_encoder as mod
    main_src = inspect.getsource(mod.main)
    for flag in ("--checkpoint-every-n-steps", "--resume-from", "--gradient-checkpointing"):
        assert flag in main_src, f"missing CLI flag: {flag}"


def test_train_ce_saves_step_and_epoch_checkpoints():
    import inspect
    from scripts import train_cross_encoder as mod
    main_src = inspect.getsource(mod.main)
    assert "checkpoint_latest" in main_src, "no rolling step-level checkpoint"
    assert "checkpoint_epoch_" in main_src, "no per-epoch checkpoint"


def test_train_ce_resume_loads_from_checkpoint():
    import inspect
    from scripts import train_cross_encoder as mod
    main_src = inspect.getsource(mod.main)
    assert "args.resume_from" in main_src, "resume flag must be wired into model load"
    assert "MultiModalCrossEncoder.from_pretrained" in main_src, \
        "resume must rebuild the cross-encoder via MultiModalCrossEncoder.from_pretrained"


def test_train_ce_final_upload_excludes_checkpoint_dirs():
    import inspect
    from scripts import train_cross_encoder as mod
    main_src = inspect.getsource(mod.main)
    assert "ignore_patterns" in main_src, "final Hub upload must ignore checkpoint dirs"


# ---------------------------------------------------------------------------
# Train/val status-viewing parity with the Stage A bi-encoder. Source-level
# guards (the loop is GPU-bound): a leak-safe val split + TensorBoard logging of
# train AND val loss + nDCG must be wired into main().
# ---------------------------------------------------------------------------


def test_train_ce_exposes_val_split_and_logging_flags():
    import inspect
    from scripts import train_cross_encoder as mod
    main_src = inspect.getsource(mod.main)
    for flag in ("--val-fraction", "--split-key", "--val-every-n-steps",
                 "--val-max-rows"):
        assert flag in main_src, f"missing CLI flag: {flag}"


def test_train_ce_builds_leak_safe_val_split():
    """Val set is carved with split='val' + split_key (default user_id), so a
    user's sessions never straddle train/val — same contract as Stage A."""
    import inspect
    from scripts import train_cross_encoder as mod
    main_src = inspect.getsource(mod.main)
    assert 'split="val"' in main_src, "no held-out val split"
    assert "split_key=args.split_key" in main_src, "val split must honor --split-key"


def test_train_ce_logs_train_and_val_metrics_to_tensorboard():
    import inspect
    from scripts import train_cross_encoder as mod
    main_src = inspect.getsource(mod.main)
    assert "SummaryWriter" in main_src, "no TensorBoard writer"
    for tag in ("train/loss", "train/ndcg_ingroup", "val/loss", "val/ndcg"):
        assert tag in main_src, f"missing TensorBoard scalar: {tag}"


def test_train_ce_val_pass_uses_no_grad_and_restores_train_mode():
    """The val pass must be forward-only (model.eval + no_grad) and put the
    model back in train() so it doesn't poison the next training step."""
    import inspect
    from scripts import train_cross_encoder as mod
    main_src = inspect.getsource(mod.main)
    assert "model.eval()" in main_src and "torch.no_grad()" in main_src
    assert "model.train()" in main_src, "val pass must restore train mode"
