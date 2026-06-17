"""K3b config knobs — tests for cross-encoder LoRA fine-tune config schema.

Per .claude/documents/features/53_K3b_ce_lora_finetune.md §9 and
.claude/documents/features/11_F2_interfaces_contracts_config.md §9.

Tests: load a default config (no overrides) and assert every K3b knob is
exposed under rerank.neural.* with the correct default value and type.
Also verifies partial overrides work and unknown keys are still rejected.
"""
from __future__ import annotations

import pytest

from mcrs.config import load_config


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _default() -> object:
    """Load a fully-defaulted config (no YAML, no overrides)."""
    return load_config(None)


# ---------------------------------------------------------------------------
# rerank.neural.lora
# ---------------------------------------------------------------------------

def test_lora_enabled_default():
    cfg = _default()
    assert cfg.rerank.neural.lora.enabled is False


def test_lora_r_default():
    cfg = _default()
    assert cfg.rerank.neural.lora.r == 16
    assert isinstance(cfg.rerank.neural.lora.r, int)


def test_lora_alpha_default():
    cfg = _default()
    assert cfg.rerank.neural.lora.alpha == 32
    assert isinstance(cfg.rerank.neural.lora.alpha, int)


def test_lora_dropout_default():
    cfg = _default()
    assert cfg.rerank.neural.lora.dropout == pytest.approx(0.05)
    assert isinstance(cfg.rerank.neural.lora.dropout, float)


def test_lora_target_modules_default():
    cfg = _default()
    assert cfg.rerank.neural.lora.target_modules == ["query", "value"]
    assert isinstance(cfg.rerank.neural.lora.target_modules, list)


# ---------------------------------------------------------------------------
# rerank.neural top-level
# ---------------------------------------------------------------------------

def test_neural_max_length_default():
    cfg = _default()
    assert cfg.rerank.neural.max_length == 2048
    assert isinstance(cfg.rerank.neural.max_length, int)


def test_neural_max_doc_tokens_default():
    cfg = _default()
    assert cfg.rerank.neural.max_doc_tokens == 1100
    assert isinstance(cfg.rerank.neural.max_doc_tokens, int)


def test_neural_dtype_default():
    cfg = _default()
    assert cfg.rerank.neural.dtype == "auto"
    assert isinstance(cfg.rerank.neural.dtype, str)


def test_neural_adapter_revision_default():
    cfg = _default()
    assert cfg.rerank.neural.adapter_revision is None


# ---------------------------------------------------------------------------
# rerank.neural.negatives
# ---------------------------------------------------------------------------

def test_negatives_n_default():
    cfg = _default()
    assert cfg.rerank.neural.negatives.n == 15
    assert isinstance(cfg.rerank.neural.negatives.n, int)


def test_negatives_k_min_default():
    cfg = _default()
    assert cfg.rerank.neural.negatives.k_min == 4
    assert isinstance(cfg.rerank.neural.negatives.k_min, int)


def test_negatives_sampling_default():
    cfg = _default()
    assert cfg.rerank.neural.negatives.sampling == "rank_strat"
    assert isinstance(cfg.rerank.neural.negatives.sampling, str)


def test_negatives_same_artist_default():
    cfg = _default()
    assert cfg.rerank.neural.negatives.same_artist == "soft_downweight"
    assert isinstance(cfg.rerank.neural.negatives.same_artist, str)


def test_negatives_denoise_near_dup_default():
    cfg = _default()
    assert cfg.rerank.neural.negatives.denoise_near_dup is True


def test_negatives_skip_top_rank_default():
    cfg = _default()
    assert cfg.rerank.neural.negatives.skip_top_rank is False


# ---------------------------------------------------------------------------
# rerank.neural.goal_progress
# ---------------------------------------------------------------------------

def test_goal_progress_enabled_default():
    cfg = _default()
    assert cfg.rerank.neural.goal_progress.enabled is False


def test_goal_progress_w_low_default():
    cfg = _default()
    assert cfg.rerank.neural.goal_progress.w_low == pytest.approx(0.3)
    assert isinstance(cfg.rerank.neural.goal_progress.w_low, float)


# ---------------------------------------------------------------------------
# rerank.neural.oof
# ---------------------------------------------------------------------------

def test_oof_folds_default():
    cfg = _default()
    assert cfg.rerank.neural.oof.folds == 3
    assert isinstance(cfg.rerank.neural.oof.folds, int)


def test_oof_dedup_cross_fold_default():
    cfg = _default()
    assert cfg.rerank.neural.oof.dedup_cross_fold is True


def test_oof_score_norm_default():
    cfg = _default()
    assert cfg.rerank.neural.oof.score_norm == "within_pool"
    assert isinstance(cfg.rerank.neural.oof.score_norm, str)


# ---------------------------------------------------------------------------
# rerank.neural.train
# ---------------------------------------------------------------------------

def test_train_epochs_default():
    cfg = _default()
    assert cfg.rerank.neural.train.epochs == 3
    assert isinstance(cfg.rerank.neural.train.epochs, int)


def test_train_lr_default():
    cfg = _default()
    assert cfg.rerank.neural.train.lr == pytest.approx(1e-4)
    assert isinstance(cfg.rerank.neural.train.lr, float)


def test_train_weight_decay_default():
    cfg = _default()
    assert cfg.rerank.neural.train.weight_decay == pytest.approx(0.0)
    assert isinstance(cfg.rerank.neural.train.weight_decay, float)


def test_train_batch_groups_default():
    cfg = _default()
    assert cfg.rerank.neural.train.batch_groups == 2
    assert isinstance(cfg.rerank.neural.train.batch_groups, int)


def test_train_grad_accum_default():
    cfg = _default()
    assert cfg.rerank.neural.train.grad_accum == 16
    assert isinstance(cfg.rerank.neural.train.grad_accum, int)


def test_train_warmup_default():
    cfg = _default()
    assert cfg.rerank.neural.train.warmup == pytest.approx(0.05)
    assert isinstance(cfg.rerank.neural.train.warmup, float)


def test_train_early_stop_patience_default():
    cfg = _default()
    assert cfg.rerank.neural.train.early_stop_patience == 1
    assert isinstance(cfg.rerank.neural.train.early_stop_patience, int)


def test_train_group_by_length_default():
    cfg = _default()
    assert cfg.rerank.neural.train.group_by_length is True


def test_train_log_every_default():
    cfg = _default()
    assert cfg.rerank.neural.train.log_every == 50
    assert isinstance(cfg.rerank.neural.train.log_every, int)


def test_train_seed_default():
    cfg = _default()
    assert cfg.rerank.neural.train.seed == 0
    assert isinstance(cfg.rerank.neural.train.seed, int)


# ---------------------------------------------------------------------------
# Partial override — value round-trips correctly
# ---------------------------------------------------------------------------

def test_override_lora_r():
    cfg = load_config({"rerank": {"neural": {"lora": {"r": 8}}}})
    assert cfg.rerank.neural.lora.r == 8
    # sibling untouched
    assert cfg.rerank.neural.lora.alpha == 32


def test_override_train_epochs():
    cfg = load_config({"rerank": {"neural": {"train": {"epochs": 5}}}})
    assert cfg.rerank.neural.train.epochs == 5
    # other defaults intact
    assert cfg.rerank.neural.train.lr == pytest.approx(1e-4)


def test_override_adapter_revision():
    cfg = load_config({"rerank": {"neural": {"adapter_revision": "abc123"}}})
    assert cfg.rerank.neural.adapter_revision == "abc123"


# ---------------------------------------------------------------------------
# Unknown key still rejected (regression guard)
# ---------------------------------------------------------------------------

def test_unknown_neural_key_rejected():
    with pytest.raises(ValueError):
        load_config({"rerank": {"neural": {"bogus_key": 1}}})


def test_unknown_lora_key_rejected():
    with pytest.raises(ValueError):
        load_config({"rerank": {"neural": {"lora": {"unknown": True}}}})


def test_unknown_negatives_key_rejected():
    with pytest.raises(ValueError):
        load_config({"rerank": {"neural": {"negatives": {"mystery": 1}}}})


# ---------------------------------------------------------------------------
# rerank.neural.train.gradient_checkpointing (new)
# ---------------------------------------------------------------------------

def test_train_gradient_checkpointing_default():
    cfg = _default()
    assert cfg.rerank.neural.train.gradient_checkpointing is True


# ---------------------------------------------------------------------------
# rerank.neural.split (new section)
# ---------------------------------------------------------------------------

def test_split_key_default():
    cfg = _default()
    assert cfg.rerank.neural.split.key == "session"
    assert isinstance(cfg.rerank.neural.split.key, str)


def test_split_dedup_near_dup_default():
    cfg = _default()
    assert cfg.rerank.neural.split.dedup_near_dup is True


# ---------------------------------------------------------------------------
# query (new top-level section)
# ---------------------------------------------------------------------------

def test_query_markers_default():
    cfg = _default()
    assert cfg.query.markers is True


def test_query_taste_items_default():
    cfg = _default()
    assert cfg.query.taste_items == 5
    assert isinstance(cfg.query.taste_items, int)


# ---------------------------------------------------------------------------
# logging (new top-level section)
# ---------------------------------------------------------------------------

def test_logging_trackio_enabled_default():
    cfg = _default()
    assert cfg.logging.trackio.enabled is False


def test_logging_trackio_project_default():
    cfg = _default()
    assert cfg.logging.trackio.project == "recsys2026"
    assert isinstance(cfg.logging.trackio.project, str)


# ---------------------------------------------------------------------------
# Existing top-level keys still resolve (non-regression)
# ---------------------------------------------------------------------------

def test_existing_keys_unaffected():
    cfg = _default()
    assert cfg.seed == 42
    assert cfg.segment.cold_threshold == 1
    assert cfg.retrieval.topk == 300
    assert cfg.paths.data_root == "./data"
