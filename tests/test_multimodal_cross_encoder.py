"""Unit tests for the Stage B MultiModalCrossEncoder (Phase 6b).

Uses a tiny stub backbone so tests run without downloading the 570M
bge-reranker-v2-m3 weights. The stub mirrors the minimal transformers
surface the model relies on: ``.config.hidden_size``,
``get_input_embeddings()``, and a ``forward(inputs_embeds=, attention_mask=)``
returning an object with ``.last_hidden_state``.
"""
from types import SimpleNamespace

import torch
import torch.nn as nn

from mcrs.training.multimodal_bi_encoder import MultiModalConfig
from mcrs.training.multimodal_cross_encoder import MultiModalCrossEncoder


class _StubBackbone(nn.Module):
    """Minimal stand-in for a HF encoder backbone."""

    def __init__(self, vocab_size: int, hidden_dim: int):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_dim)
        self._embed = nn.Embedding(vocab_size, hidden_dim)

    def get_input_embeddings(self):
        return self._embed

    def forward(self, inputs_embeds=None, attention_mask=None):
        # Mean-mix over attended tokens into position 0 so the pooled CLS
        # depends on EVERY token (incl. modality tokens) — lets tests detect
        # that masking a modality actually changes the score.
        mask = attention_mask.unsqueeze(-1).to(inputs_embeds.dtype)  # (B, L, 1)
        summed = (inputs_embeds * mask).sum(dim=1)                   # (B, D)
        denom = mask.sum(dim=1).clamp(min=1.0)                       # (B, 1)
        pooled = summed / denom                                      # (B, D)
        lhs = inputs_embeds.clone()
        lhs[:, 0, :] = pooled
        return SimpleNamespace(last_hidden_state=lhs)


def _tiny_config(hidden_dim: int = 16) -> MultiModalConfig:
    return MultiModalConfig(
        backbone_name="stub",
        hidden_dim=hidden_dim,
        audio_dim=8,
        cf_dim=6,
        tag_vocab_size=5,
        max_tags=4,
        lora_rank=0,  # Stage B is full fine-tune, no LoRA
    )


def _dummy_pair_batch(batch_size: int = 2, q_len: int = 5, d_len: int = 7) -> dict:
    return dict(
        query_input_ids=torch.randint(1, 50, (batch_size, q_len)),
        query_attention_mask=torch.ones(batch_size, q_len, dtype=torch.long),
        query_user_cf=torch.randn(batch_size, 6),
        doc_input_ids=torch.randint(1, 50, (batch_size, d_len)),
        doc_attention_mask=torch.ones(batch_size, d_len, dtype=torch.long),
        doc_clap=torch.randn(batch_size, 8),
        doc_cf=torch.randn(batch_size, 6),
        doc_tags=torch.randint(0, 5, (batch_size, 4)),
        doc_year=torch.tensor([2000, 1995]),
    )


def test_forward_returns_one_finite_score_per_pair():
    """The cross-encoder scores each (query, doc) pair to a single scalar."""
    D = 16
    model = MultiModalCrossEncoder(_tiny_config(D), backbone=_StubBackbone(50, D))
    batch = _dummy_pair_batch(batch_size=2)

    scores = model(**batch)

    assert scores.shape == (2,)
    assert torch.isfinite(scores).all()


def test_modality_mask_zeros_a_modality_and_changes_score():
    """Zeroing a modality token via modality_mask must change the pair score —
    proves the modality token participates and the mask (used for training
    dropout + Phase 9 ablation gates) is wired through. Mirrors Stage A's
    modality_mask dict semantics."""
    D = 16
    torch.manual_seed(0)
    model = MultiModalCrossEncoder(_tiny_config(D), backbone=_StubBackbone(50, D))
    model.eval()
    batch = _dummy_pair_batch(batch_size=2)

    with torch.no_grad():
        full = model(**batch)
        audio_off = model(**batch, modality_mask={"audio": torch.zeros(2, 1)})

    assert audio_off.shape == full.shape
    assert not torch.allclose(full, audio_off)


def test_modality_mask_all_ones_matches_unmasked():
    """An all-ones mask is a no-op — identical scores to passing no mask."""
    D = 16
    torch.manual_seed(0)
    model = MultiModalCrossEncoder(_tiny_config(D), backbone=_StubBackbone(50, D))
    model.eval()
    batch = _dummy_pair_batch(batch_size=2)
    all_ones = {k: torch.ones(2, 1) for k in ("user_cf", "audio", "cf", "tag", "release")}

    with torch.no_grad():
        unmasked = model(**batch)
        masked = model(**batch, modality_mask=all_ones)

    assert torch.allclose(unmasked, masked)


def test_save_and_load_round_trip(tmp_path):
    """save_pretrained → from_pretrained restores the modality heads + scoring
    head, reproducing identical scores. Backbone is injected (same instance)
    so this isolates the heads/config round-trip that mirrors Stage A."""
    D = 16
    torch.manual_seed(0)
    backbone = _StubBackbone(50, D)
    model = MultiModalCrossEncoder(_tiny_config(D), backbone=backbone)
    model.eval()
    batch = _dummy_pair_batch(batch_size=2)
    with torch.no_grad():
        before = model(**batch)

    out_dir = tmp_path / "ce_model"
    model.save_pretrained(str(out_dir))
    loaded = MultiModalCrossEncoder.from_pretrained(str(out_dir), backbone=backbone)
    loaded.eval()
    with torch.no_grad():
        after = loaded(**batch)

    assert torch.allclose(before, after, atol=1e-6)
