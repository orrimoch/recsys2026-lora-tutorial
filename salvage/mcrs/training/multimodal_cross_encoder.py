"""Stage B: multi-modal cross-encoder reranker (Phase 6b).

Single-tower reranker over a fused (query, candidate) sequence::

    [CLS, <user_cf>, query_text..., <audio>, <cf>, <tag>, <release>, doc_text...]

Full self-attention lets the model learn cross-features the bi-encoder cannot
represent (``user_cf · track_cf``, ``query_text ↔ track_audio``). Backbone is
``BAAI/bge-reranker-v2-m3`` with FULL fine-tune (no LoRA). The modality
projections deliberately share shapes with Stage A's ``MultiModalBiEncoder``
so the Phase 0 precomputed artifacts (CLAP / CF / tag vocab / release year)
are reused unchanged.

The ``CLS`` used for scoring is the query sequence's position-0 token; the
``user_cf`` token is spliced right after it, then the rest of the query text,
the four candidate modality tokens, and finally the candidate text.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Optional

import torch
import torch.nn as nn

from mcrs.training.multimodal_bi_encoder import DEFAULT_LORA_TARGETS, MultiModalConfig


class MultiModalCrossEncoder(nn.Module):
    """Cross-encoder scorer: (query + modalities, doc + modalities) → relevance logit."""

    def __init__(self, config: MultiModalConfig, backbone: Optional[nn.Module] = None):
        super().__init__()
        self.config = config

        if backbone is None:
            from transformers import AutoModel
            backbone = AutoModel.from_pretrained(config.backbone_name)
        bb_dim = getattr(backbone.config, "hidden_size", None)
        if bb_dim is None or bb_dim != config.hidden_dim:
            raise ValueError(
                f"backbone hidden_size={bb_dim} doesn't match config.hidden_dim="
                f"{config.hidden_dim}. Update MultiModalConfig.hidden_dim or pass a "
                f"different backbone."
            )
        # Stage B is a full fine-tune — the backbone trains directly, no LoRA
        # wrap even if config.lora_rank > 0 (that field is Stage A's).
        self.backbone = backbone

        D = config.hidden_dim
        # Modality projections — shared shapes with Stage A so Phase 0 artifacts
        # load unchanged. Fully trained (tiny, ~5M params).
        self.audio_proj = nn.Sequential(nn.Linear(config.audio_dim, D), nn.LayerNorm(D))
        self.cf_track_proj = nn.Sequential(nn.Linear(config.cf_dim, D), nn.LayerNorm(D))
        self.cf_user_proj = nn.Sequential(nn.Linear(config.cf_dim, D), nn.LayerNorm(D))
        if config.tag_vocab_size < 2:
            raise ValueError(
                f"tag_vocab_size must be >= 2 (need pad + at least 1 real tag), "
                f"got {config.tag_vocab_size}. Pass a real vocab built from "
                f"experiments/cache/multimodal/tag_vocab.json."
            )
        self.tag_embed = nn.Embedding(config.tag_vocab_size, D, padding_idx=0)
        self.release_proj = nn.Linear(1, D)
        for module in (self.audio_proj[0], self.cf_track_proj[0],
                       self.cf_user_proj[0], self.release_proj):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.tag_embed.weight, std=0.02)
        with torch.no_grad():
            self.tag_embed.weight[0].zero_()  # pad row stays zero

        # Scoring head: CLS → scalar relevance logit (BCE-with-logits at train,
        # sigmoid at inference).
        self.score_head = nn.Linear(D, 1)

    # --------------------------------------------------------------- helpers
    def _input_embeds(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.backbone.get_input_embeddings()(input_ids)

    def _backbone_forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor):
        return self.backbone(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

    def _year_to_scalar(self, year: torch.Tensor) -> torch.Tensor:
        """Year-int → scalar in [-1, 1], shape (B, 1). Mirrors Stage A."""
        c = self.config
        y = year.float().clamp(min=c.year_origin)
        normalized = ((y - c.year_origin) / c.year_span).clamp(-1.0, 1.0)
        unknown = (year < 0)
        normalized = torch.where(unknown, torch.zeros_like(normalized), normalized)
        return normalized.unsqueeze(-1)

    def _tag_token(self, tag_ids: torch.Tensor) -> torch.Tensor:
        """Mean-pool tag embeddings over non-pad positions. Returns (B, D)."""
        embs = self.tag_embed(tag_ids)               # (B, max_tags, D)
        mask = (tag_ids != 0).float().unsqueeze(-1)   # (B, max_tags, 1)
        summed = (embs * mask).sum(dim=1)             # (B, D)
        denom = mask.sum(dim=1).clamp(min=1.0)        # (B, 1)
        return summed / denom

    # --------------------------------------------------------------- forward
    def forward(
        self,
        query_input_ids: torch.Tensor,        # (B, Lq) long
        query_attention_mask: torch.Tensor,   # (B, Lq)
        query_user_cf: torch.Tensor,          # (B, cf_dim) — mean fallback for cold
        doc_input_ids: torch.Tensor,          # (B, Ld) long
        doc_attention_mask: torch.Tensor,     # (B, Ld)
        doc_clap: torch.Tensor,               # (B, audio_dim)
        doc_cf: torch.Tensor,                 # (B, cf_dim)
        doc_tags: torch.Tensor,               # (B, max_tags) long, pad=0
        doc_year: torch.Tensor,               # (B,) long (int year or -1)
        modality_mask: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Score each (query, doc) pair. Returns logits of shape (B,).

        ``modality_mask`` (optional): dict with optional keys ``user_cf``,
        ``audio``, ``cf``, ``tag``, ``release``. Each value is a (B, 1) float
        tensor in {0., 1.} used to zero out that modality token (training
        dropout / Phase 9 ablation). Missing key = keep the token. Mirrors
        Stage A's ``MultiModalBiEncoder`` modality-mask semantics.
        """
        B = query_input_ids.shape[0]
        device = query_input_ids.device

        q_embeds = self._input_embeds(query_input_ids)   # (B, Lq, D)
        d_embeds = self._input_embeds(doc_input_ids)      # (B, Ld, D)

        user_tok = self.cf_user_proj(query_user_cf)                          # (B, D)
        audio_tok = self.audio_proj(doc_clap)                                # (B, D)
        cf_tok = self.cf_track_proj(doc_cf)                                  # (B, D)
        tag_tok = self._tag_token(doc_tags)                                  # (B, D)
        release_tok = self.release_proj(self._year_to_scalar(doc_year))      # (B, D)

        if modality_mask is not None:
            ones = torch.ones(B, 1, device=device)
            user_tok = user_tok * modality_mask.get("user_cf", ones)
            audio_tok = audio_tok * modality_mask.get("audio", ones)
            cf_tok = cf_tok * modality_mask.get("cf", ones)
            tag_tok = tag_tok * modality_mask.get("tag", ones)
            release_tok = release_tok * modality_mask.get("release", ones)

        user_tok = user_tok.unsqueeze(1)                                     # (B,1,D)
        modality_seq = torch.stack([audio_tok, cf_tok, tag_tok, release_tok], dim=1)  # (B,4,D)

        # [CLS, USER_CF, query_rest, AUDIO, CF, TAG, RELEASE, doc_text]
        cls = q_embeds[:, :1, :]
        q_rest = q_embeds[:, 1:, :]
        fused = torch.cat([cls, user_tok, q_rest, modality_seq, d_embeds], dim=1)

        am = query_attention_mask
        one = torch.ones(B, 1, device=device, dtype=am.dtype)
        modality_attn = torch.ones(B, 4, device=device, dtype=am.dtype)
        extended_mask = torch.cat(
            [am[:, :1], one, am[:, 1:], modality_attn, doc_attention_mask], dim=1
        )

        out = self._backbone_forward(fused, extended_mask)
        cls_out = out.last_hidden_state[:, 0, :]      # CLS pooling
        return self.score_head(cls_out).squeeze(-1)   # (B,)

    # ------------------------------------------------------------- save/load
    def save_pretrained(self, out_dir: str, save_tokenizer: bool = True) -> None:
        """Save backbone + modality/scoring heads + config (+ tokenizer).

        Stage B is a full fine-tune, so the backbone is saved directly (no
        LoRA merge). Layout mirrors Stage A's ``MultiModalBiEncoder``:
        ``out_dir/backbone/``, ``out_dir/modality_heads.pt``,
        ``out_dir/multimodal_config.json``.
        """
        os.makedirs(out_dir, exist_ok=True)
        backbone_dir = os.path.join(out_dir, "backbone")
        if hasattr(self.backbone, "save_pretrained"):
            self.backbone.save_pretrained(backbone_dir)
        else:
            # Non-HF backbone (e.g. a test stub): persist its state dict so
            # from_pretrained can restore it when no backbone is injected.
            os.makedirs(backbone_dir, exist_ok=True)
            torch.save(self.backbone.state_dict(),
                       os.path.join(backbone_dir, "backbone_state.pt"))
        if save_tokenizer:
            # I5 (Stage A): persist tokenizer alongside backbone so
            # from_pretrained needn't re-fetch upstream. Best-effort.
            try:
                from transformers import AutoTokenizer
                _tok = AutoTokenizer.from_pretrained(self.config.backbone_name)
                _tok.save_pretrained(backbone_dir)
            except Exception as _e:
                print(f"[MultiModalCrossEncoder.save_pretrained] WARN: tokenizer "
                      f"save skipped ({_e}); from_pretrained will fall back to "
                      f"upstream Hub for {self.config.backbone_name!r}")
        heads = {
            "audio_proj": self.audio_proj.state_dict(),
            "cf_track_proj": self.cf_track_proj.state_dict(),
            "cf_user_proj": self.cf_user_proj.state_dict(),
            "tag_embed": self.tag_embed.state_dict(),
            "release_proj": self.release_proj.state_dict(),
            "score_head": self.score_head.state_dict(),
        }
        torch.save(heads, os.path.join(out_dir, "modality_heads.pt"))
        cfg_dict = asdict(self.config)
        cfg_dict["lora_targets"] = list(cfg_dict["lora_targets"])  # JSON-safe
        with open(os.path.join(out_dir, "multimodal_config.json"), "w") as f:
            json.dump(cfg_dict, f, indent=2)

    @classmethod
    def from_pretrained(
        cls,
        in_dir: str,
        backbone: Optional[nn.Module] = None,
        device: Optional[str] = None,
    ) -> "MultiModalCrossEncoder":
        """Reconstruct from ``save_pretrained`` output. Hub-aware.

        Args:
            in_dir: local dir written by ``save_pretrained``, OR a HF Hub repo
                id (auto snapshot-downloaded on first load).
            backbone: optional pre-built backbone instance to use directly,
                skipping the disk/Hub backbone load (used by tests and by
                callers that already hold the base model).
            device: optional device to move the model to after load.
        """
        if not os.path.isdir(in_dir):
            try:
                from huggingface_hub import snapshot_download
                print(f"[MultiModalCrossEncoder.from_pretrained] {in_dir!r} is not "
                      f"a local dir; downloading from HF Hub...")
                in_dir = snapshot_download(repo_id=in_dir)
            except Exception as e:
                raise FileNotFoundError(
                    f"MultiModalCrossEncoder.from_pretrained: {in_dir!r} is "
                    f"neither a local directory nor a downloadable HF Hub repo "
                    f"({type(e).__name__}: {e})."
                ) from e

        with open(os.path.join(in_dir, "multimodal_config.json"), "r") as f:
            cfg_dict = json.load(f)
        cfg_dict["lora_targets"] = tuple(cfg_dict.get("lora_targets") or DEFAULT_LORA_TARGETS)
        config = MultiModalConfig(**cfg_dict)

        if backbone is None:
            from transformers import AutoModel
            backbone = AutoModel.from_pretrained(os.path.join(in_dir, "backbone"))

        # __init__ never LoRA-wraps (full FT), so config passes through as-is.
        model = cls(config, backbone=backbone)

        heads_state = torch.load(
            os.path.join(in_dir, "modality_heads.pt"), map_location="cpu",
        )
        model.audio_proj.load_state_dict(heads_state["audio_proj"])
        model.cf_track_proj.load_state_dict(heads_state["cf_track_proj"])
        model.cf_user_proj.load_state_dict(heads_state["cf_user_proj"])
        model.tag_embed.load_state_dict(heads_state["tag_embed"])
        model.release_proj.load_state_dict(heads_state["release_proj"])
        model.score_head.load_state_dict(heads_state["score_head"])

        if device:
            model = model.to(device)
        return model
