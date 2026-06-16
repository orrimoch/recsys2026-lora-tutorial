"""Multi-modal bi-encoder for RecSys 2026 Music CRS — Stage A of the
``fresh-model`` branch.

Architecture
------------

Two towers, both output 768-d L2-normalized embeddings, scored by dot
product at inference time. Modalities are fused INSIDE the encoder via
token-level cross-attention ("fusion-in-encoder"), not by RRF over
separate retrievers. Each modality becomes a learned 768-d pseudo-token
spliced into the input embedding sequence; self-attention then blends
it with text through every transformer layer.

Track tower input sequence (positions 0..L+3)::

    [CLS, text_1, ..., text_{L-1}, AUDIO, CF, TAG, RELEASE]

Query tower input sequence (positions 0..L)::

    [CLS, USER_CF, text_1, ..., text_{L-1}]

``USER_CF`` is prepended (not appended) so user identity shapes attention
from layer 1. Cold users get the train-set mean CF vector as a fallback
— NOT a zero vector, which would be a distribution the model has never
seen.

Modalities
----------

- **CLAP audio** (512-d, per track): from
  ``talkpl-ai/TalkPlayData-Challenge-Track-Embeddings[audio-laion_clap]``.
- **CF-BPR track** (128-d, per track): from
  ``talkpl-ai/TalkPlayData-Challenge-Track-Embeddings[cf-bpr]``.
- **CF-BPR user** (128-d, per user): from
  ``talkpl-ai/TalkPlayData-Challenge-User-Embeddings[cf-bpr]``.
- **Tag embedding** (learned, ``tag_vocab_size`` entries): mean-pooled
  over a track's normalized tag list (pad-id 0 masked out).
- **Release year** (scalar → 768-d): sinusoidal-style normalized scalar
  fed through a linear projection.

The modality projection heads train FULLY (no LoRA) — they're tiny (~5M
params total) and need rank-1 freedom to specialize from random init.
The backbone is wrapped in LoRA at the standard rank=128.

Save / load
-----------

``save_pretrained(out_dir)`` writes:

  out_dir/
    backbone/                  # peft-adapter directory OR merged weights
    modality_heads.pt          # state_dict of all modality projections
    multimodal_config.json     # hyperparameters

``from_pretrained(in_dir, base_model=...)`` reconstructs the model. For
inference we additionally provide ``merge_and_unload()`` to fold LoRA
into the backbone (matches the pattern used by ``train_bi_encoder.py``'s
text-only path for the merged Hub push).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_BACKBONE = "BAAI/bge-base-en-v1.5"
DEFAULT_LORA_TARGETS = ("query", "key", "value", "dense")
DEFAULT_AUDIO_DIM = 512   # LAION-CLAP
DEFAULT_CF_DIM = 128      # CF-BPR (user + track share dim)
DEFAULT_HIDDEN_DIM = 768  # bge-base-en output


@dataclass
class MultiModalConfig:
    backbone_name: str = DEFAULT_BACKBONE
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    audio_dim: int = DEFAULT_AUDIO_DIM
    cf_dim: int = DEFAULT_CF_DIM
    tag_vocab_size: int = 0       # required at construction; saved with the model
    max_tags: int = 20
    lora_rank: int = 128
    lora_alpha: int = 256
    lora_targets: tuple[str, ...] = DEFAULT_LORA_TARGETS
    # Year normalization: scalar (year - year_origin) / year_span, clamped to [-1, 1].
    year_origin: int = 1950
    year_span: int = 100


class MultiModalBiEncoder(nn.Module):
    """Bi-encoder with token-level multi-modal fusion. See module docstring."""

    def __init__(self, config: MultiModalConfig, backbone: Optional[nn.Module] = None):
        super().__init__()
        self.config = config

        if backbone is None:
            from transformers import AutoModel
            backbone = AutoModel.from_pretrained(config.backbone_name)
        # Verify hidden dim matches the configured value (catches silent
        # backbone swaps).
        bb_dim = getattr(backbone.config, "hidden_size", None)
        if bb_dim is None or bb_dim != config.hidden_dim:
            raise ValueError(
                f"backbone hidden_size={bb_dim} doesn't match config.hidden_dim="
                f"{config.hidden_dim}. Update MultiModalConfig.hidden_dim or pass a "
                f"different backbone."
            )

        # Optional LoRA wrap. peft is a training-only dep; keep import local.
        if config.lora_rank > 0:
            from peft import LoraConfig, TaskType, get_peft_model
            lora_cfg = LoraConfig(
                r=config.lora_rank, lora_alpha=config.lora_alpha,
                target_modules=list(config.lora_targets),
                lora_dropout=0.0, bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            backbone = get_peft_model(backbone, lora_cfg)
        self.backbone = backbone

        D = config.hidden_dim
        # Modality projections — fully trained, no LoRA. Tiny (~5M params).
        self.audio_proj = nn.Sequential(nn.Linear(config.audio_dim, D), nn.LayerNorm(D))
        self.cf_track_proj = nn.Sequential(nn.Linear(config.cf_dim, D), nn.LayerNorm(D))
        self.cf_user_proj  = nn.Sequential(nn.Linear(config.cf_dim, D), nn.LayerNorm(D))
        # tag_vocab_size required > 0 at construction. pad_id=0 reserved per
        # scripts/precompute_multimodal_artifacts.py.
        if config.tag_vocab_size < 2:
            raise ValueError(
                f"tag_vocab_size must be >= 2 (need pad + at least 1 real tag), "
                f"got {config.tag_vocab_size}. Pass a real vocab built from "
                f"experiments/cache/multimodal/tag_vocab.json."
            )
        self.tag_embed = nn.Embedding(config.tag_vocab_size, D, padding_idx=0)
        # Release year: scalar in [-1, 1] → 768-d via a single Linear.
        self.release_proj = nn.Linear(1, D)
        # Init projections small so they start as a perturbation to the
        # text-only signal — model gradually learns to use them.
        for module in (self.audio_proj[0], self.cf_track_proj[0], self.cf_user_proj[0],
                       self.release_proj):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.tag_embed.weight, std=0.02)
        with torch.no_grad():
            self.tag_embed.weight[0].zero_()  # pad row stays zero

        # Final projection head — identity-initialized so the model
        # starts equivalent to direct CLS pooling.
        self.proj_head = nn.Linear(D, D, bias=False)
        nn.init.eye_(self.proj_head.weight)

    # --------------------------------------------------------------- helpers
    def _input_embeds(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Pull the backbone's input-token embeddings.

        Works whether the backbone is bare ``AutoModel`` or wrapped in
        ``peft`` — both expose ``get_input_embeddings()``.
        """
        return self.backbone.get_input_embeddings()(input_ids)

    def _backbone_forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor):
        """Run the (possibly peft-wrapped) backbone on token embeddings.

        We pass ``inputs_embeds`` instead of ``input_ids`` so we can splice
        modality tokens into the sequence at chosen positions.
        """
        return self.backbone(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

    def _year_to_scalar(self, year: torch.Tensor) -> torch.Tensor:
        """Year-int → scalar in [-1, 1]. Returns shape ``(B, 1)``.

        Mapping:
          - year >= year_origin: linear ``(year - year_origin) / year_span``
            clamped to ``[-1, 1]``. Default origin=1950, span=100 → year=2050
            maps to 1.0; year=2150+ saturates at 1.0.
          - year < 0 (the explicit "unknown" sentinel from the precompute
            artifacts): output 0.0. Model learns a neutral release-token
            for unknown-year tracks.
          - year in [0, year_origin): clamps to year_origin → outputs 0.0.
            **Collides with the unknown sentinel** — pre-1950 tracks
            (negligible in this catalog but worth flagging) get the same
            release token as unknown-year tracks. Fix this by lowering
            year_origin or using a learned unknown-year embedding if the
            catalog ever needs pre-1950 distinction.

        I7 note: the collision is intentional for the current music catalog
        (1950+ exclusively per spot-check); documenting so future-catalog
        changes don't accidentally regress without noticing.
        """
        c = self.config
        # Treat -1 (unknown) as the origin year so the scalar is 0.
        y = year.float().clamp(min=c.year_origin)
        normalized = ((y - c.year_origin) / c.year_span).clamp(-1.0, 1.0)
        # Restore the unknown-year mask: where year was -1, output 0.
        unknown = (year < 0)
        normalized = torch.where(unknown, torch.zeros_like(normalized), normalized)
        return normalized.unsqueeze(-1)  # (B, 1)

    def _tag_token(self, tag_ids: torch.Tensor) -> torch.Tensor:
        """Mean-pool tag embeddings over non-pad positions. Returns (B, D)."""
        # tag_ids: (B, max_tags), pad_id=0
        embs = self.tag_embed(tag_ids)                  # (B, max_tags, D)
        mask = (tag_ids != 0).float().unsqueeze(-1)      # (B, max_tags, 1)
        summed = (embs * mask).sum(dim=1)                # (B, D)
        # Tracks with NO tags get all-zero token (mask sums to 0). Clamp the
        # divisor to avoid NaN; the resulting zero token is fine because the
        # backbone learns to treat it as a no-signal contribution.
        denom = mask.sum(dim=1).clamp(min=1.0)           # (B, 1)
        return summed / denom

    # --------------------------------------------------------- forward_track
    def forward_track(
        self,
        input_ids: torch.Tensor,        # (B, L_text) long
        attention_mask: torch.Tensor,   # (B, L_text) long/bool
        clap: torch.Tensor,             # (B, audio_dim) float
        cf_track: torch.Tensor,         # (B, cf_dim) float
        tag_ids: torch.Tensor,          # (B, max_tags) long, pad=0
        year: torch.Tensor,             # (B,) long (int year or -1)
        modality_mask: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Encode tracks with text + 4 modality tokens spliced after CLS.

        ``modality_mask`` (optional): dict with optional keys ``audio``,
        ``cf``, ``tag``, ``release``. Each value is a (B, 1) float tensor
        in {0., 1.} used to zero out the modality token (for dropout /
        ablation). Missing key = keep the token.

        Returns L2-normalized 768-d embeddings of shape (B, D).
        """
        B = input_ids.shape[0]
        device = input_ids.device

        text_embeds = self._input_embeds(input_ids)  # (B, L, D)

        audio_tok   = self.audio_proj(clap)                          # (B, D)
        cf_tok      = self.cf_track_proj(cf_track)                   # (B, D)
        tag_tok     = self._tag_token(tag_ids)                       # (B, D)
        release_tok = self.release_proj(self._year_to_scalar(year))  # (B, D)

        if modality_mask is not None:
            audio_tok   = audio_tok   * modality_mask.get("audio",   torch.ones(B, 1, device=device))
            cf_tok      = cf_tok      * modality_mask.get("cf",      torch.ones(B, 1, device=device))
            tag_tok     = tag_tok     * modality_mask.get("tag",     torch.ones(B, 1, device=device))
            release_tok = release_tok * modality_mask.get("release", torch.ones(B, 1, device=device))

        # Stack 4 modality tokens: (B, 4, D)
        modality_seq = torch.stack([audio_tok, cf_tok, tag_tok, release_tok], dim=1)

        # Splice: [CLS, text_1..L-1, AUDIO, CF, TAG, RELEASE]
        fused_embeds = torch.cat([text_embeds, modality_seq], dim=1)  # (B, L+4, D)

        # Extend attention mask: 4 ones (modality tokens always attended).
        modality_attn = torch.ones(B, 4, device=device, dtype=attention_mask.dtype)
        extended_mask = torch.cat([attention_mask, modality_attn], dim=1)

        out = self._backbone_forward(fused_embeds, extended_mask)
        cls = out.last_hidden_state[:, 0, :]  # CLS pooling
        cls = self.proj_head(cls)
        return F.normalize(cls, dim=-1)

    # --------------------------------------------------------- forward_query
    def forward_query(
        self,
        input_ids: torch.Tensor,        # (B, L_text)
        attention_mask: torch.Tensor,   # (B, L_text)
        user_cf: torch.Tensor,          # (B, cf_dim) — mean fallback for cold
        modality_mask: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Encode queries with text + 1 user_cf token prepended after CLS.

        Cold users: caller supplies the train-set mean CF vector via
        ``user_cf`` — model never sees a zero vector unless explicitly
        ablated via ``modality_mask``.

        Returns L2-normalized 768-d embeddings of shape (B, D).
        """
        B = input_ids.shape[0]
        device = input_ids.device

        text_embeds = self._input_embeds(input_ids)  # (B, L, D)
        user_tok = self.cf_user_proj(user_cf)        # (B, D)

        if modality_mask is not None:
            user_tok = user_tok * modality_mask.get("user_cf", torch.ones(B, 1, device=device))

        # Splice: [CLS, USER_CF, text_1..L-1]
        cls_embed = text_embeds[:, :1, :]                            # (B, 1, D)
        text_rest = text_embeds[:, 1:, :]                            # (B, L-1, D)
        fused_embeds = torch.cat([cls_embed, user_tok.unsqueeze(1), text_rest], dim=1)  # (B, L+1, D)

        # Extend attention mask: 1 one for USER_CF, inserted after CLS.
        usercf_attn = torch.ones(B, 1, device=device, dtype=attention_mask.dtype)
        extended_mask = torch.cat([attention_mask[:, :1], usercf_attn, attention_mask[:, 1:]], dim=1)

        out = self._backbone_forward(fused_embeds, extended_mask)
        cls = out.last_hidden_state[:, 0, :]
        cls = self.proj_head(cls)
        return F.normalize(cls, dim=-1)

    # ------------------------------------------------------------- save/load
    def save_pretrained(self, out_dir: str, merge_lora: bool = False,
                          save_tokenizer: bool = True) -> None:
        """Save backbone + modality heads + config (+ tokenizer).

        Args:
            out_dir: target directory.
            merge_lora: ``True`` folds LoRA into the base weights and saves a
                plain ``AutoModel`` (suitable for ``DENSE_MULTIMODAL_LOCAL``
                inference). ``False`` saves the peft adapter (resumeable
                training).
            save_tokenizer: ``True`` (default) also saves the upstream
                tokenizer into ``out_dir/backbone/``. I5 fix: prevents silent
                drift if the upstream Hub tokenizer changes between training
                and inference. Set ``False`` if the caller will save the
                tokenizer separately (e.g., the train script's checkpoint
                path already calls tokenizer.save_pretrained next to the
                model dir, not inside backbone/).
        """
        os.makedirs(out_dir, exist_ok=True)
        backbone_dir = os.path.join(out_dir, "backbone")
        if merge_lora and hasattr(self.backbone, "merge_and_unload"):
            merged = self.backbone.merge_and_unload()
            merged.save_pretrained(backbone_dir)
        else:
            self.backbone.save_pretrained(backbone_dir)
        if save_tokenizer:
            # I5 fix: persist the tokenizer alongside the backbone so
            # from_pretrained doesn't have to re-fetch from upstream Hub.
            # Lazy import: AutoTokenizer is heavy.
            try:
                from transformers import AutoTokenizer
                _tok = AutoTokenizer.from_pretrained(self.config.backbone_name)
                _tok.save_pretrained(backbone_dir)
            except Exception as _e:
                # Best-effort: don't break the save if tokenizer fetch fails
                # (e.g., offline mode). Surface the issue but continue.
                print(f"[MultiModalBiEncoder.save_pretrained] WARN: tokenizer "
                      f"save skipped ({_e}); from_pretrained will fall back to "
                      f"upstream Hub for {self.config.backbone_name!r}")
        heads = {
            "audio_proj": self.audio_proj.state_dict(),
            "cf_track_proj": self.cf_track_proj.state_dict(),
            "cf_user_proj": self.cf_user_proj.state_dict(),
            "tag_embed": self.tag_embed.state_dict(),
            "release_proj": self.release_proj.state_dict(),
            "proj_head": self.proj_head.state_dict(),
        }
        torch.save(heads, os.path.join(out_dir, "modality_heads.pt"))
        cfg_dict = asdict(self.config)
        # tuples aren't JSON-native; convert to list for round-trip.
        cfg_dict["lora_targets"] = list(cfg_dict["lora_targets"])
        with open(os.path.join(out_dir, "multimodal_config.json"), "w") as f:
            json.dump(cfg_dict, f, indent=2)

    @classmethod
    def from_pretrained(
        cls,
        in_dir: str,
        backbone_override: Optional[str] = None,
        device: Optional[str] = None,
        is_trainable: bool = False,
    ) -> "MultiModalBiEncoder":
        """Reconstruct the model from ``save_pretrained`` output.

        Args:
            in_dir: path to the directory written by ``save_pretrained``.
            backbone_override: load the backbone from a different path/Hub
                repo than the one recorded in config (useful for swapping to
                a fresh base before applying our LoRA adapter).
            device: optional device to move the model to after load.
            is_trainable: when ``True``, loaded PEFT adapters are returned
                with ``requires_grad=True`` so training can continue
                (warm-start). When ``False`` (default), adapters load in
                inference mode — fine for embedding catalog / scoring
                queries, but wrong for ``--resume-from`` because the LoRA
                params would be silently frozen and only modality heads
                would receive gradient updates. C1 fix.

        Accepts EITHER a local filesystem path OR a HuggingFace Hub repo
        id. Hub repos are auto-downloaded via ``huggingface_hub.snapshot_
        download`` on first load (subsequent loads hit the local HF cache).
        """
        # Hub-aware: if `in_dir` isn't a local directory, treat it as a
        # Hub repo id and snapshot_download it first. Matches the
        # AutoModel.from_pretrained / SentenceTransformer behavior so
        # nb 70 cell 6's Hub-fallback path (TRAIN_OUTPUT_DIR/merged
        # cleared by --cleanup-after-push) works.
        if not os.path.isdir(in_dir):
            try:
                from huggingface_hub import snapshot_download
                print(f"[MultiModalBiEncoder.from_pretrained] {in_dir!r} is not a "
                      f"local dir; downloading from HF Hub...")
                in_dir = snapshot_download(repo_id=in_dir)
                print(f"[MultiModalBiEncoder.from_pretrained] downloaded to {in_dir}")
            except Exception as e:
                raise FileNotFoundError(
                    f"MultiModalBiEncoder.from_pretrained: {in_dir!r} is "
                    f"neither a local directory nor a downloadable HF Hub "
                    f"repo ({type(e).__name__}: {e})."
                ) from e

        with open(os.path.join(in_dir, "multimodal_config.json"), "r") as f:
            cfg_dict = json.load(f)
        cfg_dict["lora_targets"] = tuple(cfg_dict.get("lora_targets") or DEFAULT_LORA_TARGETS)
        config = MultiModalConfig(**cfg_dict)

        from transformers import AutoModel
        backbone_path = backbone_override or os.path.join(in_dir, "backbone")
        # If the saved backbone is a peft adapter dir, we need the base model
        # first then attach the adapter. If it's a merged AutoModel, load
        # directly with lora_rank=0 to skip the wrapping.
        if os.path.isfile(os.path.join(backbone_path, "adapter_config.json")):
            # PEFT adapter — load base + apply adapter
            from peft import PeftModel
            base = AutoModel.from_pretrained(config.backbone_name)
            backbone = PeftModel.from_pretrained(
                base, backbone_path, is_trainable=is_trainable,
            )
            # Skip the LoRA wrap in __init__ — we've already attached it.
            config_no_lora = MultiModalConfig(**{**asdict(config), "lora_rank": 0})
            # Manually re-tuple the targets after the dict round-trip.
            config_no_lora.lora_targets = config.lora_targets
            model = cls(config_no_lora, backbone=backbone)
            model.config = config  # restore the real config for save round-trips
        else:
            # Merged base — load fresh, no LoRA wrap.
            base = AutoModel.from_pretrained(backbone_path)
            config_no_lora = MultiModalConfig(**{**asdict(config), "lora_rank": 0})
            config_no_lora.lora_targets = config.lora_targets
            model = cls(config_no_lora, backbone=base)
            model.config = config

        heads_state = torch.load(
            os.path.join(in_dir, "modality_heads.pt"),
            map_location="cpu",
        )
        model.audio_proj.load_state_dict(heads_state["audio_proj"])
        model.cf_track_proj.load_state_dict(heads_state["cf_track_proj"])
        model.cf_user_proj.load_state_dict(heads_state["cf_user_proj"])
        model.tag_embed.load_state_dict(heads_state["tag_embed"])
        model.release_proj.load_state_dict(heads_state["release_proj"])
        model.proj_head.load_state_dict(heads_state["proj_head"])

        if device:
            model = model.to(device)
        return model
