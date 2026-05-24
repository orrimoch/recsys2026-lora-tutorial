"""Stage B: train the multi-modal cross-encoder reranker (Phase 6c).

Trains ``MultiModalCrossEncoder`` (BAAI/bge-reranker-v2-m3 backbone, FULL
fine-tune + the shared Stage A modality projections) on the Stage B triples from
``scripts/build_cross_encoder_training_data.py``. Reuses ``train_bi_encoder.py``'s
``TripleJsonlDataset`` + ``MultiModalArtifacts`` for data loading and modality
joins, so the data path stays identical to Stage A.

Objective: pairwise BCE — ``BCE(score(q, pos), 1) + BCE(score(q, neg), 0)`` over
the gold + Stage A hard negatives. v1 is single-positive (teacher rank-weighting
dormant — see the builder's v1 NOTE), so the negative term is unweighted.

Hyperparams (plan §7): full FT, lr 2e-5, bs 8 (~570M backbone), epochs 3,
max_length 512, bf16 autocast on GPU.

Usage:
  python scripts/train_cross_encoder.py \
    --triples experiments/cache/retrieval_v2/triples_reranker_mm.jsonl \
    --multimodal-artifacts experiments/cache/multimodal \
    --output-dir /content/mm_reranker_finetune \
    --hub-repo OrRim123/recsys2026-mm-reranker-v1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def pairwise_bce_loss(pos_logits, neg_logits, neg_weights=None):
    """Pairwise binary cross-entropy for the Stage B reranker.

    ``loss = BCE(score(q, pos), 1) + BCE(score(q, neg), 0)`` — the relevance
    objective from the plan (Phase 6 §6.5). ``pos_logits`` / ``neg_logits`` are
    raw scoring-head outputs (B,) / (N,); targets are applied internally so the
    caller never materializes 1/0 tensors.

    ``neg_weights`` (optional, shape matching ``neg_logits``): per-negative
    weights for the negative term — used for v2 rank-weighting of harder Stage A
    negatives. v1 passes None (unweighted), since teacher scores are dormant.
    """
    import torch
    import torch.nn.functional as F

    pos_loss = F.binary_cross_entropy_with_logits(
        pos_logits, torch.ones_like(pos_logits)
    )
    if neg_logits.numel() == 0:
        return pos_loss  # no negatives in batch — positives-only term (avoid NaN)
    neg_targets = torch.zeros_like(neg_logits)
    if neg_weights is not None:
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_logits, neg_targets, weight=neg_weights
        )
    else:
        neg_loss = F.binary_cross_entropy_with_logits(neg_logits, neg_targets)
    return pos_loss + neg_loss


def flatten_ce_pairs(rows: list) -> dict:
    """Flatten ``TripleJsonlDataset`` rows into per-(query, doc) scoring pairs.

    Each row carries 1 gold positive + K sampled negatives plus aligned
    modality data (from ``train_bi_encoder.py``'s multi-modal ``__getitem__``).
    The cross-encoder scores one (query, doc) pair at a time, so we expand every
    row into K+1 pairs: the gold (``is_positive=True``) first, then each
    negative (``is_positive=False``). ``query`` and ``user_cf`` are repeated
    across a row's pairs so the scoring head sees the same user/query context
    for each candidate. Returns a dict of parallel lists (length = total pairs);
    the trainer splits logits by ``is_positive`` for ``pairwise_bce_loss``.
    """
    out: dict[str, list] = {
        "query": [], "doc_text": [], "doc_clap": [], "doc_cf": [],
        "doc_tags": [], "doc_year": [], "user_cf": [], "is_positive": [],
    }
    for row in rows:
        query = row["query"]
        user_cf = row["user_cf"]
        # Gold positive pair first, then each negative — aligned modality lists.
        pairs = [(
            row["positive"], row["pos_clap"], row["pos_cf_track"],
            row["pos_tag_ids"], row["pos_year"], True,
        )]
        for i, neg_text in enumerate(row.get("negatives", [])):
            pairs.append((
                neg_text, row["neg_clap"][i], row["neg_cf_track"][i],
                row["neg_tag_ids"][i], row["neg_years"][i], False,
            ))
        for doc_text, clap, cf, tags, year, is_pos in pairs:
            out["query"].append(query)
            out["user_cf"].append(user_cf)
            out["doc_text"].append(doc_text)
            out["doc_clap"].append(clap)
            out["doc_cf"].append(cf)
            out["doc_tags"].append(tags)
            out["doc_year"].append(year)
            out["is_positive"].append(is_pos)
    return out


def _pad_tag_ids(tag_id_lists: list, max_tags: int, pad: int = 0) -> list:
    """Pad/truncate variable-length tag-id lists to a rectangular (N, max_tags)."""
    out = []
    for ids in tag_id_lists:
        ids = list(ids)[:max_tags]
        out.append(ids + [pad] * (max_tags - len(ids)))
    return out


def compute_batch_loss(model, tokenizer, rows, max_length: int = 512,
                       max_tags: int = 20, device: str = "cpu"):
    """One Stage B training step → a single differentiable loss scalar.

    Flattens the batch's ``TripleJsonlDataset`` rows into (query, doc) pairs,
    tokenizes the query and doc sides, builds the modality tensors, scores every
    pair with the ``MultiModalCrossEncoder``, then splits the logits by the
    positive/negative flag and applies ``pairwise_bce_loss``.

    Teacher-score weights are absent in v1 (single-positive), so the negative
    term is unweighted. ``tokenizer`` / ``model`` are injected so the step is
    unit-testable with stubs (no 570M reranker download, no GPU).
    """
    import numpy as np
    import torch

    flat = flatten_ce_pairs(rows)
    q_enc = tokenizer(flat["query"], padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt")
    d_enc = tokenizer(flat["doc_text"], padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt")

    def _f32(x):  # list-of-arrays -> contiguous (N, D) tensor (avoids slow torch.tensor path)
        return torch.tensor(np.asarray(x, dtype=np.float32), device=device)

    logits = model(
        query_input_ids=q_enc["input_ids"].to(device),
        query_attention_mask=q_enc["attention_mask"].to(device),
        query_user_cf=_f32(flat["user_cf"]),
        doc_input_ids=d_enc["input_ids"].to(device),
        doc_attention_mask=d_enc["attention_mask"].to(device),
        doc_clap=_f32(flat["doc_clap"]),
        doc_cf=_f32(flat["doc_cf"]),
        doc_tags=torch.tensor(_pad_tag_ids(flat["doc_tags"], max_tags),
                              dtype=torch.long, device=device),
        doc_year=torch.tensor(flat["doc_year"], dtype=torch.long, device=device),
    )
    is_pos = torch.tensor(flat["is_positive"], dtype=torch.bool, device=device)
    return pairwise_bce_loss(logits[is_pos], logits[~is_pos])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--triples", required=True,
                        help="Stage B triples JSONL (build_cross_encoder_training_data.py).")
    parser.add_argument("--multimodal-artifacts", required=True,
                        help="Phase 0 cache dir (tag_vocab.json, track_clap/cf, user_cf*). "
                             "Same dir as nb 70/71 MULTIMODAL_ARTIFACTS.")
    parser.add_argument("--base-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hub-repo", default="",
                        help="If set, push the saved model dir to this repo. Empty = skip.")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Rows per step; each row expands to 1 pos + n-negatives pairs.")
    parser.add_argument("--n-negatives", type=int, default=7,
                        help="Negatives sampled per row (pairs/row = n_negatives + 1).")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-tags", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--checkpoint-every-n-steps", type=int, default=0,
                        help="Save a rolling checkpoint to {output_dir}/checkpoint_latest/ "
                             "every N optimizer steps (0 = off). A single epoch here is "
                             "~15k steps (hours), so step-level checkpoints are the real "
                             "disconnect protection — set output_dir on Drive so they survive.")
    parser.add_argument("--resume-from", default="",
                        help="Path (or Hub repo) of a checkpoint to resume from. Loads the "
                             "model via MultiModalCrossEncoder.from_pretrained; the optimizer "
                             "restarts fresh (--epochs counts epochs to run from here).")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Enable backbone gradient checkpointing: trades ~30%% per-step "
                             "speed for much lower activation memory, so a larger --batch-size "
                             "fits on a memory-constrained GPU (e.g. 16-24 GB).")
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModel, AutoTokenizer

    # Reuse Stage A's dataset + artifacts + the cross-encoder model class.
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "music-crs-baselines"))
    sys.path.insert(0, str(repo_root / "scripts"))
    from mcrs.training.multimodal_bi_encoder import MultiModalConfig
    from mcrs.training.multimodal_cross_encoder import MultiModalCrossEncoder
    # MultiModalArtifacts lives in train_bi_encoder (mirrors Stage A), NOT in the
    # mcrs.training.multimodal_bi_encoder module.
    from train_bi_encoder import MultiModalArtifacts, TripleJsonlDataset

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- modality artifacts + tag vocab size ---
    artifacts = MultiModalArtifacts(args.multimodal_artifacts)
    with open(os.path.join(args.multimodal_artifacts, "tag_vocab.json")) as f:
        tag_vocab_size = len(json.load(f))

    # --- model: full FT (lora_rank=0). Resume from a checkpoint if given (loads
    # backbone + modality/scoring heads + config), else build fresh. Loading the
    # backbone first pins the config's hidden_dim to the reranker's hidden size
    # (1024 for bge-reranker-v2-m3).
    if args.resume_from:
        print(f"[train-ce] RESUME from {args.resume_from} (optimizer restarts fresh)",
              file=sys.stderr)
        model = MultiModalCrossEncoder.from_pretrained(args.resume_from, device=device)
    else:
        backbone = AutoModel.from_pretrained(args.base_model)
        cfg = MultiModalConfig(
            backbone_name=args.base_model,
            hidden_dim=backbone.config.hidden_size,
            audio_dim=artifacts.clap_dim,
            cf_dim=artifacts.cf_dim,
            tag_vocab_size=tag_vocab_size,
            max_tags=args.max_tags,
            lora_rank=0,
        )
        model = MultiModalCrossEncoder(cfg, backbone=backbone).to(device)
    if args.gradient_checkpointing and hasattr(model.backbone, "gradient_checkpointing_enable"):
        model.backbone.gradient_checkpointing_enable()
        if hasattr(model.backbone, "config"):
            model.backbone.config.use_cache = False
        print("[train-ce] gradient checkpointing ON", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # --- data: reuse Stage A dataset (single-positive v1: teacher_scores=None) ---
    dataset = TripleJsonlDataset(
        args.triples, n_negatives=args.n_negatives, seed=args.seed,
        split="all", artifacts=artifacts, teacher_scores=None,
    )
    # collate_fn=identity: compute_batch_loss consumes the raw list of row dicts.
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=lambda rows: rows)
    print(f"[train-ce] {len(dataset)} rows, bs={args.batch_size}, "
          f"{args.n_negatives} negs/row, device={device}", file=sys.stderr)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    use_amp = device == "cuda"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    def _save_checkpoint(tag: str, epoch: int, step: int, loss_val: float) -> None:
        """Save the full model to {output_dir}/{tag}/ + a training_state.json.
        Written to a temp dir then swapped in, so a disconnect mid-save can't
        corrupt an existing checkpoint. {tag}=checkpoint_latest is rolling
        (overwritten) to bound Drive usage; checkpoint_epoch_N is per-epoch."""
        import shutil
        final_dir = os.path.join(args.output_dir, tag)
        tmp_dir = os.path.join(args.output_dir, f".{tag}.tmp")
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
        Path(tmp_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(tmp_dir)
        with open(os.path.join(tmp_dir, "training_state.json"), "w") as f:
            json.dump({"epoch": epoch, "global_step": step, "loss": loss_val}, f)
        if os.path.isdir(final_dir):
            shutil.rmtree(final_dir)
        os.replace(tmp_dir, final_dir)
        print(f"[train-ce] checkpoint -> {final_dir} (epoch={epoch} step={step})",
              file=sys.stderr)

    model.train()
    step = 0
    last_loss = float("nan")
    for epoch in range(args.epochs):
        for rows in loader:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = compute_batch_loss(
                    model, tokenizer, rows, max_length=args.max_length,
                    max_tags=args.max_tags, device=device,
                )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            last_loss = float(loss.item())
            if step % args.log_every == 0:
                print(f"[train-ce] epoch={epoch} step={step} loss={last_loss:.4f}",
                      file=sys.stderr)
            if args.checkpoint_every_n_steps and step % args.checkpoint_every_n_steps == 0:
                _save_checkpoint("checkpoint_latest", epoch, step, last_loss)
        # Per-epoch checkpoint (resumable; survives disconnect when on Drive).
        _save_checkpoint(f"checkpoint_epoch_{epoch + 1}", epoch, step, last_loss)

    # --- final save (backbone + modality/scoring heads + config) + optional Hub push ---
    model.save_pretrained(args.output_dir)
    print(f"[train-ce] saved → {args.output_dir}", file=sys.stderr)
    if args.hub_repo:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.hub_repo, repo_type="model", exist_ok=True, private=False)
        # Exclude the Drive-only checkpoint_* / temp dirs (~1.6 GB each) from the
        # Hub push so the model repo holds only the final model.
        api.upload_folder(
            folder_path=args.output_dir, repo_id=args.hub_repo, repo_type="model",
            ignore_patterns=["checkpoint_*", ".*"],
        )
        print(f"[train-ce] pushed → {args.hub_repo}", file=sys.stderr)


if __name__ == "__main__":
    main()
