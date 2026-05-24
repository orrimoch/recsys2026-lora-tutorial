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


def score_ce_batch(model, tokenizer, rows, max_length: int = 512,
                   max_tags: int = 20, device: str = "cpu"):
    """Forward one batch of triples → ``(logits, is_positive)``.

    Flattens the batch's ``TripleJsonlDataset`` rows into (query, doc) pairs,
    tokenizes both sides, builds the modality tensors, and scores every pair
    with the ``MultiModalCrossEncoder``. Returns the raw scoring-head ``logits``
    (N,) and the aligned ``is_positive`` bool mask (N,). Shared by
    ``compute_batch_loss`` and the train/val ranking metrics so a step needs
    only ONE forward pass. ``tokenizer`` / ``model`` are injected so the step is
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
    return logits, is_pos


def listwise_softmax_loss(logits, is_positive, temperature: float = 1.0):
    """Listwise softmax cross-entropy (LCE) for the Stage B reranker.

    For each per-query group ``[pos, neg_1, ..., neg_K]`` (every ``True`` in
    ``is_positive`` starts a group, matching ``flatten_ce_pairs`` order), take a
    softmax over the group's scores ``/temperature`` and apply cross-entropy
    with the positive (local index 0) as the target:
    ``-log( exp(s_pos/T) / sum_j exp(s_j/T) )``.

    Optimizes RELATIVE ranking (positive ranked first) rather than absolute
    calibration, so unlike ``pairwise_bce_loss`` it descends from
    ``ln(group_size)`` toward 0 as the positive's margin grows instead of
    flooring at ``2*ln2`` under noisy labels — and it directly targets the
    nDCG/top1 the reranker is judged on. ``temperature`` (T) sharpens (<1) or
    softens (>1) the distribution. Averaged over groups. Pure tensor op.
    """
    import torch
    import torch.nn.functional as F

    pos_idx = torch.nonzero(is_positive, as_tuple=False).flatten().tolist()
    if not pos_idx:
        return logits.sum() * 0.0  # no positives -> 0 (keeps graph + device)
    scaled = logits / temperature
    bounds = pos_idx + [logits.shape[0]]
    target = torch.zeros(1, dtype=torch.long, device=logits.device)
    losses = []
    for gi, start in enumerate(pos_idx):
        group = scaled[start:bounds[gi + 1]]  # positive at local index 0
        losses.append(F.cross_entropy(group.unsqueeze(0), target))
    return torch.stack(losses).mean()


def resolve_max_norm(max_grad_norm: float) -> float:
    """Map the ``--max-grad-norm`` flag to a ``clip_grad_norm_`` max_norm.

    ``0`` (or negative) -> ``inf`` (measure the grad norm only, clip nothing —
    the original behavior). ``>0`` -> that value, so the global grad norm is
    clipped to it. Either way ``clip_grad_norm_`` returns the PRE-clip norm, so
    the logged ``grad_norm`` still surfaces spikes even when clipping is on
    (e.g. a 230 spike shows as 230 in the log but the applied step is capped).
    """
    return max_grad_norm if max_grad_norm and max_grad_norm > 0 else float("inf")


def compute_ce_loss(logits, is_positive, loss_type: str = "bce",
                    temperature: float = 1.0):
    """Dispatch the Stage B loss by name.

    ``bce``     -> ``pairwise_bce_loss`` (per-pair calibration; v1 default).
    ``softmax`` -> ``listwise_softmax_loss`` (in-group ranking; LCE).
    Shared by ``compute_batch_loss`` and the train/val loops so both honor the
    ``--loss`` flag.
    """
    if loss_type == "bce":
        return pairwise_bce_loss(logits[is_positive], logits[~is_positive])
    if loss_type == "softmax":
        return listwise_softmax_loss(logits, is_positive, temperature=temperature)
    raise ValueError(f"unknown loss_type {loss_type!r} (expected 'bce'|'softmax')")


def compute_batch_loss(model, tokenizer, rows, max_length: int = 512,
                       max_tags: int = 20, device: str = "cpu",
                       loss_type: str = "bce", temperature: float = 1.0):
    """One Stage B training step → a single differentiable loss scalar.

    Scores the batch via ``score_ce_batch`` then applies the chosen loss
    (``compute_ce_loss``). Teacher-score weights are absent in v1
    (single-positive), so the BCE negative term is unweighted.
    """
    logits, is_pos = score_ce_batch(
        model, tokenizer, rows, max_length=max_length, max_tags=max_tags,
        device=device,
    )
    return compute_ce_loss(logits, is_pos, loss_type=loss_type,
                           temperature=temperature)


def ce_group_metrics(logits, is_positive):
    """In-group ranking metrics → ``(top1_accuracy, mean_nDCG)`` as floats.

    ``logits`` (N,) and ``is_positive`` (N,) come from ``flatten_ce_pairs``
    order: each row contributes ``[positive, neg_1, ..., neg_K]``, so every
    ``True`` in ``is_positive`` starts a new per-query group. Mirrors the
    bi-encoder's ``_val_metrics_from_scores`` exactly so the train/val curves
    are comparable across both stages:
      - rank of the positive = 1 + (# candidates scoring strictly higher),
        so a tie does NOT outrank the positive (matches argmax==0 semantics);
      - nDCG (single relevant item) = 1 / log2(rank + 1), ideal 1.0 at rank 1;
      - top1 = fraction of groups whose positive is ranked first.
    Averaged over groups. Returns (0.0, 0.0) when there are no positives.
    """
    import math

    import torch

    pos_idx = torch.nonzero(is_positive, as_tuple=False).flatten().tolist()
    if not pos_idx:
        return 0.0, 0.0
    bounds = pos_idx + [len(logits)]
    top1_sum = 0.0
    ndcg_sum = 0.0
    for gi, start in enumerate(pos_idx):
        group = logits[start:bounds[gi + 1]]
        pos_score = logits[start]
        rank = float((group > pos_score).sum().item()) + 1.0
        top1_sum += 1.0 if rank == 1.0 else 0.0
        ndcg_sum += 1.0 / math.log2(rank + 1.0)
    n = len(pos_idx)
    return top1_sum / n, ndcg_sum / n


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
    parser.add_argument("--loss", default="bce", choices=["bce", "softmax"],
                        help="Training loss. 'bce' = per-pair calibration (v1 default; "
                             "floors at 2*ln2 under noisy labels). 'softmax' = in-group "
                             "listwise cross-entropy (LCE) — optimizes ranking directly "
                             "and descends from ln(1+n_negatives) toward 0.")
    parser.add_argument("--loss-temperature", type=float, default=1.0,
                        help="Softmax-loss temperature (only used when --loss softmax). "
                             "<1 sharpens, >1 softens the in-group distribution.")
    parser.add_argument("--max-grad-norm", type=float, default=0.0,
                        help="Clip the global gradient norm to this value (0 = off, "
                             "measure-only — the original behavior). Set ~25 to neutralize "
                             "rare spikes (e.g. softmax-loss bursts to 200+) without "
                             "throttling the productive ~20 gradients. Logged grad_norm "
                             "stays the PRE-clip value so spikes remain visible.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Rows per step; each row expands to 1 pos + n-negatives pairs.")
    parser.add_argument("--n-negatives", type=int, default=7,
                        help="Negatives sampled per row (pairs/row = n_negatives + 1).")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-tags", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=25,
                        help="Log train loss + in-group top1/nDCG every N opt-steps "
                             "(stderr + TensorBoard at {output_dir}/runs).")
    parser.add_argument("--val-fraction", type=float, default=0.05,
                        help="Hold this fraction of the triples out for validation "
                             "(0 = no val pass). Split is leak-safe via --split-key, "
                             "mirroring the Stage A bi-encoder.")
    parser.add_argument("--split-key", default="user_id",
                        choices=["user_id", "session_id", "row"],
                        help="Train/val partition key. 'user_id' (default) keeps every "
                             "session of a user on one side; 'row' = no grouping.")
    parser.add_argument("--val-every-n-steps", type=int, default=500,
                        help="Run a val pass (loss + top1/nDCG) every N opt-steps. A CE "
                             "val pass is a full cross-encoder forward, so this is gated "
                             "and bounded by --val-max-rows.")
    parser.add_argument("--val-max-rows", type=int, default=1000,
                        help="Cap rows scored per val pass so each pass stays ~1 min "
                             "(0 = use the whole held-out val set).")
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

    # --- data: reuse Stage A dataset (single-positive v1: teacher_scores=None).
    # Carve a leak-safe val split off the triples so we can log val loss + nDCG
    # during training, exactly like the Stage A bi-encoder. val_fraction=0 (or a
    # split that yields an empty val set) → train on everything, no val pass.
    _use_val = args.val_fraction > 0
    dataset = TripleJsonlDataset(
        args.triples, n_negatives=args.n_negatives, seed=args.seed,
        split="train" if _use_val else "all", val_fraction=args.val_fraction,
        split_key=args.split_key, artifacts=artifacts, teacher_scores=None,
    )
    # collate_fn=identity: score_ce_batch consumes the raw list of row dicts.
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=lambda rows: rows)
    val_loader = None
    if _use_val:
        val_dataset = TripleJsonlDataset(
            args.triples, n_negatives=args.n_negatives, seed=args.seed,
            split="val", val_fraction=args.val_fraction,
            split_key=args.split_key, artifacts=artifacts, teacher_scores=None,
        )
        if len(val_dataset) > 0:
            # shuffle=False → the val set is a fixed, deterministic slice so the
            # val curve is apples-to-apples across opt-steps.
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                    shuffle=False, collate_fn=lambda rows: rows)
        else:
            _use_val = False
    print(f"[train-ce] {len(dataset)} train rows"
          + (f" / {len(val_dataset)} val rows (split_key={args.split_key})"
             if val_loader is not None else " (no val split)")
          + f", bs={args.batch_size}, {args.n_negatives} negs/row, "
          + f"loss={args.loss}"
          + (f"(T={args.loss_temperature})" if args.loss == "softmax" else "")
          + (f", clip@{args.max_grad_norm}" if args.max_grad_norm and args.max_grad_norm > 0
             else ", clip=off")
          + f", device={device}",
          file=sys.stderr)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    use_amp = device == "cuda"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # TensorBoard: same channel as the bi-encoder (open {output_dir}/runs in nb 71's
    # launcher cell, in parallel with training). Falls back to stderr-only if the
    # writer can't be created.
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(Path(args.output_dir) / "runs"))
    except Exception as e:  # pragma: no cover - TB optional
        writer = None
        print(f"[train-ce] TensorBoard unavailable ({e}); stderr metrics only",
              file=sys.stderr)

    def _log_scalar(tag: str, value: float, global_step: int) -> None:
        if writer is not None:
            writer.add_scalar(tag, value, global_step)

    def _val_now():
        """Forward-only pass over up to --val-max-rows of the held-out val set.
        Returns (mean_loss, top1, mean_nDCG) — the same triple the bi-encoder's
        _val_loss_now reports. Restores model.train() on exit."""
        if val_loader is None:
            return None
        # Re-seed the val dataset's negative sampler so every pass scores the
        # SAME fixed candidate sets — otherwise the per-__getitem__ RNG advances
        # and each pass would draw different negatives, adding sampling noise to
        # the val curve. With a fixed seed the model is the only variable, so
        # val_loss/nDCG are directly comparable across opt-steps.
        val_loader.dataset.rng.seed(args.seed)
        model.eval()
        loss_sum = 0.0
        n_batches = 0
        t1_sum = 0.0
        ndcg_sum = 0.0
        groups = 0
        seen = 0
        with torch.no_grad():
            for vrows in val_loader:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                    enabled=use_amp):
                    vlogits, vis_pos = score_ce_batch(
                        model, tokenizer, vrows, max_length=args.max_length,
                        max_tags=args.max_tags, device=device,
                    )
                    vloss = compute_ce_loss(vlogits, vis_pos, loss_type=args.loss,
                                            temperature=args.loss_temperature)
                loss_sum += float(vloss.item())
                n_batches += 1
                ng = int(vis_pos.sum().item())  # one positive per row → groups in batch
                bt1, bndcg = ce_group_metrics(vlogits.float(), vis_pos)
                t1_sum += bt1 * ng       # micro-average over groups, not batches
                ndcg_sum += bndcg * ng
                groups += ng
                seen += len(vrows)
                if args.val_max_rows and seen >= args.val_max_rows:
                    break
        model.train()
        if n_batches == 0 or groups == 0:
            return None
        return loss_sum / n_batches, t1_sum / groups, ndcg_sum / groups

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
    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        for rows in loader:
            # One forward → reused for the loss (backprop) AND the train metrics,
            # so logging adds no extra forward pass.
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits, is_pos = score_ce_batch(
                    model, tokenizer, rows, max_length=args.max_length,
                    max_tags=args.max_tags, device=device,
                )
                loss = compute_ce_loss(logits, is_pos, loss_type=args.loss,
                                       temperature=args.loss_temperature)
            loss.backward()
            # Grad-norm logged BEFORE the optimizer step. clip_grad_norm_ returns
            # the PRE-clip norm (so spikes stay visible) and, when --max-grad-norm>0,
            # also scales the grads down to that norm; 0 -> inf -> measure-only.
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=resolve_max_norm(args.max_grad_norm)))
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            last_loss = float(loss.item())
            if step % args.log_every == 0:
                with torch.no_grad():
                    t_top1, t_ndcg = ce_group_metrics(logits.detach().float(), is_pos)
                _log_scalar("train/loss", last_loss, step)
                _log_scalar("train/lr", optimizer.param_groups[0]["lr"], step)
                _log_scalar("train/grad_norm", grad_norm, step)
                _log_scalar("train/top1_ingroup", t_top1, step)
                _log_scalar("train/ndcg_ingroup", t_ndcg, step)
                print(f"[train-ce] epoch={epoch} step={step} loss={last_loss:.4f} "
                      f"train_top1={t_top1:.3f} train_ndcg={t_ndcg:.4f} "
                      f"grad_norm={grad_norm:.3f}", file=sys.stderr)
            # Periodic val pass (loss + top1/nDCG) — fires after the optimizer step
            # so it reflects the latest update, mirroring the bi-encoder.
            if val_loader is not None and args.val_every_n_steps > 0 \
                    and step % args.val_every_n_steps == 0:
                vres = _val_now()
                if vres is not None:
                    vl, vt1, vndcg = vres
                    _log_scalar("val/loss", vl, step)
                    _log_scalar("val/top1_acc", vt1, step)
                    _log_scalar("val/ndcg", vndcg, step)
                    improved = vl < best_val_loss
                    if improved:
                        best_val_loss = vl
                    print(f"[train-ce] epoch={epoch} step={step} val_loss={vl:.4f} "
                          f"val_top1={vt1:.3f} val_ndcg={vndcg:.4f}"
                          f"{' (new best)' if improved else ''}", file=sys.stderr)
            if args.checkpoint_every_n_steps and step % args.checkpoint_every_n_steps == 0:
                _save_checkpoint("checkpoint_latest", epoch, step, last_loss)
        # Per-epoch checkpoint (resumable; survives disconnect when on Drive).
        _save_checkpoint(f"checkpoint_epoch_{epoch + 1}", epoch, step, last_loss)

    # Final val pass (logged at the last step so it lands on the same x-axis as
    # the periodic val curve), in case --val-every-n-steps missed the last step.
    if val_loader is not None:
        vres = _val_now()
        if vres is not None:
            vl, vt1, vndcg = vres
            _log_scalar("val/loss", vl, step)
            _log_scalar("val/top1_acc", vt1, step)
            _log_scalar("val/ndcg", vndcg, step)
            best_val_loss = min(best_val_loss, vl)
            print(f"[train-ce] FINAL step={step} val_loss={vl:.4f} "
                  f"val_top1={vt1:.3f} val_ndcg={vndcg:.4f} "
                  f"(best_loss={best_val_loss:.4f})", file=sys.stderr)
    if writer is not None:
        writer.close()

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
