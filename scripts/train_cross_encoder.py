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
    import torch

    flat = flatten_ce_pairs(rows)
    q_enc = tokenizer(flat["query"], padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt")
    d_enc = tokenizer(flat["doc_text"], padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt")

    logits = model(
        query_input_ids=q_enc["input_ids"].to(device),
        query_attention_mask=q_enc["attention_mask"].to(device),
        query_user_cf=torch.tensor(flat["user_cf"], dtype=torch.float32, device=device),
        doc_input_ids=d_enc["input_ids"].to(device),
        doc_attention_mask=d_enc["attention_mask"].to(device),
        doc_clap=torch.tensor(flat["doc_clap"], dtype=torch.float32, device=device),
        doc_cf=torch.tensor(flat["doc_cf"], dtype=torch.float32, device=device),
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
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModel, AutoTokenizer

    # Reuse Stage A's dataset + artifacts + the cross-encoder model class.
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "music-crs-baselines"))
    sys.path.insert(0, str(repo_root / "scripts"))
    from mcrs.training.multimodal_bi_encoder import MultiModalArtifacts, MultiModalConfig
    from mcrs.training.multimodal_cross_encoder import MultiModalCrossEncoder
    from train_bi_encoder import TripleJsonlDataset

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- modality artifacts + tag vocab size ---
    artifacts = MultiModalArtifacts(args.multimodal_artifacts)
    with open(os.path.join(args.multimodal_artifacts, "tag_vocab.json")) as f:
        tag_vocab_size = len(json.load(f))

    # --- model: full FT (lora_rank=0). Load the backbone first so the config's
    # hidden_dim matches the reranker's hidden size (1024 for bge-reranker-v2-m3).
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
    model.train()
    step = 0
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
            if step % args.log_every == 0:
                print(f"[train-ce] epoch={epoch} step={step} loss={loss.item():.4f}",
                      file=sys.stderr)

    # --- save (backbone + modality/scoring heads + config) + optional Hub push ---
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    print(f"[train-ce] saved → {args.output_dir}", file=sys.stderr)
    if args.hub_repo:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.hub_repo, repo_type="model", exist_ok=True, private=False)
        api.upload_folder(folder_path=args.output_dir, repo_id=args.hub_repo, repo_type="model")
        print(f"[train-ce] pushed → {args.hub_repo}", file=sys.stderr)


if __name__ == "__main__":
    main()
