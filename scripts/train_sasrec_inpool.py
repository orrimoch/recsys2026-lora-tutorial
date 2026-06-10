"""In-pool contrastive fine-tune of SASRec  (SASRec_Improved_Plan.md).

Warm-start the recall-trained `sasrec_v1`, then fine-tune its dual-encoder to
RANK the gold within the actual recall pool — softmax-CE over each sample's own
SASRec-free pool (bm25 + dense + same_artist top-K), not the full catalog. This
trains the model on the exact serve task (the 18.2% conversion gap) instead of the
recall objective it was first trained on.

Design (locked in the plan):
- Pool = wrrf_union_v1 with use_sasrec=False  -> independent of the model being
  trained -> NO in-sample leak -> NO OOF needed.
- gold NOT in pool -> SKIP the sample (recall miss is unrecoverable by a ranker).
- Context = goal-ful (build_sasrec_context); the SAME format must be used at the
  dev gate (nb74 # 12c-inpool) and at serve (crs_baseline sasrec_context_use_goal).
- Select on dev nDCG@20 (the real metric), gate vs lgbm_relev=0.1652.

Colab GPU. Smoke first: --n-sessions 300 --epochs 1  (validates shapes + the
pool/retrieval path before the full run).

Usage:
  python scripts/train_sasrec_inpool.py --cache-dir <CACHE_DIR> \
      --warm-start sasrec_v1 --out sasrec_inpool_v1 --topk 100 --epochs 3
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "music-crs-baselines"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from datasets import load_dataset  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

# Reuse the proven helpers from train_sasrec (sibling import; no main() runs).
from train_sasrec import load_item_feats, encode_dialogs_cached, CTX_MODEL  # noqa: E402
from mcrs.retrieval_modules.sasrec_model import (  # noqa: E402
    SasrecModel, build_sasrec_context, inpool_loss, inpool_target_index,
    prior_turns)
from mcrs.retrieval_modules import load_retrieval_module  # noqa: E402
from mcrs.crs_baseline import build_retrieval_query  # noqa: E402
from mcrs.db_item import MusicCatalogDB  # noqa: E402

CONV = "talkpl-ai/TalkPlayData-Challenge-Dataset"
ITEM_DB = "talkpl-ai/TalkPlayData-Challenge-Track-Metadata"
CORPUS = ["track_name", "artist_name", "album_name"]


def build_samples(split, item_db, max_seq, n_sessions=None):
    """Per (session, music-turn): the union query (raw_with_goal, for pool recall),
    the goal-ful SASRec context text, prior played tids, and the gold tid.
    Mirrors build_lgbm_features' causal walk + prior_turns parity."""
    rows = list(split)
    if n_sessions:
        rows = rows[:n_sessions]
    queries, ctx_texts, played_tids, gold_tids, hist_dialog = [], [], [], [], []
    for sess in tqdm(rows, desc="samples"):
        df = pd.DataFrame(sess["conversations"])
        goal = ((sess.get("conversation_goal") or {}).get("listener_goal") or "").strip()
        up = sess.get("user_profile") or {}
        for _, music in df[df["role"] == "music"].iterrows():
            tn = int(music["turn_number"])
            gold = music["content"]
            prior = prior_turns(df, tn)
            prior_rows = prior.to_dict("records")
            # union query (matches serve raw_with_goal); music turns -> metadata text
            sm = [{"role": ("assistant" if t["role"] == "music" else t["role"]),
                   "content": (item_db.id_to_metadata(t["content"])
                               if t["role"] == "music" else t["content"])}
                  for t in prior_rows]
            queries.append(build_retrieval_query(sm, mode="raw_with_goal",
                                                 goal_text=goal, user_profile=up))
            ctx_texts.append(build_sasrec_context(prior_rows, goal_text=goal))
            played = list(df[(df["role"] == "music") & (df["turn_number"] < tn)]["content"])
            played_tids.append(played[-max_seq:])
            gold_tids.append(gold)
            hist_dialog.append(prior_rows)
    return queries, ctx_texts, played_tids, gold_tids


def _pad_pool(pool_idx_rows, K, n_items, rng):
    """Pad each row's pool to exactly K columns with random non-pool catalog
    indices (fake easy negatives) so the batch is rectangular. Returns (B,K) int."""
    out = np.zeros((len(pool_idx_rows), K), dtype=np.int64)
    for i, row in enumerate(pool_idx_rows):
        row = list(row[:K])
        if len(row) < K:
            have = set(row)
            while len(row) < K:
                j = int(rng.integers(n_items))
                if j not in have:
                    row.append(j); have.add(j)
        out[i] = row
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--warm-start", default="sasrec_v1",
                   help="dir under retrieval_v2/sasrec/ to warm-start from (sasrec.pt)")
    p.add_argument("--out", default="sasrec_inpool_v1")
    p.add_argument("--topk", type=int, default=100, help="pool size K (train==serve)")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-seq", type=int, default=50)
    p.add_argument("--n-sessions", type=int, default=None, help="smoke: cap train sessions")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # --- catalog + warm-start model ---------------------------------------
    item_db = MusicCatalogDB(ITEM_DB, ["all_tracks"], CORPUS)
    track_ids, feats, modality_dims = load_item_feats(["all_tracks"])
    tid_to_idx = {t: i for i, t in enumerate(track_ids)}
    feats_t = torch.as_tensor(feats, dtype=torch.float32, device=dev)
    n_items = len(track_ids)

    ckpt = torch.load(
        os.path.join(args.cache_dir, "retrieval_v2", "sasrec", args.warm_start, "sasrec.pt"),
        map_location="cpu", weights_only=False)
    model = SasrecModel(**ckpt["model_kwargs"]).to(dev)
    model.load_state_dict(ckpt["state_dict"])
    print(f"[inpool] warm-started {args.warm_start} | d={model.d} items={n_items}")

    # --- samples + SASRec-free pool ---------------------------------------
    queries, ctx_texts, played_tids, gold_tids = build_samples(
        load_dataset(CONV, split="train"), item_db, args.max_seq, args.n_sessions)
    print(f"[inpool] {len(queries)} (session,turn) samples")

    # SASRec-free union (default extra_config = bm25 + dense + same_artist).
    union = load_retrieval_module("wrrf_union_v1", ITEM_DB, ["all_tracks"], CORPUS,
                                  args.cache_dir, extra_config={})
    batch_ctx = [{"history_tids": pt} for pt in played_tids]
    pools = []
    CH = 256
    for i in tqdm(range(0, len(queries), CH), desc="pool"):
        pools.extend(union.batch_text_to_item_retrieval(
            queries[i:i + CH], topk=args.topk, batch_context=batch_ctx[i:i + CH]))

    # Keep only gold-in-pool samples (skip recall misses).
    keep, targets = [], []
    for i, (pool, gold) in enumerate(zip(pools, gold_tids)):
        t = inpool_target_index(pool, gold)
        if t is not None:
            keep.append(i); targets.append(t)
    print(f"[inpool] gold-in-pool: {len(keep)}/{len(queries)} "
          f"({100*len(keep)/max(len(queries),1):.1f}%) -> trainable")

    # --- tensors ----------------------------------------------------------
    st = SentenceTransformer(CTX_MODEL, device=dev)
    cache_kw = dict(cache_dir=args.cache_dir, encoder_name=CTX_MODEL,
                    max_seq_length=256, truncation_side="left")
    ctx = encode_dialogs_cached(st, [ctx_texts[i] for i in keep],
                                split_label="train_inpool", **cache_kw)
    ctx_t = torch.as_tensor(ctx, dtype=torch.float32, device=dev)

    pool_idx = _pad_pool([[tid_to_idx[t] for t in pools[i] if t in tid_to_idx]
                          for i in keep], args.topk, n_items, rng)
    pool_idx_t = torch.as_tensor(pool_idx, device=dev)
    target_t = torch.as_tensor(targets, dtype=torch.long, device=dev)

    # played-track feature sequence (padded to max_seq) + lengths
    L = args.max_seq
    item_in = feats.shape[1]
    played_t = torch.zeros(len(keep), L, item_in, dtype=torch.float32, device=dev)
    lengths = torch.zeros(len(keep), dtype=torch.long, device=dev)
    for r, i in enumerate(keep):
        idxs = [tid_to_idx[t] for t in played_tids[i] if t in tid_to_idx][-L:]
        if idxs:
            played_t[r, :len(idxs)] = feats_t[idxs]
            lengths[r] = len(idxs)

    # --- fine-tune --------------------------------------------------------
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    N = len(keep)
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(N, device=dev)
        tot = 0.0
        for s in tqdm(range(0, N, args.batch_size), desc=f"epoch {ep+1}/{args.epochs}"):
            b = perm[s:s + args.batch_size]
            pool_feats = feats_t[pool_idx_t[b]]            # (B,K,item_in)
            loss = inpool_loss(model, ctx_t[b], played_t[b], lengths[b],
                               pool_feats, target_t[b])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(b)
        print(f"[inpool] epoch {ep+1}/{args.epochs} train_loss={tot/max(N,1):.4f}")

    # --- save (same layout the sasrec factory loads) ----------------------
    out_dir = os.path.join(args.cache_dir, "retrieval_v2", "sasrec", args.out)
    os.makedirs(out_dir, exist_ok=True)
    torch.save({
        "state_dict": model.cpu().state_dict(),
        "model_kwargs": ckpt["model_kwargs"],
        "item_feats": feats_t.cpu(), "track_ids": track_ids,
    }, os.path.join(out_dir, "sasrec.pt"))
    print(f"[inpool] saved -> {out_dir}/sasrec.pt  "
          f"(gate it: nb74 # 12c-inpool, dev nDCG@20 vs 0.1652)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
