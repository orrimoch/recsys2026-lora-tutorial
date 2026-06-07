"""Train the intent->content two-tower recall channel (Tier-1 #3.3).

Item tower = ItemFusion over the 5 frozen catalog modalities (metadata +
attributes + lyrics Qwen3, audio-CLAP, image-SigLIP). Query tower = a learned MLP
head over a frozen Qwen3-Embedding-0.6B query embedding. Trained with in-batch
InfoNCE on causal (query->gold) pairs, with same-artist hard negatives.

LEAK-FREE: time-based holdout (latest sessions by session_date) via
carve_temporal_selection_set — NOT a random split (random leaks future artists,
the documented failure mode). Causal queries (turn k uses turns < k via
prior_turns). Played tracks excluded from candidates and negatives.

Saves {cache_dir}/retrieval_v2/two_tower/<out>/two_tower.pt with
{model_kwargs, state_dict, item_feats, track_ids} — the layout the
load_retrieval_module('two_tower') factory branch reads.

Colab GPU. Smoke first: --n-sessions 300 --epochs 1.

Usage:
  python scripts/train_two_tower.py --cache-dir <CACHE_DIR> --out two_tower_v1
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "music-crs-baselines"))

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402
from datasets import load_dataset, concatenate_datasets  # noqa: E402

from mcrs.retrieval_modules.two_tower_model import TwoTowerModel, info_nce_loss  # noqa: E402
from mcrs.retrieval_modules.sasrec_model import prior_turns  # noqa: E402
from mcrs.crs_baseline import build_retrieval_query  # noqa: E402
from mcrs.db_item import MusicCatalogDB  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from carve_temporal_selection_set import select_temporal_holdout  # noqa: E402

TRACK_EMB = "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings"
CONV = "talkpl-ai/TalkPlayData-Challenge-Dataset"
MODALITY_COLS = (
    "metadata-qwen3_embedding_0.6b",
    "attributes-qwen3_embedding_0.6b",
    "lyrics-qwen3_embedding_0.6b",
    "audio-laion_clap",
    "image-siglip2",
)


def _impute(mat, valid):
    """Replace empty/missing rows with the column mean of the valid rows."""
    if valid.all():
        return mat
    mean = mat[valid].mean(axis=0, keepdims=True) if valid.any() else np.zeros((1, mat.shape[1]))
    mat[~valid] = mean
    return mat


def load_item_feats(splits=("all_tracks",)):
    """Load + concat the 5 frozen modalities into one (N, sum_dims) matrix aligned
    to a single track_id order. Returns (track_ids, feats, modality_dims)."""
    ds = concatenate_datasets([load_dataset(TRACK_EMB)[s] for s in splits])
    track_ids = list(ds["track_id"])
    parts, modality_dims = [], []
    for col in MODALITY_COLS:
        rows = ds[col]
        dim = next((len(r) for r in rows if r is not None and len(r)), 0)
        mat = np.zeros((len(rows), dim), dtype=np.float32)
        valid = np.zeros(len(rows), dtype=bool)
        for i, r in enumerate(rows):
            if r is not None and len(r) == dim:
                mat[i] = r
                valid[i] = True
        mat = _impute(mat, valid)
        parts.append(mat)
        modality_dims.append(dim)
    return track_ids, np.concatenate(parts, axis=1), modality_dims


def build_pairs(hf_split, tid_to_idx, holdout_ids, item_db):
    """Causal (query_text, gold_idx, artist) pairs. Sessions in holdout_ids go to
    val; the rest to train. Query = raw_enriched text at the turn; gold excluded
    from being its own negative downstream via in-batch labels."""
    train, val = [], []
    for sess in hf_split:
        sid = str(sess.get("session_id"))
        df = pd.DataFrame(sess["conversations"])
        goal = sess.get("conversation_goal") or {}
        gt = (goal.get("listener_goal") or "").strip()
        up = sess.get("user_profile") or {}
        bucket = val if sid in holdout_ids else train
        for _, music in df[df["role"] == "music"].iterrows():
            tn = int(music["turn_number"])
            gold = tid_to_idx.get(music["content"])
            if gold is None:
                continue
            prior = prior_turns(df, tn)
            sm = [{"role": ("assistant" if t["role"] == "music" else t["role"]),
                   "content": (item_db.id_to_metadata(t["content"])
                               if t["role"] == "music" else t["content"])}
                  for _, t in prior.iterrows()]
            q = build_retrieval_query(sm, mode="raw_enriched", goal_text=gt, user_profile=up)
            artist = ""
            try:
                artist = (item_db.id_to_metadata(music["content"]) or "")
            except Exception:
                pass
            bucket.append((q, gold, artist))
    return train, val


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--dataset-name", required=True, help="catalog dataset path for MusicCatalogDB")
    p.add_argument("--out", default="two_tower_v1")
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--holdout-frac", type=float, default=0.15)
    p.add_argument("--n-sessions", type=int, default=999999)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    item_db = MusicCatalogDB(args.dataset_name, ["all_tracks"],
                             ["track_name", "artist_name", "album_name"])
    track_ids, feats, modality_dims = load_item_feats(["all_tracks"])
    tid_to_idx = {t: i for i, t in enumerate(track_ids)}
    print(f"[two-tower] item feats {feats.shape} modality_dims={modality_dims}")

    train_split = load_dataset(CONV, split="train")
    sessions = [{"session_id": r["session_id"], "session_date": r["session_date"]}
                for r in train_split]
    _, holdout_ids = select_temporal_holdout(sessions, frac=args.holdout_frac)
    rows = list(train_split)[: args.n_sessions]
    train_pairs, val_pairs = build_pairs(rows, tid_to_idx, holdout_ids, item_db)
    print(f"[two-tower] train pairs={len(train_pairs)} val pairs={len(val_pairs)} "
          f"(time-based holdout {len(holdout_ids)} sessions)")

    # Frozen Qwen3 query encoder (same as the dense channel) -> cache query embs.
    from mcrs.retrieval_modules import QWEN3_MUSIC_INSTRUCT
    from mcrs.retrieval_modules.dense_precomputed import DENSE_PRECOMPUTED
    qenc = DENSE_PRECOMPUTED(args.dataset_name, ["all_tracks"],
                             ["track_name", "artist_name", "album_name"], args.cache_dir,
                             embed_col="metadata-qwen3_embedding_0.6b",
                             instruct=QWEN3_MUSIC_INSTRUCT, instruct_label="instruct-music-v1")

    def encode_queries(texts):
        return np.asarray(qenc._encode_queries(list(texts)), dtype=np.float32)

    q_in_dim = encode_queries([train_pairs[0][0]]).shape[1]
    feats_t = torch.as_tensor(feats, dtype=torch.float32)
    model = TwoTowerModel(item_modality_dims=modality_dims, q_in_dim=q_in_dim,
                          d=args.d).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def run_epoch(pairs, train=True):
        model.train(train)
        order = np.random.permutation(len(pairs)) if train else np.arange(len(pairs))
        total, n = 0.0, 0
        for s in tqdm(range(0, len(order), args.batch_size), disable=not train):
            idx = order[s:s + args.batch_size]
            q_txt = [pairs[i][0] for i in idx]
            gold = [pairs[i][1] for i in idx]
            q_emb = torch.as_tensor(encode_queries(q_txt)).to(device)
            pos = feats_t[gold].to(device)
            loss = info_nce_loss(model, q_emb, pos)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.detach()) * len(idx); n += len(idx)
        return total / max(n, 1)

    for ep in range(args.epochs):
        tr = run_epoch(train_pairs, train=True)
        with torch.no_grad():
            va = run_epoch(val_pairs, train=False) if val_pairs else float("nan")
        print(f"[two-tower] epoch {ep+1}/{args.epochs} train_loss={tr:.4f} val_loss={va:.4f}")

    out_dir = Path(args.cache_dir) / "retrieval_v2" / "two_tower" / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_kwargs": {"item_modality_dims": modality_dims, "q_in_dim": q_in_dim,
                         "d": args.d},
        "state_dict": model.cpu().state_dict(),
        "item_feats": feats_t.cpu(),
        "track_ids": track_ids,
    }, out_dir / "two_tower.pt")
    print(f"[two-tower] saved -> {out_dir / 'two_tower.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
