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
    val; the rest to train. Query = raw_with_goal text at the turn (MATCHES the
    served query_preprocessing_mode in config 194/197 and the dev harness, so the
    query tower sees the same distribution at train / eval / serve — no skew);
    gold excluded from being its own negative downstream via in-batch labels."""
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
            q = build_retrieval_query(sm, mode="raw_with_goal", goal_text=gt, user_profile=up)
            artist = ""
            try:
                artist = (item_db.id_to_metadata(music["content"]) or "")
            except Exception:
                pass
            bucket.append((q, gold, artist))
    return train, val


def build_artist_index(track_ids, metadata_dict):
    """idx_to_artist (lowercased, aligned to track_ids) + artist -> [track indices].
    Tracks with no artist are not grouped (can't form same-artist negatives)."""
    idx_to_artist, artist_to_idx = [], {}
    for i, tid in enumerate(track_ids):
        md = metadata_dict.get(tid) or {}
        a = md.get("artist_name")
        a = (", ".join(map(str, a)) if isinstance(a, list) else str(a or "")).strip().lower()
        idx_to_artist.append(a)
        if a:
            artist_to_idx.setdefault(a, []).append(i)
    return idx_to_artist, artist_to_idx


def sample_hard_negatives(gold_indices, idx_to_artist, artist_to_idx, n_per, rng):
    """Same-artist, different-track negative indices for a batch of golds. Excludes the
    batch golds (they are in-batch positives). Returns a sorted unique index list.
    Forces the model to separate a track from other tracks by the same artist —
    finer content discrimination than random in-batch negatives."""
    golds = set(int(g) for g in gold_indices)
    negs = set()
    for g in gold_indices:
        pool = artist_to_idx.get(idx_to_artist[g], ())
        cands = [j for j in pool if j != g]
        if not cands:
            continue
        k = min(int(n_per), len(cands))
        for c in rng.choice(len(cands), size=k, replace=False):
            negs.add(int(cands[int(c)]))
    negs.difference_update(golds)
    return sorted(negs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--dataset-name", required=True, help="catalog dataset path for MusicCatalogDB")
    p.add_argument("--out", default="two_tower_v1")
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01,
                   help="AdamW L2 regularization; fights the fast overfit (0.0 = off)")
    p.add_argument("--dropout", type=float, default=0.3,
                   help="dropout in both towers; bump to 0.5 if val still overfits")
    p.add_argument("--holdout-frac", type=float, default=0.15)
    p.add_argument("--n-sessions", type=int, default=999999)
    p.add_argument("--n-hard-negs", type=int, default=0,
                   help="same-artist hard negatives per batch gold appended to the "
                        "in-batch InfoNCE bank (0 = off). Forces finer content "
                        "discrimination; try 4.")
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

    # Same-artist hard negatives (opt-in): an artist -> track-indices index so each
    # batch can append same-artist different-track negatives to the InfoNCE bank.
    idx_to_artist, artist_to_idx = (build_artist_index(track_ids, item_db.metadata_dict)
                                    if args.n_hard_negs > 0 else (None, None))
    if args.n_hard_negs > 0:
        print(f"[two-tower] hard negatives ON: n_per={args.n_hard_negs}, "
              f"{len(artist_to_idx)} artists with >=1 track")

    train_split = load_dataset(CONV, split="train")
    # Subset FIRST, then carve the temporal holdout WITHIN the subset, so a small
    # --n-sessions (a quick trend run) still yields a leak-free val set (latest
    # sessions by date). Computing holdout over the full split + taking the first
    # N rows would leave a small subset with ZERO val pairs -> no val curve.
    rows = list(train_split)[: args.n_sessions]
    sessions = [{"session_id": r["session_id"], "session_date": r["session_date"]}
                for r in rows]
    _, holdout_ids = select_temporal_holdout(sessions, frac=args.holdout_frac)
    train_pairs, val_pairs = build_pairs(rows, tid_to_idx, holdout_ids, item_db)
    print(f"[two-tower] train pairs={len(train_pairs)} val pairs={len(val_pairs)} "
          f"(n_sessions={len(rows)}, time-based holdout {len(holdout_ids)} sessions)")

    # Frozen Qwen3 query encoder (same as the dense channel) -> cache query embs.
    from mcrs.retrieval_modules import QWEN3_MUSIC_INSTRUCT
    from mcrs.retrieval_modules.dense_precomputed import DENSE_PRECOMPUTED
    qenc = DENSE_PRECOMPUTED(args.dataset_name, ["all_tracks"],
                             ["track_name", "artist_name", "album_name"], args.cache_dir,
                             embed_col="metadata-qwen3_embedding_0.6b",
                             instruct=QWEN3_MUSIC_INSTRUCT, instruct_label="instruct-music-v1")

    # Pre-encode every UNIQUE query ONCE (reuses + fills the shared query cache),
    # so the epoch loop is pure cache lookups instead of re-encoding ~103k queries
    # 5x via the private _encode_queries path (which bypasses the cache). This is
    # the single biggest speedup: ~23 min/epoch -> seconds/epoch.
    uniq = list({p[0] for p in train_pairs} | {p[0] for p in val_pairs})
    miss = [q for q in uniq if q not in qenc._query_cache]
    print(f"[two-tower] pre-encoding {len(miss)}/{len(uniq)} uncached queries "
          f"({len(uniq) - len(miss)} cache hits)...")
    for i in tqdm(range(0, len(miss), 128)):
        chunk = miss[i:i + 128]
        for q, e in zip(chunk, qenc._encode_queries(chunk)):
            qenc._query_cache[q] = e.astype(np.float32)
    if miss:
        qenc._query_cache_dirty = True
        qenc._save_query_cache()  # persist so future runs are instant too

    def encode_queries(texts):
        return np.stack([qenc._query_cache[q] for q in texts]).astype(np.float32)

    q_in_dim = encode_queries([train_pairs[0][0]]).shape[1]
    feats_t = torch.as_tensor(feats, dtype=torch.float32)
    # Self-describing model_kwargs: pin EVERY constructor arg so the factory
    # rebuilds an identically-shaped model even if a default changes later.
    model_kwargs = {"item_modality_dims": modality_dims, "q_in_dim": q_in_dim,
                    "d": args.d, "hidden": 1536, "dropout": args.dropout, "temperature": 0.07}
    model = TwoTowerModel(**model_kwargs).to(device)
    # AdamW (decoupled weight decay) instead of Adam — the half-data run overfit
    # by epoch 2 (train kept dropping, val rose); weight decay + dropout are the
    # cheap regularizers before reaching for hard negatives / more data.
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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
            neg_feats = None
            if train and args.n_hard_negs > 0 and artist_to_idx:
                neg_idx = sample_hard_negatives(gold, idx_to_artist, artist_to_idx,
                                                args.n_hard_negs, np.random)
                if neg_idx:
                    neg_feats = feats_t[neg_idx].to(device)
            loss = info_nce_loss(model, q_emb, pos, extra_neg_feats=neg_feats)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.detach()) * len(idx); n += len(idx)
        return total / max(n, 1)

    out_dir = Path(args.cache_dir) / "retrieval_v2" / "two_tower" / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    def _save(path, epoch, val, rec100=float("nan")):
        # CPU-copy the state_dict WITHOUT moving the live model off-GPU (a plain
        # model.cpu() mid-loop would force the next epoch onto CPU). Self-contained
        # checkpoint: the factory rebuilds from model_kwargs + state_dict +
        # item_feats + track_ids.
        torch.save({
            "model_kwargs": model_kwargs,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "item_feats": feats_t.cpu(),
            "track_ids": track_ids,
            "epoch": epoch, "val_loss": float(val), "val_recall_100": float(rec100),
        }, path)

    def val_recall(ks=(20, 100)):
        """Honest metric: rank each val gold against the FULL 47k catalog (not just
        in-batch). InfoNCE val_loss overfits while this — the metric # 4-tt actually
        cares about — can still be improving. Print it instead of staring at loss."""
        if not val_pairs:
            return {k: float("nan") for k in ks}
        model.eval()
        with torch.no_grad():
            item_repr = model.encode_item(feats_t.to(device))      # (N,d) L2-normed
            hit = {k: 0 for k in ks}
            for s in range(0, len(val_pairs), args.batch_size):
                batch = val_pairs[s:s + args.batch_size]
                q_emb = torch.as_tensor(encode_queries([p[0] for p in batch])).to(device)
                gold = torch.tensor([p[1] for p in batch], device=device)
                top = (model.encode_query(q_emb) @ item_repr.t()).topk(max(ks), dim=1).indices
                for k in ks:
                    hit[k] += (top[:, :k] == gold[:, None]).any(dim=1).sum().item()
            del item_repr
            return {k: hit[k] / len(val_pairs) for k in ks}

    best_recall = -1.0
    for ep in range(args.epochs):
        tr = run_epoch(train_pairs, train=True)
        with torch.no_grad():
            va = run_epoch(val_pairs, train=False) if val_pairs else float("nan")
        vr = val_recall()
        print(f"[two-tower] epoch {ep+1}/{args.epochs} train_loss={tr:.4f} val_loss={va:.4f} "
              f"val_recall@20={vr[20]:.4f} val_recall@100={vr[100]:.4f}")
        # Checkpoint EVERY epoch -> an interrupt / Colab disconnect keeps the
        # latest completed epoch instead of losing the whole run.
        _save(out_dir / "two_tower_last.pt", ep + 1, va, vr[100])
        # two_tower.pt (what the factory loads) tracks BEST val_recall@100 — the
        # real metric — NOT val_loss (which overfits while recall still climbs).
        # Falls back to the latest epoch when there is no val set.
        score = vr[100] if val_pairs else float(ep)
        if score > best_recall:
            best_recall = score
            _save(out_dir / "two_tower.pt", ep + 1, va, vr[100])
            print(f"[two-tower]   checkpoint -> two_tower.pt (epoch {ep+1}, "
                  f"val_recall@100={vr[100]:.4f})")

    print(f"[two-tower] done. best -> {out_dir / 'two_tower.pt'} (best val_recall@100={best_recall:.4f}); "
          f"latest -> {out_dir / 'two_tower_last.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
