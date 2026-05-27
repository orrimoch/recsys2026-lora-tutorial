"""Train the dialog-conditioned content-fused SASRec and save it for the
sasrec_seq channel. Leakage-safe: TRAIN split only. Runs on Colab GPU.

Saves {cache_dir}/retrieval_v2/sasrec/{out}/sasrec.pt = dict(state_dict,
model_kwargs, item_feats (N, item_in_dim), track_ids (len N)).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "music-crs-baselines"))
from datasets import load_dataset, concatenate_datasets  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
from mcrs.retrieval_modules.sasrec_model import (  # noqa: E402
    SasrecModel, build_user_dialog, next_item_loss)

TRACK_EMB = "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings"
META_COL, CLAP_COL, CF_COL = "metadata-qwen3_embedding_0.6b", "audio-laion_clap", "cf-bpr"
CTX_MODEL = "BAAI/bge-base-en-v1.5"


def _impute(mat, valid):
    """Replace invalid rows with the column mean of valid rows."""
    if valid.all():
        return mat
    gmean = mat[valid].mean(axis=0)
    mat = mat.copy()
    mat[~valid] = gmean
    return mat


def load_item_feats(splits):
    """Concatenate the 3 frozen modality embeddings -> (N, sum of modality dims),
    imputed, aligned to one track_id order. Each modality's dim is inferred from
    its first non-empty row, so a dim change can't silently wipe a modality."""
    ds = concatenate_datasets([load_dataset(TRACK_EMB)[s] for s in splits])
    track_ids = list(ds["track_id"])
    parts = []
    for col in (META_COL, CLAP_COL, CF_COL):
        raw = ds[col]
        dim = next((len(v) for v in raw if v is not None and len(v) > 0), None)
        if dim is None:
            raise ValueError(f"column {col!r} has no non-empty rows")
        mat = np.zeros((len(track_ids), dim), dtype=np.float32)
        valid = np.zeros(len(track_ids), dtype=bool)
        for i, v in enumerate(raw):
            if v is not None and len(v) == dim:
                mat[i] = v
                valid[i] = True
        print(f"[sasrec] {col}: dim={dim} valid={int(valid.sum())}/{len(track_ids)}")
        parts.append(_impute(mat, valid))
    return track_ids, np.concatenate(parts, axis=1)


def build_examples(tid_to_idx, max_seq):
    """Walk TRAIN sessions -> per music turn: (user-turns dialog up to t,
    played-track-index prefix, target index). Only targets in the catalog."""
    tr = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="train")
    dialogs, prefixes, targets = [], [], []
    for sess in tr:
        df = pd.DataFrame(sess["conversations"])
        played = []
        for _, m in df[df["role"] == "music"].iterrows():
            tn = int(m["turn_number"])
            tgt = tid_to_idx.get(m["content"])
            prior = df[(df["turn_number"] < tn) |
                       ((df["turn_number"] == tn) & (df["role"] == "user"))]
            if tgt is not None:
                dialogs.append(build_user_dialog(prior.to_dict("records")))
                prefixes.append(played[-max_seq:])
                targets.append(tgt)
                played.append(tgt)
    return dialogs, prefixes, targets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--out", default="sasrec_v1")
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--max-seq", type=int, default=50)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    track_ids, feats = load_item_feats(["all_tracks"])
    tid_to_idx = {t: i for i, t in enumerate(track_ids)}
    feats_t = torch.as_tensor(feats, device=dev)
    print(f"[sasrec] item feats {feats.shape}")

    dialogs, prefixes, targets = build_examples(tid_to_idx, args.max_seq)
    print(f"[sasrec] {len(targets)} training examples")

    st = SentenceTransformer(CTX_MODEL)
    st.max_seq_length = 512
    try:
        st.tokenizer.truncation_side = "left"
    except Exception:
        pass
    ctx = st.encode(dialogs, convert_to_numpy=True, normalize_embeddings=True,
                    batch_size=256, show_progress_bar=True).astype(np.float32)
    ctx_t = torch.as_tensor(ctx, device=dev)
    print(f"[sasrec] ctx {ctx.shape}")

    model = SasrecModel(item_in_dim=feats.shape[1], ctx_in_dim=int(ctx.shape[1]),
                        d=args.d, max_len=args.max_seq).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    order = np.arange(len(targets))
    for ep in range(args.epochs):
        np.random.shuffle(order)
        tot, nb = 0.0, 0
        for s in range(0, len(order), args.batch_size):
            bi = order[s:s + args.batch_size]
            L = max((len(prefixes[i]) for i in bi), default=0)
            L = max(L, 1)
            fb = torch.zeros(len(bi), L, feats.shape[1], device=dev)
            ln = torch.zeros(len(bi), dtype=torch.long, device=dev)
            for j, i in enumerate(bi):
                pf = prefixes[i]
                ln[j] = len(pf)
                if pf:
                    fb[j, :len(pf)] = feats_t[pf]
            cb = ctx_t[bi]
            tb = torch.as_tensor([targets[i] for i in bi], device=dev)
            opt.zero_grad()
            item_matrix = model.item_fusion(feats_t)
            loss = next_item_loss(model, cb, fb, ln, tb, item_matrix)
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        print(f"[sasrec] epoch {ep} mean_loss {tot / max(1, nb):.4f}")

    out_dir = os.path.join(args.cache_dir, "retrieval_v2", "sasrec", args.out)
    os.makedirs(out_dir, exist_ok=True)
    torch.save({
        "state_dict": model.cpu().state_dict(),
        "model_kwargs": {"item_in_dim": feats.shape[1], "ctx_in_dim": int(ctx.shape[1]),
                         "d": args.d, "max_len": args.max_seq},
        "item_feats": feats, "track_ids": track_ids,
    }, os.path.join(out_dir, "sasrec.pt"))
    print(f"[sasrec] saved -> {out_dir}/sasrec.pt")


if __name__ == "__main__":
    main()
