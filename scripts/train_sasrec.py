"""Train the dialog-conditioned content-fused SASRec and save it for the
sasrec_seq channel.

Leakage discipline:
- Training and per-epoch validation both come from the TRAIN split. The val
  slice is held out SESSION-DISJOINTLY (so no turns of a val session ever
  appear in training). Deterministic via SHA1(session_id).
- The TEST split (Challenge-Dataset 'test') is used ONCE at the end for a
  standalone sanity number; model selection uses val, never test.

Per-epoch output: train_loss, val_loss, val_recall@20, val_recall@100. Best
model (by val_recall@100) is kept and saved. Final line: standalone test
recall@{20,100}. The union+SASRec recall ablation lives in the notebook cell.

Saves {cache_dir}/retrieval_v2/sasrec/{out}/sasrec.pt = dict(state_dict,
model_kwargs, item_feats (N, item_in_dim), track_ids (len N)).
"""
import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

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


def _is_val_session(session_id, val_frac):
    """Deterministic session-disjoint hold-out: fixed by SHA1(session_id)."""
    h = int(hashlib.sha1(str(session_id).encode()).hexdigest()[:8], 16)
    return (h % 10000) < int(val_frac * 10000)


def _walk_split(hf_split, tid_to_idx, max_seq):
    """Walk an HF conversation split -> per (session_id, music turn) yield
    (session_id, dialog text up to t, played-index prefix, target index)."""
    out = []
    for sess in hf_split:
        sid = str(sess.get("session_id") or id(sess))
        df = pd.DataFrame(sess["conversations"])
        played = []
        for _, m in df[df["role"] == "music"].iterrows():
            tn = int(m["turn_number"])
            tgt = tid_to_idx.get(m["content"])
            prior = df[(df["turn_number"] < tn) |
                       ((df["turn_number"] == tn) & (df["role"] == "user"))]
            if tgt is not None:
                out.append((sid,
                            build_user_dialog(prior.to_dict("records")),
                            played[-max_seq:],
                            tgt))
                played.append(tgt)
    return out


def build_train_val(tid_to_idx, max_seq, val_frac):
    """Train-split sessions -> (train, val) lists, session-disjoint."""
    tr = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="train")
    train_d, train_p, train_t = [], [], []
    val_d, val_p, val_t = [], [], []
    for sid, d, p, t in _walk_split(tr, tid_to_idx, max_seq):
        if _is_val_session(sid, val_frac):
            val_d.append(d); val_p.append(p); val_t.append(t)
        else:
            train_d.append(d); train_p.append(p); train_t.append(t)
    return (train_d, train_p, train_t), (val_d, val_p, val_t)


def build_test(tid_to_idx, max_seq):
    """Test-split examples (used ONCE, end-of-run sanity)."""
    te = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="test")
    rows = _walk_split(te, tid_to_idx, max_seq)
    return ([r[1] for r in rows], [r[2] for r in rows], [r[3] for r in rows])


def encode_dialogs(st, dialogs, batch_size=256):
    """bge-base-en CLS embeddings, L2-normalized; oldest-first truncation."""
    return st.encode(dialogs, convert_to_numpy=True, normalize_embeddings=True,
                     batch_size=batch_size, show_progress_bar=True).astype(np.float32)


def evaluate(model, ctx_t, prefixes, targets, feats_t, item_in_dim, batch_size, device):
    """Mean per-example CE loss + recall@20 + recall@100, computed in eval mode.
    Recomputes the catalog item-repr matrix once with no_grad."""
    model.eval()
    loss_sum, hit20, hit100, n = 0.0, 0, 0, 0
    with torch.no_grad():
        item_matrix = model.item_fusion(feats_t)
        for s in range(0, len(targets), batch_size):
            bi = list(range(s, min(s + batch_size, len(targets))))
            L = max((len(prefixes[i]) for i in bi), default=0)
            L = max(L, 1)
            fb = torch.zeros(len(bi), L, item_in_dim, device=device)
            ln = torch.zeros(len(bi), dtype=torch.long, device=device)
            for j, i in enumerate(bi):
                pf = prefixes[i]
                ln[j] = len(pf)
                if pf:
                    fb[j, :len(pf)] = feats_t[pf]
            cb = ctx_t[bi]
            tb = torch.as_tensor([targets[i] for i in bi], device=device)
            state = model.encode(cb, fb, ln)
            logits = model.score(state, item_matrix)
            loss = F.cross_entropy(logits, tb, reduction="sum")
            top100 = logits.topk(100, dim=1).indices
            tgt_col = tb.unsqueeze(1)
            hit100 += int((top100 == tgt_col).any(dim=1).sum().item())
            hit20 += int((top100[:, :20] == tgt_col).any(dim=1).sum().item())
            loss_sum += float(loss.item())
            n += len(bi)
    model.train()
    return loss_sum / max(1, n), hit20 / max(1, n), hit100 / max(1, n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--out", default="sasrec_v1")
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--max-seq", type=int, default=50)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.1)
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    track_ids, feats = load_item_feats(["all_tracks"])
    tid_to_idx = {t: i for i, t in enumerate(track_ids)}
    feats_t = torch.as_tensor(feats, device=dev)
    item_in_dim = feats.shape[1]
    print(f"[sasrec] item feats {feats.shape}")

    (tr_d, tr_p, tr_t), (val_d, val_p, val_t) = build_train_val(
        tid_to_idx, args.max_seq, args.val_frac)
    te_d, te_p, te_t = build_test(tid_to_idx, args.max_seq)
    print(f"[sasrec] examples train={len(tr_t)} val={len(val_t)} test={len(te_t)} "
          f"(val_frac={args.val_frac}, session-disjoint)")

    st = SentenceTransformer(CTX_MODEL)
    st.max_seq_length = 512
    try:
        st.tokenizer.truncation_side = "left"
    except Exception:
        pass
    print("[sasrec] encoding train dialogs")
    tr_ctx = encode_dialogs(st, tr_d, args.batch_size)
    print("[sasrec] encoding val dialogs")
    val_ctx = encode_dialogs(st, val_d, args.batch_size)
    print("[sasrec] encoding test dialogs")
    te_ctx = encode_dialogs(st, te_d, args.batch_size)
    ctx_in_dim = int(tr_ctx.shape[1])
    tr_ctx_t = torch.as_tensor(tr_ctx, device=dev)
    val_ctx_t = torch.as_tensor(val_ctx, device=dev)
    te_ctx_t = torch.as_tensor(te_ctx, device=dev)
    print(f"[sasrec] ctx dim {ctx_in_dim}")

    model = SasrecModel(item_in_dim=item_in_dim, ctx_in_dim=ctx_in_dim,
                        d=args.d, max_len=args.max_seq).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_r100 = -1.0
    best_state = None
    order = np.arange(len(tr_t))
    for ep in range(args.epochs):
        np.random.shuffle(order)
        model.train()
        tot, nb = 0.0, 0
        for s in range(0, len(order), args.batch_size):
            bi = order[s:s + args.batch_size]
            L = max((len(tr_p[i]) for i in bi), default=0)
            L = max(L, 1)
            fb = torch.zeros(len(bi), L, item_in_dim, device=dev)
            ln = torch.zeros(len(bi), dtype=torch.long, device=dev)
            for j, i in enumerate(bi):
                pf = tr_p[i]
                ln[j] = len(pf)
                if pf:
                    fb[j, :len(pf)] = feats_t[pf]
            cb = tr_ctx_t[bi]
            tb = torch.as_tensor([tr_t[i] for i in bi], device=dev)
            opt.zero_grad()
            item_matrix = model.item_fusion(feats_t)
            loss = next_item_loss(model, cb, fb, ln, tb, item_matrix)
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        train_loss = tot / max(1, nb)
        val_loss, val_r20, val_r100 = evaluate(
            model, val_ctx_t, val_p, val_t, feats_t, item_in_dim, args.batch_size, dev)
        marker = ""
        if val_r100 > best_r100:
            best_r100 = val_r100
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            marker = "  * best"
        print(f"[sasrec] epoch {ep} | train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_recall@20={val_r20:.4f} "
              f"val_recall@100={val_r100:.4f}{marker}")

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(dev)
        print(f"[sasrec] restored best model (val_recall@100={best_r100:.4f})")

    te_loss, te_r20, te_r100 = evaluate(
        model, te_ctx_t, te_p, te_t, feats_t, item_in_dim, args.batch_size, dev)
    print(f"[sasrec] TEST (standalone SASRec) recall@20={te_r20:.4f} "
          f"recall@100={te_r100:.4f} loss={te_loss:.4f}")

    out_dir = os.path.join(args.cache_dir, "retrieval_v2", "sasrec", args.out)
    os.makedirs(out_dir, exist_ok=True)
    torch.save({
        "state_dict": model.cpu().state_dict(),
        "model_kwargs": {"item_in_dim": item_in_dim, "ctx_in_dim": ctx_in_dim,
                         "d": args.d, "max_len": args.max_seq},
        "item_feats": feats, "track_ids": track_ids,
    }, os.path.join(out_dir, "sasrec.pt"))
    print(f"[sasrec] saved -> {out_dir}/sasrec.pt")


if __name__ == "__main__":
    main()
