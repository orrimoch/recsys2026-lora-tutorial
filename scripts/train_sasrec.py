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
import random
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
# v2 (post-review): dropped cf-bpr — it's structurally inert for the 62% new-
# artist majority (BPR factors learned from sparse interactions; cold tracks
# get noise). Keeping metadata (text) + CLAP (audio) only.
META_COL, CLAP_COL = "metadata-qwen3_embedding_0.6b", "audio-laion_clap"
MODALITY_COLS = (META_COL, CLAP_COL)
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
    """Concatenate the configured frozen modalities -> (N, sum of dims), imputed,
    aligned to one track_id order. Each modality's dim is inferred from its
    first non-empty row, so a dim change can't silently wipe a modality.

    Returns: (track_ids, feats, modality_dims) where modality_dims is the
    per-modality dim list (used to construct ItemFusion's per-modality LNs)."""
    ds = concatenate_datasets([load_dataset(TRACK_EMB)[s] for s in splits])
    track_ids = list(ds["track_id"])
    parts, modality_dims = [], []
    for col in MODALITY_COLS:
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
        modality_dims.append(dim)
    return track_ids, np.concatenate(parts, axis=1), modality_dims


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


def _ctx_cache_path(cache_dir, encoder_name, max_seq_length, truncation_side,
                    split_label, dialogs):
    """Content-addressed disk path for a (config, dialogs) pair."""
    safe_enc = encoder_name.replace("/", "_")
    h = hashlib.sha1(("\n".join(dialogs)).encode("utf-8")).hexdigest()[:16]
    return os.path.join(
        cache_dir, "retrieval_v2", "sasrec", "ctx_cache",
        f"{safe_enc}__msl{max_seq_length}__ts{truncation_side}__{split_label}__{h}.npy")


def encode_dialogs_cached(st, dialogs, cache_dir, encoder_name, max_seq_length,
                          truncation_side, split_label, batch_size=256):
    """Encode dialogs with the SentenceTransformer, persisted under a
    content-hash on disk. Re-runs with identical dialogs + identical encoder
    config skip the encoding entirely.

    Cache key = (encoder_name, max_seq_length, truncation_side, sha1(dialogs)).
    Path: {cache_dir}/retrieval_v2/sasrec/ctx_cache/...npy  (Drive-symlinked).
    """
    p = _ctx_cache_path(cache_dir, encoder_name, max_seq_length, truncation_side,
                        split_label, dialogs)
    if os.path.exists(p):
        ctx = np.load(p)
        print(f"[sasrec] ctx cache HIT  {split_label}: {os.path.basename(p)} "
              f"shape={ctx.shape}")
        return ctx
    print(f"[sasrec] ctx cache MISS {split_label}: encoding {len(dialogs)} dialogs")
    ctx = st.encode(dialogs, convert_to_numpy=True, normalize_embeddings=True,
                    batch_size=batch_size, show_progress_bar=True).astype(np.float32)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.save(p, ctx)
    print(f"[sasrec] ctx cached -> {os.path.basename(p)}")
    return ctx


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
    p.add_argument("--d", type=int, default=192)
    p.add_argument("--max-seq", type=int, default=50)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # Reproducibility: seed every RNG that can affect the run. Note: the bge
    # encoder is deterministic at this batch size/precision, and the
    # session-disjoint val split is already deterministic via SHA1(session_id).
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"[sasrec] seed={args.seed}")

    track_ids, feats, modality_dims = load_item_feats(["all_tracks"])
    tid_to_idx = {t: i for i, t in enumerate(track_ids)}
    feats_t = torch.as_tensor(feats, device=dev)
    print(f"[sasrec] item feats {feats.shape} modality_dims={modality_dims}")

    (tr_d, tr_p, tr_t), (val_d, val_p, val_t) = build_train_val(
        tid_to_idx, args.max_seq, args.val_frac)
    te_d, te_p, te_t = build_test(tid_to_idx, args.max_seq)
    print(f"[sasrec] examples train={len(tr_t)} val={len(val_t)} test={len(te_t)} "
          f"(val_frac={args.val_frac}, session-disjoint)")

    # Diagnostics — partition the val/test gap into its drivers.
    def _stats(label, prefixes, dialogs):
        plens = np.array([len(p) for p in prefixes]) if prefixes else np.array([0])
        clens = np.array([len(d) for d in dialogs]) if dialogs else np.array([0])
        print(f"[sasrec] {label}: prefix_len median={int(np.median(plens))} "
              f"p95={int(np.percentile(plens, 95))} max={int(plens.max())} | "
              f"dialog_chars median={int(np.median(clens))} "
              f"p95={int(np.percentile(clens, 95))} max={int(clens.max())}")
    _stats("train", tr_p, tr_d)
    _stats("val",   val_p, val_d)
    _stats("test",  te_p, te_d)

    st = SentenceTransformer(CTX_MODEL)
    st.max_seq_length = 512
    try:
        st.tokenizer.truncation_side = "left"
    except Exception:
        pass
    cache_kw = dict(cache_dir=args.cache_dir, encoder_name=CTX_MODEL,
                    max_seq_length=512, truncation_side="left",
                    batch_size=args.batch_size)
    tr_ctx = encode_dialogs_cached(st, tr_d, split_label="train", **cache_kw)
    val_ctx = encode_dialogs_cached(st, val_d, split_label="val", **cache_kw)
    te_ctx = encode_dialogs_cached(st, te_d, split_label="test", **cache_kw)
    ctx_in_dim = int(tr_ctx.shape[1])
    tr_ctx_t = torch.as_tensor(tr_ctx, device=dev)
    val_ctx_t = torch.as_tensor(val_ctx, device=dev)
    te_ctx_t = torch.as_tensor(te_ctx, device=dev)
    print(f"[sasrec] ctx dim {ctx_in_dim}")

    model = SasrecModel(item_modality_dims=modality_dims, ctx_in_dim=ctx_in_dim,
                        d=args.d, max_len=args.max_seq).to(dev)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[sasrec] model: d={args.d} trainable_params={n_params:,}")
    # AdamW + weight_decay=1e-2: decouples L2 from the adaptive denominator
    # (the well-known Adam pitfall) and was a direct response to train_loss
    # diverging from val_loss after epoch 5 on the 10-epoch run.
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    # CosineAnnealingLR: stepped ONCE per epoch (not per batch). T_max =
    # args.epochs so the cosine reaches eta_min at the final epoch; eta_min
    # = lr * 0.1 keeps a non-trivial late-epoch LR floor.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.1)

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
            fb = torch.zeros(len(bi), L, feats.shape[1], device=dev)
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
        # Step the LR schedule once per epoch, AFTER the inner batch loop
        # and BEFORE the val pass — so the printed LR is the rate that was
        # used during this epoch.
        cur_lr = opt.param_groups[0]["lr"]
        sched.step()
        val_loss, val_r20, val_r100 = evaluate(
            model, val_ctx_t, val_p, val_t, feats_t, feats.shape[1], args.batch_size, dev)
        marker = ""
        if val_r100 > best_r100:
            best_r100 = val_r100
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            marker = "  * best"
        print(f"[sasrec] epoch {ep} | lr={cur_lr:.2e} train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_recall@20={val_r20:.4f} "
              f"val_recall@100={val_r100:.4f}{marker}")

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(dev)
        print(f"[sasrec] restored best model (val_recall@100={best_r100:.4f})")

    te_loss, te_r20, te_r100 = evaluate(
        model, te_ctx_t, te_p, te_t, feats_t, feats.shape[1], args.batch_size, dev)
    print(f"[sasrec] TEST (standalone SASRec) recall@20={te_r20:.4f} "
          f"recall@100={te_r100:.4f} loss={te_loss:.4f}")

    out_dir = os.path.join(args.cache_dir, "retrieval_v2", "sasrec", args.out)
    os.makedirs(out_dir, exist_ok=True)
    # Save item_feats as a torch tensor (NOT numpy) so the checkpoint loads
    # under PyTorch 2.6's weights_only=True. track_ids is a list[str] which is
    # also weights_only-safe. See the loader in retrieval_modules/__init__.py
    # (sasrec_seq branch) for the matching legacy-fallback warning.
    item_feats_t = torch.as_tensor(feats, dtype=torch.float32).cpu()
    torch.save({
        "state_dict": model.cpu().state_dict(),
        "model_kwargs": {"item_modality_dims": list(modality_dims),
                         "ctx_in_dim": ctx_in_dim,
                         "d": args.d, "max_len": args.max_seq},
        "item_feats": item_feats_t, "track_ids": track_ids,
    }, os.path.join(out_dir, "sasrec.pt"))
    print(f"[sasrec] saved -> {out_dir}/sasrec.pt")


if __name__ == "__main__":
    main()
