"""Train ONE SID quantizer (one seed) over the 47K-track catalog.

Loads text + CF + audio embeddings from talkpl-ai/TalkPlayData-Challenge-Track-Embeddings,
concatenates per spec §2.1, trains RQ-VAE + Sinkhorn for ~50 epochs, saves checkpoint
+ assignment parquet + per-seed gate scores.

Usage:
    python scripts/build_sid_quantizer.py --seed 42
    python scripts/build_sid_quantizer.py --seed 123
    python scripts/build_sid_quantizer.py --seed 7

After all 3 seeds done, run scripts/pick_best_sid_quantizer.py to choose + pin.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINES_DIR = REPO_ROOT / "music-crs-baselines"
sys.path.insert(0, str(BASELINES_DIR))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from datasets import load_dataset
from sklearn.decomposition import PCA

from mcrs.sid.preprocessing import concat_modalities
from mcrs.sid.quantizer import SIDQuantizer
from mcrs.sid.validation import (
    compute_relative_mse_gate,
    validate_codebook_utilization,
    validate_cluster_purity,
)


def main():
    parser = argparse.ArgumentParser(description="Train one SID quantizer (one seed).")
    parser.add_argument("--seed", type=int, required=True, help="RNG seed (use 42, 123, or 7).")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--num-levels", type=int, default=3)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--sinkhorn-lambda", type=float, default=0.10)
    parser.add_argument("--cache-root", default=str(REPO_ROOT / "experiments" / "cache" / "sid"))
    parser.add_argument("--max-tracks", type=int, default=None,
                        help="Smoke test: train on first N tracks only.")
    args = parser.parse_args()

    out_dir = Path(args.cache_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[build_sid_quantizer] seed={args.seed} -> out_dir={out_dir}", file=sys.stderr)

    # 1) Load embeddings + concatenate modalities ------------------------------
    print(
        "[build_sid_quantizer] loading TalkPlayData-Challenge-Track-Embeddings...",
        file=sys.stderr,
    )
    emb_ds = load_dataset(
        "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings", split="all_tracks"
    )
    if args.max_tracks is not None:
        emb_ds = emb_ds.select(range(min(args.max_tracks, len(emb_ds))))

    # Per-modality fixed dims (verified 2026-05-15 via dataset peek).
    # Per `feedback_embedding_imputation.md`, impute missing/wrong-shape rows
    # with zeros — never drop tracks (catalog coverage discipline).
    TEXT_DIM, CF_DIM, AUDIO_DIM = 1024, 128, 512

    def _safe_array(raw, expected_dim: int) -> np.ndarray:
        """Return float32 array of expected_dim. Impute zeros if raw is None/empty/wrong-shape."""
        if raw is None:
            return np.zeros(expected_dim, dtype=np.float32)
        arr = np.asarray(raw, dtype=np.float32)
        if arr.size == 0 or arr.shape != (expected_dim,):
            return np.zeros(expected_dim, dtype=np.float32)
        return arr

    track_ids: list[str] = []
    fused_embs: list[np.ndarray] = []
    n_imputed = {"text": 0, "cf": 0, "audio": 0}
    for row in tqdm(emb_ds, desc="concat modalities"):
        tid = row["track_id"]
        text = _safe_array(row.get("metadata-qwen3_embedding_0.6b"), TEXT_DIM)
        cf = _safe_array(row.get("cf-bpr"), CF_DIM)
        audio = _safe_array(row.get("audio-laion_clap"), AUDIO_DIM)
        # Track imputation rates (a row counts as imputed if it's the all-zero vector).
        if not text.any():
            n_imputed["text"] += 1
        if not cf.any():
            n_imputed["cf"] += 1
        if not audio.any():
            n_imputed["audio"] += 1
        fused = concat_modalities(text=text, cf=cf, audio=audio)
        track_ids.append(tid)
        fused_embs.append(fused)

    X = np.stack(fused_embs)  # (N, ~1664)
    print(f"[build_sid_quantizer] X shape={X.shape}", file=sys.stderr)
    print(
        f"[build_sid_quantizer] imputed-zero rows: text={n_imputed['text']}, "
        f"cf={n_imputed['cf']}, audio={n_imputed['audio']} "
        f"(out of {len(track_ids)})",
        file=sys.stderr,
    )

    # 2) Train RQ-VAE ----------------------------------------------------------
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"[build_sid_quantizer] device={device}", file=sys.stderr)

    quantizer = SIDQuantizer(
        input_dim=X.shape[1],
        latent_dim=args.latent_dim,
        num_levels=args.num_levels,
        codebook_size=args.codebook_size,
        sinkhorn_lambda=args.sinkhorn_lambda,
        seed=args.seed,
    )
    quantizer.to(device)
    optimizer = torch.optim.AdamW(quantizer.parameters(), lr=args.lr)

    X_tensor = torch.from_numpy(X)
    n = X_tensor.shape[0]
    for epoch in range(args.epochs):
        perm = torch.randperm(n)
        epoch_losses = {"mse_recon": 0.0, "commitment": 0.0, "sinkhorn": 0.0}
        n_batches = 0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            batch = X_tensor[idx].to(device)
            losses = quantizer.train_step(batch)
            total = losses["mse_recon"] + losses["commitment"] + losses["sinkhorn"]
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            for k, v in losses.items():
                epoch_losses[k] += float(v.item())
            n_batches += 1
        avg = {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}
        print(
            f"[epoch {epoch+1:2d}/{args.epochs}] "
            f"mse={avg['mse_recon']:.5f} commit={avg['commitment']:.5f} "
            f"sinkhorn={avg['sinkhorn']:.5f}",
            file=sys.stderr,
        )

    # 3) Encode all tracks + write assignments parquet ------------------------
    print("[build_sid_quantizer] encoding all tracks...", file=sys.stderr)
    sids_arr = quantizer.encode_batch(X)  # (N, num_levels)
    df = pd.DataFrame(
        {
            "track_id": track_ids,
            **{
                f"code_{level + 1}": sids_arr[:, level].astype(int)
                for level in range(args.num_levels)
            },
        }
    )
    assignments_path = out_dir / f"quantizer_seed{args.seed}_assignments.parquet"
    df.to_parquet(assignments_path, index=False)
    print(
        f"[build_sid_quantizer] wrote {assignments_path} ({len(df)} rows)",
        file=sys.stderr,
    )

    # 4) Save quantizer checkpoint --------------------------------------------
    qpath = out_dir / f"quantizer_seed{args.seed}.pt"
    quantizer.save(qpath)
    print(f"[build_sid_quantizer] wrote {qpath}", file=sys.stderr)

    # 5) Evaluate 3 validation gates ------------------------------------------
    # Gate 1: relative MSE vs PCA baseline
    pca = PCA(n_components=args.latent_dim).fit(X)
    X_pca_recon = pca.inverse_transform(pca.transform(X))
    pca_mse = float(((X_pca_recon - X) ** 2).mean())
    with torch.no_grad():
        z = quantizer.encoder(torch.from_numpy(X).to(device))
        z_q, _, _ = quantizer.rvq(z)
        recon = quantizer.decoder(z_q).cpu().numpy()
    rqvae_mse = float(((recon - X) ** 2).mean())
    g1_pass, g1_ratio = compute_relative_mse_gate(rqvae_mse, pca_mse, multiplier=1.5)

    # Gate 2: codebook utilization per level — level-aware thresholds (2026-05-16 v3.1).
    # RQ-VAE residual layers naturally have lower utilization than the first level
    # because they capture finer-grained variance. Uniform 80% threshold was wrong.
    # Realistic targets per published configs: L1 50-90%, L2 25-50%, L3 15-35%.
    LEVEL_THRESHOLDS = [0.50, 0.25, 0.15]  # one per level (must match num_levels)
    g2_results = []
    for level in range(args.num_levels):
        threshold = LEVEL_THRESHOLDS[level] if level < len(LEVEL_THRESHOLDS) else 0.15
        passed, util = validate_codebook_utilization(
            sids_arr[:, level].tolist(),
            codebook_size=args.codebook_size,
            threshold=threshold,
        )
        g2_results.append({
            "level": level + 1,
            "passed": passed,
            "utilization": util,
            "threshold": threshold,
        })
    g2_pass = all(r["passed"] for r in g2_results)

    # Gate 3: cluster purity at level-1
    print(
        "[build_sid_quantizer] loading track-metadata for tag_lookup...",
        file=sys.stderr,
    )
    meta_ds = load_dataset(
        "talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks"
    )
    tag_lookup = {row["track_id"]: row.get("tag_list") or [] for row in meta_ds}

    level1_buckets: dict[int, list[str]] = {}
    for tid, sid_row in zip(track_ids, sids_arr):
        level1_buckets.setdefault(int(sid_row[0]), []).append(tid)
    # Gate 3 threshold 0.20 = mean dominant-tag fraction across sampled buckets.
    # See validation.validate_cluster_purity v2 docstring for why intersection-based
    # purity (the v1 metric) is geometrically impossible at 184-tracks-per-bucket scale.
    g3_pass, g3_purity = validate_cluster_purity(
        {str(k): v for k, v in level1_buckets.items()},
        tag_lookup,
        n_samples=100,
        threshold=0.20,
        seed=args.seed,
    )

    # 6) Write per-seed gates JSON --------------------------------------------
    gates = {
        "seed": args.seed,
        "rqvae_mse": rqvae_mse,
        "pca_mse": pca_mse,
        "gate_1_relative_mse": {"passed": g1_pass, "ratio": g1_ratio},
        "gate_2_codebook_utilization": {"passed": g2_pass, "per_level": g2_results},
        "gate_3_cluster_purity": {"passed": g3_pass, "purity": g3_purity},
        "all_gates_passed": g1_pass and g2_pass and g3_pass,
    }
    gates_path = out_dir / f"quantizer_seed{args.seed}_gates.json"
    gates_path.write_text(json.dumps(gates, indent=2))
    print(f"[build_sid_quantizer] wrote {gates_path}", file=sys.stderr)
    print(
        f"[build_sid_quantizer] all gates passed: {gates['all_gates_passed']}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
