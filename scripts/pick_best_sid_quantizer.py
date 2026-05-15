"""Pick the best of 3 trained SID quantizers (per seed), pin by SHA256.

Reads quantizer_seed{42,123,7}_gates.json, picks the seed whose quantizer:
  1. Passes all 3 gates, AND
  2. Has the highest gate_3_cluster_purity score (per spec §2.3).

Then writes:
  - quantizer_chosen.pt (copy of chosen seed's checkpoint)
  - track_to_sid.parquet (chosen seed's assignments + collision bucket ranks)
  - quantizer_chosen.sha256 (pinning hash for reproducibility)

Usage:
    python scripts/pick_best_sid_quantizer.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINES_DIR = REPO_ROOT / "music-crs-baselines"
sys.path.insert(0, str(BASELINES_DIR))

import pandas as pd
from datasets import load_dataset


def sha256_of_file(path: Path) -> str:
    """Streaming SHA256 of file contents (memory-bounded for large files)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    cache_root = REPO_ROOT / "experiments" / "cache" / "sid"
    seeds = [42, 123, 7]

    candidates: list[dict] = []
    for s in seeds:
        gates_path = cache_root / f"quantizer_seed{s}_gates.json"
        if not gates_path.exists():
            print(f"missing {gates_path} -- skipping seed {s}", file=sys.stderr)
            continue
        gates = json.loads(gates_path.read_text())
        if not gates.get("all_gates_passed", False):
            print(f"seed {s} did not pass all gates", file=sys.stderr)
            continue
        candidates.append({
            "seed": s,
            "purity": gates["gate_3_cluster_purity"]["purity"],
            "gates": gates,
        })

    if not candidates:
        raise SystemExit("No seed passed all 3 validation gates. Debug quantizer config.")

    # Pick highest purity among gate-passing candidates
    best = max(candidates, key=lambda c: c["purity"])
    print(
        f"chosen: seed={best['seed']} purity={best['purity']:.3f}",
        file=sys.stderr,
    )

    # Copy chosen artifact + SHA256-pin
    src_q = cache_root / f"quantizer_seed{best['seed']}.pt"
    chosen_q = cache_root / "quantizer_chosen.pt"
    shutil.copy(src_q, chosen_q)
    sha = sha256_of_file(chosen_q)
    (cache_root / "quantizer_chosen.sha256").write_text(sha + "\n")

    # Build track_to_sid.parquet with collision buckets + popularity ordering
    src_assign = cache_root / f"quantizer_seed{best['seed']}_assignments.parquet"
    df = pd.read_parquet(src_assign)

    print("[pick_best] loading metadata for popularity ordering...", file=sys.stderr)
    meta_ds = load_dataset(
        "talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks"
    )
    popularity = {row["track_id"]: row.get("popularity") or 0.0 for row in meta_ds}
    df["popularity"] = df["track_id"].map(popularity).fillna(0.0)

    # Sort by SID tuple then popularity (descending), assign bucket_rank (0=most popular per bucket)
    df["sid_tuple"] = list(zip(df["code_1"], df["code_2"], df["code_3"]))
    df = df.sort_values(["sid_tuple", "popularity"], ascending=[True, False]).reset_index(drop=True)
    df["bucket_rank"] = df.groupby("sid_tuple").cumcount()

    out = df[["track_id", "code_1", "code_2", "code_3", "popularity", "bucket_rank"]]
    out_path = cache_root / "track_to_sid.parquet"
    out.to_parquet(out_path, index=False)
    print(f"[pick_best] wrote {out_path} ({len(out)} rows)", file=sys.stderr)

    # Summary stats
    n_buckets = df["sid_tuple"].nunique()
    n_collisions = (df.groupby("sid_tuple").size() > 1).sum()
    print(
        f"[pick_best] {n_buckets} unique SIDs, {n_collisions} collision buckets, "
        f"chosen sha256={sha[:12]}...",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
