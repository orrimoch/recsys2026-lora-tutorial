"""Precompute shared multi-modal artifacts for the fresh-model upgrade.

One-time builder that materializes everything the multi-modal bi-encoder
(Stage A) and multi-modal cross-encoder (Stage B) consume at train and
inference time. Idempotent: each artifact is built only if missing.

Outputs under ``--cache-dir`` (default ``experiments/cache/multimodal/``):

  tag_vocab.json            {tag_str: int_id}            ~ few KB
  release_year_lookup.json  {track_id: int_year_or_-1}   ~ 1 MB
  track_clap.npy            float32 (N_tracks, 512)      ~ 96 MB (L2-normed)
  track_clap_tids.json      [track_id, ...]              ~ 2 MB
  track_cf.npy              float32 (N_tracks, 128)      ~ 24 MB (L2-normed)
  track_cf_tids.json        [track_id, ...]              ~ 2 MB
  user_cf.npy               float32 (N_users, 128)       ~ 5-50 MB (L2-normed)
  user_cf_uids.json         [user_id, ...]
  user_cf_mean.npy          float32 (128,)               cold-user fallback

Sources:
  - tag_vocab + release_year_lookup: ``talkpl-ai/TalkPlayData-Challenge-Track-Metadata``
  - CLAP audio embeddings: ``talkpl-ai/TalkPlayData-Challenge-Track-Embeddings`` col ``audio-laion_clap``
  - CF-BPR track embeddings: same dataset, col ``cf-bpr``
  - CF-BPR user embeddings: ``talkpl-ai/TalkPlayData-Challenge-User-Embeddings`` col ``cf-bpr``

Empty embedding rows are dropped (same convention as
``mcrs/retrieval_modules/cf_bpr.py``).

Usage:
  python scripts/precompute_multimodal_artifacts.py
  python scripts/precompute_multimodal_artifacts.py --cache-dir /content/drive/MyDrive/.../multimodal
  python scripts/precompute_multimodal_artifacts.py --force          # rebuild all
  python scripts/precompute_multimodal_artifacts.py --skip clap cf   # skip specific artifacts
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np


METADATA_DATASET = "talkpl-ai/TalkPlayData-Challenge-Track-Metadata"
TRACK_EMB_DATASET = "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings"
USER_EMB_DATASET = "talkpl-ai/TalkPlayData-Challenge-User-Embeddings"
CLAP_COL = "audio-laion_clap"
CF_COL = "cf-bpr"
YEAR_RE = re.compile(r"(\d{4})")


def _l2_normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.maximum(norms, 1e-9)


def _parse_year(raw) -> int:
    """Extract a 4-digit year from a release_date field. Returns -1 if absent."""
    if raw is None:
        return -1
    s = str(raw).strip()
    if not s:
        return -1
    m = YEAR_RE.search(s)
    if not m:
        return -1
    y = int(m.group(1))
    return y if 1900 <= y <= 2100 else -1


def _exists(path: str | Path) -> bool:
    return os.path.exists(str(path))


def build_metadata_artifacts(out_dir: Path, force: bool) -> None:
    tag_path = out_dir / "tag_vocab.json"
    year_path = out_dir / "release_year_lookup.json"
    if not force and _exists(tag_path) and _exists(year_path):
        print(f"[meta] skip — {tag_path.name} + {year_path.name} already exist")
        return

    from datasets import concatenate_datasets, load_dataset

    print(f"[meta] loading {METADATA_DATASET}")
    ds = load_dataset(METADATA_DATASET)
    rows = concatenate_datasets([ds[s] for s in ds])

    tag_set: set[str] = set()
    year_lookup: dict[str, int] = {}
    for r in rows:
        tid = r.get("track_id")
        if tid is None:
            continue
        # tag_list may be list[str] or None
        tags = r.get("tag_list") or []
        if isinstance(tags, str):
            tags = [tags]
        for t in tags:
            if t:
                tag_set.add(str(t).strip().lower())
        year_lookup[tid] = _parse_year(r.get("release_date"))

    # Reserve id 0 for PAD; sort tags for reproducibility.
    sorted_tags = sorted(tag_set)
    tag_vocab = {"<pad>": 0}
    for i, t in enumerate(sorted_tags, start=1):
        tag_vocab[t] = i

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(tag_path, "w") as f:
        json.dump(tag_vocab, f, indent=2, sort_keys=True)
    with open(year_path, "w") as f:
        json.dump(year_lookup, f)
    n_years_known = sum(1 for v in year_lookup.values() if v > 0)
    print(
        f"[meta] wrote tag_vocab ({len(tag_vocab)} entries incl <pad>) + "
        f"release_year_lookup ({len(year_lookup)} tracks, "
        f"{n_years_known} with known year)"
    )


def build_track_embeddings(
    embed_col: str,
    out_npy: Path,
    out_tids: Path,
    expected_dim: int,
    force: bool,
) -> None:
    if not force and _exists(out_npy) and _exists(out_tids):
        mat = np.load(out_npy, mmap_mode="r")
        print(f"[{embed_col}] skip — {out_npy.name} exists (shape={tuple(mat.shape)})")
        return

    from datasets import concatenate_datasets, load_dataset

    print(f"[{embed_col}] loading {TRACK_EMB_DATASET}")
    ds = load_dataset(TRACK_EMB_DATASET)
    rows = concatenate_datasets([ds[s] for s in ds])

    tids: list[str] = []
    vecs: list[np.ndarray] = []
    empty = 0
    for r in rows:
        emb = r.get(embed_col)
        if not emb:
            empty += 1
            continue
        v = np.asarray(emb, dtype=np.float32)
        if v.shape[0] != expected_dim:
            raise ValueError(
                f"{embed_col}: track {r.get('track_id')} has dim={v.shape[0]} "
                f"expected={expected_dim}"
            )
        tids.append(r["track_id"])
        vecs.append(v)

    mat = np.stack(vecs, axis=0)
    mat = _l2_normalize(mat)
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, mat)
    with open(out_tids, "w") as f:
        json.dump(tids, f)
    print(
        f"[{embed_col}] wrote {out_npy.name} shape={tuple(mat.shape)} "
        f"dropped_empty_rows={empty}"
    )


def build_user_embeddings(out_dir: Path, force: bool) -> None:
    out_npy = out_dir / "user_cf.npy"
    out_uids = out_dir / "user_cf_uids.json"
    out_mean = out_dir / "user_cf_mean.npy"
    if not force and _exists(out_npy) and _exists(out_uids) and _exists(out_mean):
        mat = np.load(out_npy, mmap_mode="r")
        print(f"[user-cf] skip — {out_npy.name} exists (shape={tuple(mat.shape)})")
        return

    from datasets import concatenate_datasets, load_dataset

    print(f"[user-cf] loading {USER_EMB_DATASET}")
    ds = load_dataset(USER_EMB_DATASET)
    rows = concatenate_datasets([ds[s] for s in ds])

    uids: list[str] = []
    vecs: list[np.ndarray] = []
    empty = 0
    for r in rows:
        emb = r.get(CF_COL)
        if not emb:
            empty += 1
            continue
        v = np.asarray(emb, dtype=np.float32)
        uids.append(r["user_id"])
        vecs.append(v)

    mat = np.stack(vecs, axis=0)
    mat = _l2_normalize(mat)
    # Cold-user fallback: mean of normalized vectors, re-normalized.
    # Re-norm so the fallback lives on the unit sphere like every other user.
    mean = mat.mean(axis=0)
    mean = mean / max(float(np.linalg.norm(mean)), 1e-9)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, mat)
    np.save(out_mean, mean)
    with open(out_uids, "w") as f:
        json.dump(uids, f)
    print(
        f"[user-cf] wrote user_cf.npy shape={tuple(mat.shape)} + user_cf_mean.npy "
        f"dropped_empty_rows={empty}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cache-dir",
        default="experiments/cache/multimodal",
        help="Where to write artifacts. Defaults to the repo-local cache dir.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Rebuild all artifacts even if they exist.",
    )
    p.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=["meta", "clap", "track_cf", "user_cf"],
        help="Artifact groups to skip.",
    )
    args = p.parse_args()

    out_dir = Path(args.cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[precompute-multimodal] writing to {out_dir.resolve()}")

    if "meta" not in args.skip:
        build_metadata_artifacts(out_dir, args.force)
    if "clap" not in args.skip:
        build_track_embeddings(
            embed_col=CLAP_COL,
            out_npy=out_dir / "track_clap.npy",
            out_tids=out_dir / "track_clap_tids.json",
            expected_dim=512,
            force=args.force,
        )
    if "track_cf" not in args.skip:
        build_track_embeddings(
            embed_col=CF_COL,
            out_npy=out_dir / "track_cf.npy",
            out_tids=out_dir / "track_cf_tids.json",
            expected_dim=128,
            force=args.force,
        )
    if "user_cf" not in args.skip:
        build_user_embeddings(out_dir, args.force)

    print("[precompute-multimodal] done")


if __name__ == "__main__":
    main()
