"""Multi-modal catalog re-embed: encode the 47K-track catalog with the
fused MultiModalBiEncoder (text + CLAP + CF + tag_ids + release_year).

Sibling of ``scripts/embed_catalog.py`` — same output layout
(``{cache_root}/<safe_model>/<label>/track_embeddings.pkl``) so the
downstream ``DENSE_MULTIMODAL_LOCAL`` (Phase 3b) finds embeddings at the
exact path the production wRRF factory expects.

Differences from ``embed_catalog.py``:
  - Loads ``MultiModalBiEncoder`` instead of a plain ``SentenceTransformer``.
  - Looks up per-track CLAP + CF embeddings from the multi-modal artifact
    cache (built by ``scripts/precompute_multimodal_artifacts.py``).
  - Tokenizes the text portion with the backbone's tokenizer + calls
    ``model.forward_track`` directly (the sentence-transformers wrapper
    would lose modality awareness).
  - Output embeddings have dim = ``model.config.hidden_dim`` (default 768),
    same as text-only output → existing scoring code path is unchanged.

Usage:
  python scripts/embed_catalog_multimodal.py \\
      --model-dir /content/drive/.../training/.../merged \\
      --backbone BAAI/bge-base-en-v1.5 \\
      --multimodal-artifacts experiments/cache/multimodal \\
      --catalog-out-dir /content/drive/.../dense_local/<safe>/<label> \\
      --batch-size 64

The ``--model-dir`` is the multi-modal model save dir (contains
``backbone/`` + ``modality_heads.pt`` + ``multimodal_config.json``).
The ``--catalog-out-dir`` matches the cache layout DENSE_LOCAL /
DENSE_MULTIMODAL_LOCAL read at runtime.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "music-crs-baselines"))

import numpy as np


def _format_track_text(track_id, metadata_dict, corpus_types):
    """Mirror ``MusicCatalogDB.id_to_metadata`` so the text fed to the
    model at inference matches the format used at training time (PATCH 4
    alignment from 2026-05-22)."""
    if track_id not in metadata_dict:
        return track_id
    md = metadata_dict[track_id]
    parts = [f"track_id: {track_id}"]
    for ct in corpus_types:
        val = md.get(ct)
        if val is None:
            joined = ""
        elif isinstance(val, list):
            joined = ", ".join(str(v) for v in val)
        else:
            joined = str(val)
        parts.append(f"{ct}: {joined.lower()}")
    return ", ".join(parts)


def _track_to_tag_ids(track_id, metadata_dict, tag_vocab, max_tags=20):
    """Mirror ``scripts/build_bi_encoder_training_data.py:_track_to_tag_ids``."""
    if track_id not in metadata_dict:
        return []
    tags = metadata_dict[track_id].get("tag_list") or []
    if isinstance(tags, str):
        tags = [tags]
    ids = []
    for t in tags[:max_tags]:
        norm = str(t).strip().lower()
        tid = tag_vocab.get(norm)
        if tid is not None:
            ids.append(tid)
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir", required=True,
        help="Path to a MultiModalBiEncoder.save_pretrained() output. "
             "Must contain backbone/ + modality_heads.pt + multimodal_config.json.",
    )
    parser.add_argument(
        "--backbone", default="",
        help="Optional override for the base model (passed to from_pretrained). "
             "Defaults to the backbone recorded in multimodal_config.json.",
    )
    parser.add_argument(
        "--multimodal-artifacts", required=True,
        help="Path to the precompute cache dir (tag_vocab.json, track_clap.npy, "
             "track_cf.npy + per-tid mappings). Same dir used at training time.",
    )
    parser.add_argument(
        "--catalog-out-dir", required=True,
        help="Where to write track_embeddings.pkl. Must match the path "
             "DENSE_MULTIMODAL_LOCAL constructs from (cache_dir, model, label).",
    )
    parser.add_argument(
        "--catalog-dataset", default="talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
    )
    parser.add_argument("--catalog-split", default="all_tracks")
    parser.add_argument(
        "--corpus-types", default="track_name,artist_name,album_name",
        help="Comma-separated catalog fields fed to the text tower. Defaults "
             "match the training builder + production config 021 (closes the "
             "train/inference format gap surfaced by the BlindA nDCG diagnosis).",
    )
    parser.add_argument("--max-tags", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--passage-max-len", type=int, default=192,
        help="Tokenizer max_length for the text tower (must match training's "
             "PASSAGE_MAX_LEN to keep the input distribution aligned).",
    )
    parser.add_argument("--max-tracks", type=int, default=0,
                        help="Smoke cap; 0 = full catalog.")
    parser.add_argument(
        "--ablate-modality", default="",
        choices=["", "audio", "cf", "tag", "release"],
        help="If set, zero ONE track-side modality during forward_track. "
             "Used by the Phase 5 ablation cell to measure each modality's "
             "contribution to dev nDCG. Run with a fresh --catalog-out-dir "
             "to avoid overwriting the baseline catalog.",
    )
    args = parser.parse_args()

    # Lazy imports (heavy CUDA init).
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from mcrs.training.multimodal_bi_encoder import MultiModalBiEncoder

    # ---- 1. Load multi-modal model + tokenizer ----
    print(f"[embed-mm] loading multi-modal model from {args.model_dir}",
          file=sys.stderr)
    backbone_override = args.backbone or None
    model = MultiModalBiEncoder.from_pretrained(args.model_dir, backbone_override=backbone_override)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model = model.to(device).eval()
    print(f"[embed-mm] model on {device}, hidden_dim={model.config.hidden_dim}",
          file=sys.stderr)

    tok_source = backbone_override or model.config.backbone_name
    tokenizer = AutoTokenizer.from_pretrained(tok_source)
    tokenizer.truncation_side = "right"

    # ---- 2. Load multi-modal artifacts ----
    mm_dir = Path(args.multimodal_artifacts)
    with open(mm_dir / "tag_vocab.json", "r") as f:
        tag_vocab = json.load(f)
    with open(mm_dir / "release_year_lookup.json", "r") as f:
        release_year_lookup = json.load(f)
    track_clap = np.load(mm_dir / "track_clap.npy", mmap_mode="r")
    with open(mm_dir / "track_clap_tids.json", "r") as f:
        clap_tids = json.load(f)
    clap_idx = {t: i for i, t in enumerate(clap_tids)}
    track_cf = np.load(mm_dir / "track_cf.npy", mmap_mode="r")
    with open(mm_dir / "track_cf_tids.json", "r") as f:
        cf_tids = json.load(f)
    cf_idx = {t: i for i, t in enumerate(cf_tids)}
    print(f"[embed-mm] artifacts: tag_vocab={len(tag_vocab)}, "
          f"CLAP={len(clap_tids)} tracks, CF={len(cf_tids)} tracks",
          file=sys.stderr)

    # ---- 3. Load catalog ----
    print(f"[embed-mm] loading catalog {args.catalog_dataset}[{args.catalog_split}]",
          file=sys.stderr)
    ds = load_dataset(args.catalog_dataset, split=args.catalog_split)
    if args.max_tracks > 0:
        ds = ds.select(range(min(args.max_tracks, len(ds))))
    metadata_dict = {row["track_id"]: dict(row) for row in ds}
    catalog_tids = [row["track_id"] for row in ds]
    corpus_types = [c.strip() for c in args.corpus_types.split(",") if c.strip()]
    print(f"[embed-mm] catalog: {len(catalog_tids)} tracks", file=sys.stderr)

    # Pre-materialize per-track modality lookups (avoids dict hits per batch).
    zero_clap = np.zeros(int(track_clap.shape[1]), dtype=np.float32)
    zero_cf = np.zeros(int(track_cf.shape[1]), dtype=np.float32)

    # ---- 4. Encode in batches ----
    out_embs = np.zeros((len(catalog_tids), model.config.hidden_dim), dtype=np.float32)
    from tqdm import tqdm
    pbar = tqdm(total=len(catalog_tids), desc="[embed-mm]")
    n_miss_clap = n_miss_cf = 0
    with torch.no_grad():
        for i in range(0, len(catalog_tids), args.batch_size):
            batch_tids = catalog_tids[i: i + args.batch_size]
            B = len(batch_tids)

            texts = [_format_track_text(t, metadata_dict, corpus_types) for t in batch_tids]
            clap_rows = []
            cf_rows = []
            tag_rows = []
            year_rows = []
            for t in batch_tids:
                ci = clap_idx.get(t)
                if ci is None:
                    n_miss_clap += 1
                    clap_rows.append(zero_clap)
                else:
                    clap_rows.append(np.asarray(track_clap[ci], dtype=np.float32))
                ki = cf_idx.get(t)
                if ki is None:
                    n_miss_cf += 1
                    cf_rows.append(zero_cf)
                else:
                    cf_rows.append(np.asarray(track_cf[ki], dtype=np.float32))
                tag_rows.append(_track_to_tag_ids(t, metadata_dict, tag_vocab,
                                                  max_tags=args.max_tags))
                year_rows.append(int(release_year_lookup.get(t, -1)))

            enc = tokenizer(
                texts, max_length=args.passage_max_len,
                padding=True, truncation=True, return_tensors="pt",
            ).to(device)
            clap_t = torch.from_numpy(np.stack(clap_rows, axis=0)).to(device)
            cf_t = torch.from_numpy(np.stack(cf_rows, axis=0)).to(device)
            year_t = torch.tensor(year_rows, dtype=torch.long, device=device)
            # Pad tag_ids to max_tags
            tag_mat = np.zeros((B, args.max_tags), dtype=np.int64)
            for r, ids in enumerate(tag_rows):
                ids = list(ids)[: args.max_tags]
                tag_mat[r, : len(ids)] = ids
            tag_t = torch.from_numpy(tag_mat).to(device)

            # Phase 5: optional modality ablation. Pass a modality_mask that
            # zeros the chosen token across the batch. The model handles the
            # mask via MultiModalBiEncoder.forward_track's existing pathway.
            _mm_mask = None
            if args.ablate_modality:
                _mm_mask = {args.ablate_modality: torch.zeros(B, 1, device=device)}
            embs = model.forward_track(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                clap=clap_t, cf_track=cf_t, tag_ids=tag_t, year=year_t,
                modality_mask=_mm_mask,
            )
            out_embs[i: i + B] = embs.float().cpu().numpy()
            pbar.update(B)
    pbar.close()

    # Defensive re-normalize (model already L2-norms, but cheap to guard).
    norms = np.linalg.norm(out_embs, axis=1, keepdims=True)
    out_embs = out_embs / np.maximum(norms, 1e-9)

    out_dir = Path(args.catalog_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "track_embeddings.pkl"
    with out_path.open("wb") as f:
        pickle.dump({"track_ids": catalog_tids, "track_mat": out_embs}, f)
    print(
        f"[embed-mm] wrote {out_path} shape={out_embs.shape} "
        f"missing_clap={n_miss_clap} missing_cf={n_miss_cf}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
