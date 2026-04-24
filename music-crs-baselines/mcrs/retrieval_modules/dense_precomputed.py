"""Dense retrieval with precomputed track embeddings + online query encoding.

v3 (tid=005). Loads a 1024-dim embedding per track from
`talkpl-ai/TalkPlayData-Challenge-Track-Embeddings` (column chosen by
`corpus_types[-1]`, default `attributes-qwen3_embedding_0.6b`). Query
embeddings are computed online with `Qwen/Qwen3-Embedding-0.6B` (~0.6B
params, bf16/fp16 on MPS). Scoring = cosine via L2-normalized dot
product, top-k per query via argpartition.

Interface matches `BM25_MODEL` so the factory in
`mcrs/retrieval_modules/__init__.py` can swap it in.

Notes
-----
- All track embeddings are L2-normalized once at load time; query
  embeddings are normalized per batch. After that, cosine = dot.
- Qwen3-Embedding uses last-token pooling with left padding (per the
  model's README). We implement left-padding manually via the
  tokenizer's `padding_side = "left"` config.
- The retriever caches the stacked `track_mat` + id list under
  `{cache_dir}/dense/{embed_col}/` to skip the HF→numpy pass on reruns.
"""
from __future__ import annotations

import os
import pickle
from typing import Optional

import numpy as np


DEFAULT_ENCODER = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_EMBED_COL = "attributes-qwen3_embedding_0.6b"
TRACK_EMB_DATASET = "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings"


class DENSE_PRECOMPUTED:
    def __init__(
        self,
        dataset_name: str,
        split_types: list[str],
        corpus_types: list[str],
        cache_dir: str = "./cache",
        embed_col: str = DEFAULT_EMBED_COL,
        instruct: Optional[str] = None,
        instruct_label: str = "raw",
    ) -> None:
        # corpus_types stays for signature parity (BM25_MODEL shape); the
        # dense retriever doesn't use it directly. embed_col selects the
        # track-embedding column on Challenge-Track-Embeddings; instruct is
        # an optional prefix applied to queries before encoding
        # (Qwen3-Embedding's asymmetric-retrieval pattern). instruct_label
        # is a short human id used only in the query-cache filename so
        # different prefix strategies cache to different files.
        self.embed_col = embed_col
        self.instruct = instruct
        self.instruct_label = instruct_label
        self.dataset_name = dataset_name  # unused; kept for interface parity
        self.split_types = split_types
        self.cache_dir = cache_dir
        self._encoder = None
        self._tokenizer = None
        self._device = None
        self.track_ids, self.track_mat = self._load_or_build_track_matrix()
        # Persistent query-embedding cache — shared across experiments that
        # reuse the same encoder AND the same instruct-wrapping strategy.
        self._query_cache_path = self._query_cache_file()
        self._query_cache: dict[str, np.ndarray] = self._load_query_cache()
        self._query_cache_dirty = False

    def _cache_path(self) -> str:
        safe = self.embed_col.replace("/", "_")
        # cache key includes imputation strategy so different strategies don't alias
        return os.path.join(self.cache_dir, "dense", safe, "imputed-artist-v1.pkl")

    def _load_or_build_track_matrix(self) -> tuple[list[str], np.ndarray]:
        cache_path = self._cache_path()
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                obj = pickle.load(f)
            return obj["track_ids"], obj["track_mat"]

        from datasets import concatenate_datasets, load_dataset

        print(f"[dense] loading {TRACK_EMB_DATASET}[{self.split_types}] column={self.embed_col}")
        ds = load_dataset(TRACK_EMB_DATASET)
        concat = concatenate_datasets([ds[s] for s in self.split_types])
        raw_ids = concat["track_id"]
        raw_embs = concat[self.embed_col]

        # Load artist_id per track for artist-mean imputation. Using the
        # metadata dataset passed to the factory as `dataset_name`.
        print(f"[dense] loading {self.dataset_name}[{self.split_types}] for artist_id lookup")
        meta_ds = load_dataset(self.dataset_name)
        meta_concat = concatenate_datasets([meta_ds[s] for s in self.split_types])
        artist_by_tid: dict[str, str] = {}
        for row in meta_concat:
            aid_list = row.get("artist_id") or []
            if aid_list:
                artist_by_tid[row["track_id"]] = aid_list[0]

        # Pass 1: collect valid rows, detect empties.
        expected_dim: Optional[int] = None
        valid_tids: list[str] = []
        valid_rows: list[list[float]] = []
        empty_tids: list[str] = []
        malformed = 0
        for tid, emb in zip(raw_ids, raw_embs):
            if emb is None or len(emb) == 0:
                empty_tids.append(tid)
                continue
            if expected_dim is None:
                expected_dim = len(emb)
            if len(emb) != expected_dim:
                malformed += 1
                continue
            valid_tids.append(tid)
            valid_rows.append(emb)

        if not valid_rows:
            raise RuntimeError(f"no valid embeddings in {self.embed_col}")

        valid_mat = np.asarray(valid_rows, dtype=np.float32)

        # Pass 2: build artist-level mean vectors from valid rows.
        artist_sums: dict[str, np.ndarray] = {}
        artist_counts: dict[str, int] = {}
        for tid, row in zip(valid_tids, valid_mat):
            aid = artist_by_tid.get(tid)
            if aid is None:
                continue
            if aid not in artist_sums:
                artist_sums[aid] = row.astype(np.float32).copy()
                artist_counts[aid] = 1
            else:
                artist_sums[aid] += row
                artist_counts[aid] += 1
        artist_means: dict[str, np.ndarray] = {
            aid: artist_sums[aid] / artist_counts[aid] for aid in artist_sums
        }
        global_mean = valid_mat.mean(axis=0).astype(np.float32)

        # Pass 3: impute empties with artist-mean → global-mean fallback.
        imputed_artist = 0
        imputed_global = 0
        imp_tids: list[str] = []
        imp_rows: list[np.ndarray] = []
        for tid in empty_tids:
            aid = artist_by_tid.get(tid)
            vec = artist_means.get(aid) if aid is not None else None
            if vec is None:
                vec = global_mean
                imputed_global += 1
            else:
                imputed_artist += 1
            imp_tids.append(tid)
            imp_rows.append(vec)

        kept_ids = valid_tids + imp_tids
        if imp_rows:
            kept_mat = np.vstack([valid_mat, np.asarray(imp_rows, dtype=np.float32)])
        else:
            kept_mat = valid_mat

        # L2-normalize so cosine == dot. Applies to both real and imputed rows.
        norms = np.linalg.norm(kept_mat, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-9)
        kept_mat = kept_mat / norms

        print(
            f"[dense] track_mat shape={kept_mat.shape} dtype={kept_mat.dtype} "
            f"(valid={len(valid_tids)}, imputed_artist={imputed_artist}, "
            f"imputed_global={imputed_global}, malformed_dropped={malformed})"
        )

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump({"track_ids": kept_ids, "track_mat": kept_mat}, f)
        return kept_ids, kept_mat

    def _get_encoder(self):
        if self._encoder is not None:
            return self._encoder, self._tokenizer
        import torch
        from transformers import AutoModel, AutoTokenizer

        # Qwen3-Embedding expects left-padding for last-token pooling.
        self._tokenizer = AutoTokenizer.from_pretrained(DEFAULT_ENCODER, padding_side="left")
        self._encoder = AutoModel.from_pretrained(DEFAULT_ENCODER, torch_dtype=torch.float32)
        if torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"
        self._encoder = self._encoder.to(self._device)
        self._encoder.eval()
        print(f"[dense] loaded {DEFAULT_ENCODER} on {self._device}")
        return self._encoder, self._tokenizer

    def _encode_queries(self, queries: list[str]) -> np.ndarray:
        import torch

        # Apply instruct prefix if configured (Qwen3-Embedding asymmetric pattern).
        if self.instruct:
            wrapped = [f"{self.instruct}{q}" for q in queries]
        else:
            wrapped = queries

        enc, tok = self._get_encoder()
        with torch.no_grad():
            batch = tok(
                wrapped,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(enc.device)
            out = enc(**batch)
            hidden = out.last_hidden_state  # (B, T, H)
            # Left-padded + last-token pooling: take the final position.
            pooled = hidden[:, -1, :]
            pooled = torch.nn.functional.normalize(pooled, dim=1)
            return pooled.detach().to("cpu").float().numpy()

    def _query_cache_file(self) -> str:
        """Per-(encoder, instruct-strategy) cache path."""
        safe = DEFAULT_ENCODER.replace("/", "_")
        return os.path.join(
            self.cache_dir, "dense", "query_embeddings",
            f"{safe}__{self.instruct_label}.pkl",
        )

    def _load_query_cache(self) -> dict[str, np.ndarray]:
        if os.path.exists(self._query_cache_path):
            try:
                with open(self._query_cache_path, "rb") as f:
                    cache = pickle.load(f)
                print(f"[dense] loaded query cache: {len(cache)} entries from {self._query_cache_path}")
                return cache
            except Exception as e:
                print(f"[dense] failed to load query cache ({e}); starting empty")
        return {}

    def _save_query_cache(self) -> None:
        if not self._query_cache_dirty:
            return
        os.makedirs(os.path.dirname(self._query_cache_path), exist_ok=True)
        tmp = self._query_cache_path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(self._query_cache, f)
        os.replace(tmp, self._query_cache_path)
        self._query_cache_dirty = False

    def batch_text_to_item_retrieval(self, queries: list[str], topk: int) -> list[list[str]]:
        # 1. Identify which queries are already cached.
        missing_idx = [i for i, q in enumerate(queries) if q not in self._query_cache]
        hits = len(queries) - len(missing_idx)
        if missing_idx:
            to_encode = [queries[i] for i in missing_idx]
            # Encode in sub-batches to bound MPS memory.
            SUB_B = 32
            encoded_parts: list[np.ndarray] = []
            for i in range(0, len(to_encode), SUB_B):
                encoded_parts.append(self._encode_queries(to_encode[i:i + SUB_B]))
            encoded = np.concatenate(encoded_parts, axis=0) if encoded_parts else np.zeros((0, self.track_mat.shape[1]), dtype=np.float32)
            for j, q in enumerate(to_encode):
                self._query_cache[q] = encoded[j].astype(np.float32)
            self._query_cache_dirty = True
            self._save_query_cache()
        if hits and queries:
            print(f"[dense] query cache hit/total = {hits}/{len(queries)}")

        # 2. Assemble query matrix from the (now complete) cache.
        q_mat = np.stack([self._query_cache[q] for q in queries], axis=0).astype(np.float32)

        # 3. Score all queries × all tracks.
        scores = q_mat @ self.track_mat.T  # (N, T)
        results: list[list[str]] = []
        for i in range(scores.shape[0]):
            row = scores[i]
            if topk >= row.shape[0]:
                order = np.argsort(-row)
            else:
                cand = np.argpartition(-row, topk)[:topk]
                order = cand[np.argsort(-row[cand])]
            results.append([self.track_ids[idx] for idx in order])
        return results

    def text_to_item_retrieval(self, query: str, topk: int) -> list[str]:
        return self.batch_text_to_item_retrieval([query], topk=topk)[0]
