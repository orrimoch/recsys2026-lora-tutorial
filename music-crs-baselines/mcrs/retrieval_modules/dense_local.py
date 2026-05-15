"""Dense retrieval over locally-computed catalog embeddings + sentence-transformers query encoding.

Sibling of `dense_precomputed.py`. Use this when the embedding model isn't
covered by the challenge-provided HF dataset columns (e.g., BGE-M3,
Qwen3-Embedding-4B, custom fine-tuned models).

Catalog embeddings must be precomputed via `scripts/embed_catalog.py` and
saved as a pickle at `{cache_dir}/dense_local/{safe_model}/{embed_label}/track_embeddings.pkl`
with the shape `{"track_ids": list[str], "track_mat": np.ndarray (N, dim)}`.
The retriever loads them at init and uses sentence-transformers for query
encoding (which handles model-specific tokenization + pooling automatically).

Interface matches `BM25_MODEL` / `DENSE_PRECOMPUTED` so the factory can swap
it in transparently.
"""
from __future__ import annotations

import os
import pickle
from typing import Optional

import numpy as np


# Per-(model, instruct_label) shared encoder + per-query caches across instances.
# Mirrors the dense_precomputed pattern so multiple wRRF retrievers using the
# same model don't re-load weights or re-encode the same queries.
_SHARED_ENCODER: dict[tuple, object] = {}
_SHARED_QUERY_CACHE: dict[tuple, dict[str, np.ndarray]] = {}


class DENSE_LOCAL:
    def __init__(
        self,
        dataset_name: str,
        split_types: list[str],
        corpus_types: list[str],
        cache_dir: str = "./cache",
        model_name: str = "BAAI/bge-m3",
        embed_label: str = "default",
        instruct: Optional[str] = None,
        instruct_label: str = "raw",
    ) -> None:
        # dataset_name + split_types + corpus_types kept for interface parity
        # (factory passes them); only used to assert the cache file exists.
        self.dataset_name = dataset_name
        self.split_types = split_types
        self.cache_dir = cache_dir
        self.model_name = model_name
        self.embed_label = embed_label
        self.instruct = instruct
        self.instruct_label = instruct_label

        self.track_ids, self.track_mat = self._load_track_matrix()

        self._query_cache_path = self._query_cache_file()
        self._query_cache_key = (self.model_name, self.instruct_label)
        if self._query_cache_key not in _SHARED_QUERY_CACHE:
            _SHARED_QUERY_CACHE[self._query_cache_key] = self._load_query_cache()
        self._query_cache: dict[str, np.ndarray] = _SHARED_QUERY_CACHE[self._query_cache_key]
        self._query_cache_dirty = False

    # ---- catalog embeddings ----

    def _track_cache_path(self) -> str:
        safe_model = self.model_name.replace("/", "_")
        return os.path.join(
            self.cache_dir, "dense_local", safe_model, self.embed_label, "track_embeddings.pkl"
        )

    def _load_track_matrix(self) -> tuple[list[str], np.ndarray]:
        path = self._track_cache_path()
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"DENSE_LOCAL: missing catalog embeddings at {path}\n"
                f"Run: python scripts/embed_catalog.py --model {self.model_name} "
                f"--label {self.embed_label}"
            )
        with open(path, "rb") as f:
            obj = pickle.load(f)
        track_mat = obj["track_mat"].astype(np.float32)
        # Re-normalize to be safe (embed_catalog.py already does this).
        norms = np.linalg.norm(track_mat, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-9)
        track_mat = track_mat / norms
        print(
            f"[dense_local] loaded catalog embeddings from {path} "
            f"(N={len(obj['track_ids'])}, dim={track_mat.shape[1]})"
        )
        return obj["track_ids"], track_mat

    # ---- query encoder (sentence-transformers) ----

    def _get_encoder(self):
        key = (self.model_name, self.instruct_label)
        if key in _SHARED_ENCODER:
            return _SHARED_ENCODER[key]
        import torch
        from sentence_transformers import SentenceTransformer

        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        model = SentenceTransformer(self.model_name, device=device)
        _SHARED_ENCODER[key] = model
        print(f"[dense_local] loaded encoder {self.model_name} on {device}")
        return model

    def _encode_queries(self, queries: list[str]) -> np.ndarray:
        if self.instruct:
            wrapped = [f"{self.instruct}{q}" for q in queries]
        else:
            wrapped = queries
        model = self._get_encoder()
        embeddings = model.encode(
            wrapped,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        return embeddings

    # ---- query cache (shared across instances using same model) ----

    def _query_cache_file(self) -> str:
        safe_model = self.model_name.replace("/", "_")
        return os.path.join(
            self.cache_dir, "dense_local", safe_model, "query_cache",
            f"{self.instruct_label}.pkl",
        )

    def _load_query_cache(self) -> dict[str, np.ndarray]:
        if os.path.exists(self._query_cache_path):
            try:
                with open(self._query_cache_path, "rb") as f:
                    cache = pickle.load(f)
                print(f"[dense_local] loaded query cache: {len(cache)} entries from {self._query_cache_path}")
                return cache
            except Exception as e:
                print(f"[dense_local] failed to load query cache ({e}); starting empty")
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

    # ---- public retrieval interface ----

    def batch_text_to_item_retrieval(
        self, queries: list[str], topk: int, user_ids=None,
    ) -> list[list[str]]:
        # user_ids accepted for interface parity (ignored — query-only retrieval).
        missing_idx = [i for i, q in enumerate(queries) if q not in self._query_cache]
        hits = len(queries) - len(missing_idx)
        if missing_idx:
            to_encode = [queries[i] for i in missing_idx]
            encoded = self._encode_queries(to_encode)
            for j, q in enumerate(to_encode):
                self._query_cache[q] = encoded[j].astype(np.float32)
            self._query_cache_dirty = True
            self._save_query_cache()
        if hits and queries:
            print(f"[dense_local] query cache hit/total = {hits}/{len(queries)}")

        q_mat = np.stack([self._query_cache[q] for q in queries], axis=0).astype(np.float32)
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
