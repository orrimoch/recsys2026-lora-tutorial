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


def resolve_st_dtype(device: str, override: str = "auto"):
    """torch dtype for loading a SentenceTransformer encoder.

    Default (`auto`): fp16 on CUDA, fp32 elsewhere. fp16 ~halves weight memory and
    roughly doubles throughput — the difference between a 4B encoder fitting a 16 GB
    T4/G4 vs OOM-ing in fp32. fp16 (not bf16) is deliberate: Turing/T4 has no bf16.
    CPU/MPS stay fp32 (fp16 matmul is slow/unsupported there). `override` (e.g.
    'float16', 'bfloat16', 'float32') forces a specific dtype on any device.
    """
    import torch
    if override and override != "auto":
        return getattr(torch, override)
    return torch.float16 if device == "cuda" else torch.float32


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
        # Throttled persist — see dense_precomputed for the full rationale: a
        # per-batch full re-pickle of the growing cache is O(n^2) writes and, on
        # a Drive-mounted cache_dir, fills the disk. Save every _cache_save_every
        # new entries + a forced flush at process exit.
        self._cache_save_every = 2000
        self._cache_entries_since_save = 0
        import atexit
        atexit.register(self.flush_query_cache)

    def flush_query_cache(self) -> None:
        """Force-write the query cache if anything is unsaved (atexit + callable
        explicitly at the end of a retrieval loop)."""
        if self._query_cache_dirty:
            self._save_query_cache()
            self._cache_entries_since_save = 0

    def _maybe_save_query_cache(self, n_new_entries: int) -> None:
        """Throttled persist: rewrite the pickle only after enough new entries."""
        self._cache_entries_since_save += n_new_entries
        if self._cache_entries_since_save >= self._cache_save_every:
            self._save_query_cache()
            self._cache_entries_since_save = 0

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
        dtype = resolve_st_dtype(device)
        model = SentenceTransformer(self.model_name, device=device,
                                    model_kwargs={"torch_dtype": dtype})
        _SHARED_ENCODER[key] = model
        print(f"[dense_local] loaded encoder {self.model_name} on {device} ({dtype})")
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
            self._maybe_save_query_cache(len(to_encode))
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
