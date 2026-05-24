"""Multi-modal dense retrieval — sibling of ``dense_local.py`` for the
``fresh-model`` branch's user-aware bi-encoder.

Catalog side: identical to ``DENSE_LOCAL`` — loads a precomputed pickle
``{cache_dir}/dense_local/{safe_model}/{embed_label}/track_embeddings.pkl``
written by ``scripts/embed_catalog_multimodal.py``. The pickle format is
the same (``{"track_ids": list[str], "track_mat": np.ndarray (N, dim)}``),
so the scoring code path (``q_mat @ track_mat.T``) is unchanged.

Query side: differs from ``DENSE_LOCAL`` in two material ways:

  1. **User-aware encoding**: the query embedding depends on BOTH the query
     text AND the user's CF-BPR vector. Cold users get the train-set mean
     fallback (matching the training-time distribution).
  2. **Cache keyed by (query_text, user_id_or_None)**: the same query
     produces DIFFERENT embeddings for different users (because the
     ``<user_cf>`` pseudo-token differs). Caching by text alone would
     return wrong embeddings for any subsequent user.

Interface stays compatible with the factory's
``batch_text_to_item_retrieval(queries, topk, user_ids=...)`` signature —
production code that currently ignores ``user_ids`` for ``DENSE_LOCAL``
will now actually use it when swapped to ``DENSE_MULTIMODAL_LOCAL``.
"""
from __future__ import annotations

import json
import os
import pickle
from typing import Optional

import numpy as np


# Per-(model_dir, backbone) shared encoder + per-query caches across instances.
# Mirrors dense_local.py to avoid double-loading the multi-modal model when
# multiple wRRF retrievers share it.
_SHARED_MODEL: dict[tuple, object] = {}
_SHARED_TOKENIZER: dict[tuple, object] = {}
_SHARED_QUERY_CACHE: dict[tuple, dict[tuple, np.ndarray]] = {}
_SHARED_USER_CF: dict[str, dict] = {}  # keyed by multimodal_artifacts path


def _load_user_cf_lookup(mm_artifacts_dir: str) -> dict:
    """Memoized load of user_cf.npy + uids + mean fallback. Mirrors
    ``mcrs/retrieval_modules/cf_bpr.py:_load_users`` pattern."""
    if mm_artifacts_dir in _SHARED_USER_CF:
        return _SHARED_USER_CF[mm_artifacts_dir]
    user_cf = np.load(os.path.join(mm_artifacts_dir, "user_cf.npy"))
    with open(os.path.join(mm_artifacts_dir, "user_cf_uids.json"), "r") as f:
        uids = json.load(f)
    mean = np.load(os.path.join(mm_artifacts_dir, "user_cf_mean.npy"))
    out = {
        "matrix": user_cf,
        "uid_to_idx": {u: i for i, u in enumerate(uids)},
        "mean": mean.astype(np.float32),
        "dim": int(user_cf.shape[1]),
    }
    _SHARED_USER_CF[mm_artifacts_dir] = out
    print(f"[dense_multimodal_local] user_cf loaded: warm={len(uids)}, "
          f"dim={out['dim']} (cold fallback = train-set mean)")
    return out


class DENSE_MULTIMODAL_LOCAL:
    """Multi-modal bi-encoder retriever. Drop-in replacement for ``DENSE_LOCAL``
    that injects user CF embeddings into the query side.

    Args:
        dataset_name, split_types, corpus_types: interface parity with
            ``DENSE_LOCAL`` / ``DENSE_PRECOMPUTED``. Used only to validate
            the catalog cache exists at the expected path.
        cache_dir: parent of ``dense_local/`` where the catalog pickle lives.
        model_dir: filesystem path OR Hub repo of a
            ``MultiModalBiEncoder.save_pretrained()`` output. The pickle path
            is derived from ``model_dir.replace("/", "_")`` (same convention
            as ``DENSE_LOCAL``).
        embed_label: subdirectory under ``dense_local/<safe>/`` (matches
            ``scripts/embed_catalog_multimodal.py``'s ``--catalog-out-dir``
            suffix).
        backbone_override: optional path/Hub repo override for the base
            model (passed to ``MultiModalBiEncoder.from_pretrained``).
        multimodal_artifacts: path to the precompute cache (built by
            ``scripts/precompute_multimodal_artifacts.py``). Required —
            inference needs ``user_cf.npy`` + ``user_cf_mean.npy`` for
            query-side user injection.
        query_max_len: tokenizer max_length for the query text.
    """

    def __init__(
        self,
        dataset_name: str,
        split_types: list,
        corpus_types: list,
        cache_dir: str = "./cache",
        model_dir: str = "",
        embed_label: str = "default",
        backbone_override: Optional[str] = None,
        multimodal_artifacts: str = "",
        query_max_len: int = 384,
        query_encode_batch_size: int = 64,
    ) -> None:
        if not model_dir:
            raise ValueError(
                "DENSE_MULTIMODAL_LOCAL requires model_dir= (path to a "
                "MultiModalBiEncoder save_pretrained output)."
            )
        if not multimodal_artifacts:
            raise ValueError(
                "DENSE_MULTIMODAL_LOCAL requires multimodal_artifacts= "
                "(path to the precompute cache dir with user_cf*)."
            )
        if not os.path.isdir(multimodal_artifacts):
            raise FileNotFoundError(
                f"multimodal_artifacts not found: {multimodal_artifacts}"
            )

        self.dataset_name = dataset_name
        self.split_types = split_types
        self.cache_dir = cache_dir
        self.model_dir = model_dir
        self.embed_label = embed_label
        self.backbone_override = backbone_override
        self.multimodal_artifacts = multimodal_artifacts
        self.query_max_len = int(query_max_len)
        # Mini-batch size for query encoding. Encoding all queries in one
        # forward allocates an FFN activation of n x query_max_len x
        # intermediate_size floats (~35 GiB for 8000 queries at bge-base),
        # which OOMs the GPU. Chunking keeps peak activation bounded.
        self.query_encode_batch_size = int(query_encode_batch_size)

        # User CF lookup (shared across instances).
        self._user_cf = _load_user_cf_lookup(multimodal_artifacts)

        # Catalog (same format as DENSE_LOCAL — scoring code is unchanged).
        self.track_ids, self.track_mat = self._load_track_matrix()

        # Per-(model, "mm") query cache shared across instances.
        self._cache_key = (self.model_dir, "mm")
        self._query_cache_path = self._query_cache_file()
        if self._cache_key not in _SHARED_QUERY_CACHE:
            _SHARED_QUERY_CACHE[self._cache_key] = self._load_query_cache()
        self._query_cache: dict = _SHARED_QUERY_CACHE[self._cache_key]
        self._query_cache_dirty = False
        # I4 fix: only flush to disk every N new entries (defer writes).
        # Previously fired on every batch with any miss, which for an 8K-query
        # eval pass triggered many full pickle rewrites. Caller can force a
        # flush via ``.flush_query_cache()`` (e.g., at end of an eval loop).
        self._cache_save_every = 500
        self._cache_entries_since_save = 0

    def flush_query_cache(self) -> None:
        """Force-write the query cache to disk. Safe to call after an eval
        loop to ensure the last partial batch is persisted."""
        if self._query_cache_dirty:
            self._save_query_cache_to_disk()
            self._cache_entries_since_save = 0

    # ------------------------------------------------------------ catalog

    def _track_cache_path(self) -> str:
        safe_model = self.model_dir.replace("/", "_")
        return os.path.join(
            self.cache_dir, "dense_local", safe_model, self.embed_label,
            "track_embeddings.pkl",
        )

    def _load_track_matrix(self):
        path = self._track_cache_path()
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"DENSE_MULTIMODAL_LOCAL: missing catalog embeddings at {path}\n"
                f"Run: python scripts/embed_catalog_multimodal.py "
                f"--model-dir {self.model_dir} "
                f"--multimodal-artifacts {self.multimodal_artifacts} "
                f"--catalog-out-dir {os.path.dirname(path)}"
            )
        with open(path, "rb") as f:
            obj = pickle.load(f)
        mat = obj["track_mat"].astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        mat = mat / np.maximum(norms, 1e-9)
        print(
            f"[dense_multimodal_local] loaded catalog from {path} "
            f"(N={len(obj['track_ids'])}, dim={mat.shape[1]})"
        )
        return obj["track_ids"], mat

    # ------------------------------------------------------------- encoder

    def _get_model_and_tokenizer(self):
        key = (self.model_dir, self.backbone_override or "")
        if key in _SHARED_MODEL:
            return _SHARED_MODEL[key], _SHARED_TOKENIZER[key]
        import torch
        from transformers import AutoTokenizer
        # Local import — MultiModalBiEncoder pulls peft, which is a
        # training-only dep. Inference-only deploys without peft will still
        # be able to import this module (just won't be able to instantiate).
        from mcrs.training.multimodal_bi_encoder import MultiModalBiEncoder

        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        model = MultiModalBiEncoder.from_pretrained(
            self.model_dir, backbone_override=self.backbone_override,
        )
        model = model.to(device).eval()
        tok_source = self.backbone_override or model.config.backbone_name
        tokenizer = AutoTokenizer.from_pretrained(tok_source)
        tokenizer.truncation_side = "left"  # preserve [QUERY]: at end (issue G)
        _SHARED_MODEL[key] = model
        _SHARED_TOKENIZER[key] = tokenizer
        print(f"[dense_multimodal_local] loaded model from {self.model_dir} on {device}")
        return model, tokenizer

    def _get_user_cf(self, user_id) -> np.ndarray:
        """Returns the user's CF vector (warm) or the train-set mean
        (cold/unknown). Always returns a float32 (cf_dim,) array."""
        if user_id is None:
            return self._user_cf["mean"]
        idx = self._user_cf["uid_to_idx"].get(user_id)
        if idx is None:
            return self._user_cf["mean"]
        return np.asarray(self._user_cf["matrix"][idx], dtype=np.float32)

    def _encode_queries(self, queries: list, user_ids: list) -> np.ndarray:
        """Encode (query, user_id) pairs into normalized embeddings.

        Encodes in mini-batches of ``query_encode_batch_size`` so the peak FFN
        activation stays bounded (encoding all queries at once OOMs the GPU on
        large eval/Blind sets — see the class docstring).
        """
        import torch
        model, tokenizer = self._get_model_and_tokenizer()
        device = next(model.parameters()).device

        bs = self.query_encode_batch_size
        chunks = []
        for start in range(0, len(queries), bs):
            q_chunk = queries[start:start + bs]
            u_chunk = user_ids[start:start + bs]
            enc = tokenizer(
                q_chunk, max_length=self.query_max_len, padding=True,
                truncation=True, return_tensors="pt",
            ).to(device)
            user_cf_arr = np.stack(
                [self._get_user_cf(uid) for uid in u_chunk], axis=0,
            ).astype(np.float32)
            user_cf_t = torch.from_numpy(user_cf_arr).to(device)
            with torch.no_grad():
                embs = model.forward_query(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    user_cf=user_cf_t,
                )
            chunks.append(embs.float().cpu().numpy())
            del enc, user_cf_t, embs
        return np.concatenate(chunks, axis=0)

    # --------------------------------------------------------- query cache

    def _query_cache_file(self) -> str:
        safe_model = self.model_dir.replace("/", "_")
        return os.path.join(
            self.cache_dir, "dense_local", safe_model, "query_cache",
            "multimodal.pkl",
        )

    def _load_query_cache(self) -> dict:
        if os.path.exists(self._query_cache_path):
            try:
                with open(self._query_cache_path, "rb") as f:
                    cache = pickle.load(f)
                print(
                    f"[dense_multimodal_local] query cache: {len(cache)} entries "
                    f"from {self._query_cache_path}"
                )
                return cache
            except Exception as e:
                print(f"[dense_multimodal_local] failed to load query cache "
                      f"({e}); starting empty")
        return {}

    def _save_query_cache_to_disk(self) -> None:
        """Atomic pickle dump of the full query cache. Expensive for large
        caches — call sparingly via ``flush_query_cache()`` or
        ``_maybe_save_query_cache()``."""
        os.makedirs(os.path.dirname(self._query_cache_path), exist_ok=True)
        tmp = self._query_cache_path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(self._query_cache, f)
        os.replace(tmp, self._query_cache_path)
        self._query_cache_dirty = False

    def _maybe_save_query_cache(self, n_new_entries: int) -> None:
        """Batched save: only flush every ``_cache_save_every`` new entries.
        Amortizes pickle-dump cost over many batches."""
        self._cache_entries_since_save += n_new_entries
        if self._cache_entries_since_save >= self._cache_save_every:
            self._save_query_cache_to_disk()
            self._cache_entries_since_save = 0

    # ----------------------------------------------------- public interface

    def batch_text_to_item_retrieval(
        self, queries: list, topk: int, user_ids=None,
    ):
        """Per-(query, user) → top-K track_ids.

        Cache key is ``(query_text, user_id_or_None)``: critical for
        correctness. Same query produces different embeddings per user
        because of the ``<user_cf>`` pseudo-token; caching by text alone
        would silently return the FIRST user's embedding for all later
        users with the same query.
        """
        n = len(queries)
        if user_ids is None:
            user_ids = [None] * n
        if len(user_ids) != n:
            raise ValueError(
                f"DENSE_MULTIMODAL_LOCAL: user_ids length {len(user_ids)} "
                f"!= queries length {n}"
            )

        # Cache lookup. Key includes user_id so per-user embeddings stay distinct.
        cache_keys = [(q, uid) for q, uid in zip(queries, user_ids)]
        missing_idx = [i for i, k in enumerate(cache_keys) if k not in self._query_cache]
        hits = n - len(missing_idx)
        if missing_idx:
            to_q = [queries[i] for i in missing_idx]
            to_u = [user_ids[i] for i in missing_idx]
            embs = self._encode_queries(to_q, to_u)
            for j, idx in enumerate(missing_idx):
                self._query_cache[cache_keys[idx]] = embs[j].astype(np.float32)
            self._query_cache_dirty = True
            # I4: batched save — only flushes every _cache_save_every entries.
            # Call flush_query_cache() explicitly at end of eval loop to
            # persist the final partial batch.
            self._maybe_save_query_cache(len(missing_idx))
        if hits and queries:
            print(f"[dense_multimodal_local] query cache hit/total = {hits}/{n}")

        q_mat = np.stack(
            [self._query_cache[k] for k in cache_keys], axis=0,
        ).astype(np.float32)
        scores = q_mat @ self.track_mat.T  # (n, T)
        results = []
        for i in range(scores.shape[0]):
            row = scores[i]
            if topk >= row.shape[0]:
                order = np.argsort(-row)
            else:
                cand = np.argpartition(-row, topk)[:topk]
                order = cand[np.argsort(-row[cand])]
            results.append([self.track_ids[idx] for idx in order])
        return results

    def text_to_item_retrieval(self, query: str, topk: int,
                                user_id=None) -> list:
        return self.batch_text_to_item_retrieval([query], topk=topk,
                                                  user_ids=[user_id])[0]
