"""cf-bpr user x item affinity retriever.

Uses precomputed cf-bpr embeddings (128-dim) from
`talkpl-ai/TalkPlayData-Challenge-User-Embeddings` + the `cf-bpr` column
of `talkpl-ai/TalkPlayData-Challenge-Track-Embeddings`. Score(user, track)
= cosine(user_emb, track_emb) after per-vector L2 normalization.

Warm users: dot-product ranking over all tracks -> top-K.
Cold users (no user_emb): return empty list. When used as an RRF sub,
empty list contributes 0 to the fusion, so cold users fall back to the
other branches (BM25 + dense). Zero regression for cold, net positive
for warm. See `project_fresh_model_state.md` A2.

Module-level caches:
  - _SHARED_TRACK_EMB: one (tids, np.ndarray 47k x 128) load.
  - _SHARED_USER_EMB: one {user_id: np.ndarray 128} load.
Sharing across retriever instances is automatic (singleton by
`embed_col`). Unlike the dense retriever's encoder, cf-bpr has no
online encoding; this retriever is numpy-only.
"""
from __future__ import annotations

import os
import pickle
from typing import Optional

import numpy as np


USER_EMB_DATASET = "talkpl-ai/TalkPlayData-Challenge-User-Embeddings"
TRACK_EMB_DATASET = "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings"
EMBED_COL = "cf-bpr"

_SHARED_TRACK_EMB: dict[str, tuple[list[str], np.ndarray]] = {}
_SHARED_USER_EMB: dict[str, dict[str, np.ndarray]] = {}


def _l2_normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    return m / norms


class CF_BPR:
    def __init__(
        self,
        dataset_name: str,  # unused; kept for factory parity
        split_types: list[str],  # e.g. ["all_tracks"]
        corpus_types: list[str],  # unused; kept for factory parity
        cache_dir: str = "./cache",
    ) -> None:
        # Accept the same signature as BM25 / dense retrievers so the factory
        # can swap any retriever in. `dataset_name` and `corpus_types` are
        # ignored — cf-bpr uses a fixed pair of HF datasets for its embeddings.
        self.split_types = split_types
        self.cache_dir = cache_dir
        self.track_ids, self.track_mat = self._load_tracks()
        self.user_embs = self._load_users()
        n_warm = len(self.user_embs)
        print(f"[cf-bpr] ready — tracks={len(self.track_ids)} warm_users={n_warm}")

    # ------------------------------------------------------------------ track
    def _load_tracks(self) -> tuple[list[str], np.ndarray]:
        key = EMBED_COL
        if key in _SHARED_TRACK_EMB:
            return _SHARED_TRACK_EMB[key]
        cache_path = os.path.join(self.cache_dir, "cf_bpr", "track_mat.pkl")
        if os.path.isfile(cache_path):
            with open(cache_path, "rb") as f:
                obj = pickle.load(f)
            _SHARED_TRACK_EMB[key] = (obj["track_ids"], obj["track_mat"])
            print(f"[cf-bpr] loaded track cache: {len(obj['track_ids'])} tracks")
            return _SHARED_TRACK_EMB[key]

        from datasets import concatenate_datasets, load_dataset

        print(f"[cf-bpr] loading {TRACK_EMB_DATASET}[{self.split_types}] col={EMBED_COL}")
        ds = load_dataset(TRACK_EMB_DATASET)
        concat = concatenate_datasets([ds[s] for s in self.split_types])
        tids: list[str] = []
        rows: list[np.ndarray] = []
        empty = 0
        for r in concat:
            emb = r.get(EMBED_COL)
            if not emb:
                empty += 1
                continue
            tids.append(r["track_id"])
            rows.append(np.asarray(emb, dtype=np.float32))
        mat = np.stack(rows, axis=0)
        mat = _l2_normalize(mat)
        print(f"[cf-bpr] track_mat shape={mat.shape} empty_rows_dropped={empty}")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump({"track_ids": tids, "track_mat": mat}, f)
        _SHARED_TRACK_EMB[key] = (tids, mat)
        return tids, mat

    # ------------------------------------------------------------------ user
    def _load_users(self) -> dict[str, np.ndarray]:
        key = EMBED_COL
        if key in _SHARED_USER_EMB:
            return _SHARED_USER_EMB[key]
        cache_path = os.path.join(self.cache_dir, "cf_bpr", "user_embs.pkl")
        if os.path.isfile(cache_path):
            with open(cache_path, "rb") as f:
                u = pickle.load(f)
            _SHARED_USER_EMB[key] = u
            print(f"[cf-bpr] loaded user cache: {len(u)} warm users")
            return u

        from datasets import concatenate_datasets, load_dataset

        print(f"[cf-bpr] loading {USER_EMB_DATASET} (all splits)")
        ds = load_dataset(USER_EMB_DATASET)
        concat = concatenate_datasets([ds[s] for s in ds])
        user_embs: dict[str, np.ndarray] = {}
        for r in concat:
            uid = r["user_id"]
            emb = r.get(EMBED_COL)
            if not emb:
                continue
            v = np.asarray(emb, dtype=np.float32)
            v = v / max(np.linalg.norm(v), 1e-9)
            user_embs[uid] = v
        print(f"[cf-bpr] user_embs: {len(user_embs)} warm users")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(user_embs, f)
        _SHARED_USER_EMB[key] = user_embs
        return user_embs

    # --------------------------------------------------------------- retrieve
    def batch_text_to_item_retrieval(
        self,
        queries: list[str],
        topk: int,
        user_ids: Optional[list[Optional[str]]] = None,
    ) -> list[list[str]]:
        """Score each user's preference over tracks via cosine similarity.

        queries: unused for scoring (cf-bpr is query-independent); kept
                 for interface parity with BM25 / dense.
        user_ids: optional parallel list. Warm user -> top-K by score;
                  cold user (None or not in user_embs) -> empty list.
        """
        n = len(queries)
        if user_ids is None:
            user_ids = [None] * n
        if len(user_ids) != n:
            raise ValueError(f"user_ids length {len(user_ids)} != queries length {n}")

        # Collect warm rows, build a stacked query matrix, batch-score once.
        warm_idx: list[int] = []
        warm_user_vecs: list[np.ndarray] = []
        for i, uid in enumerate(user_ids):
            if uid and uid in self.user_embs:
                warm_idx.append(i)
                warm_user_vecs.append(self.user_embs[uid])

        results: list[list[str]] = [[] for _ in range(n)]
        if warm_idx:
            U = np.stack(warm_user_vecs, axis=0)  # (W, 128)
            scores = U @ self.track_mat.T  # (W, T)
            for wi, orig_i in enumerate(warm_idx):
                row = scores[wi]
                if topk >= row.shape[0]:
                    order = np.argsort(-row)
                else:
                    cand = np.argpartition(-row, topk)[:topk]
                    order = cand[np.argsort(-row[cand])]
                results[orig_i] = [self.track_ids[idx] for idx in order]
        return results

    def text_to_item_retrieval(
        self, query: str, topk: int, user_id: Optional[str] = None,
    ) -> list[str]:
        return self.batch_text_to_item_retrieval([query], topk=topk, user_ids=[user_id])[0]
