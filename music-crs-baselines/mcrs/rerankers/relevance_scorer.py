"""Leak-free relevance features for the LGBM reranker (Tier-2 #4.1).

Two continuous query->candidate relevance signals the reranker currently lacks:
  - qwen_meta_cos : cosine(frozen pretrained Qwen3-instruct(query),
                    precomputed metadata-qwen3 catalog emb[tid]). Pretrained
                    encoder => no in-sample label leak.
  - bm25_score    : raw BM25 score of the query for the candidate (0 outside the
                    bm25 top-K).

A single RelevanceScorer computes both; build_lgbm_features (train) and
crs_baseline (serve) both call it, so the magnitudes are identical by
construction (parity). The pure core is unit-tested; the model wrapper is
exercised on Colab.
"""
from __future__ import annotations

import numpy as np


def relevance_feats_for_candidates(q_vec, catalog_norm, tid_to_idx, bm25_map,
                                   cand_tids) -> list[dict]:
    """Per-candidate {qwen_meta_cos, bm25_score}.

    q_vec: (d,) L2-normalized query embedding. catalog_norm: (N,d) L2-normalized
    catalog embeddings. tid_to_idx: tid -> row in catalog_norm. bm25_map: tid ->
    raw bm25 score (missing => 0). A tid absent from the catalog -> cos 0.0.
    """
    q = np.asarray(q_vec, dtype=np.float64)
    out = []
    for tid in cand_tids:
        idx = tid_to_idx.get(tid)
        cos = float(q @ catalog_norm[idx]) if idx is not None else 0.0
        out.append({"qwen_meta_cos": cos,
                    "bm25_score": float(bm25_map.get(tid, 0.0))})
    return out


def _l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float64)


class RelevanceScorer:
    """Frozen Qwen3 dense + BM25 scorer shared by train and serve.

    Reuses DENSE_PRECOMPUTED's shared-singleton Qwen3 encoder (no double load)
    and the precomputed metadata-qwen3 catalog matrix; loads a BM25_MODEL for
    lexical scores. feats_for_batch returns, per query, a per-candidate list of
    {qwen_meta_cos, bm25_score} aligned to that query's cand_tids.
    """

    def __init__(self, dataset_name, split_types, corpus_types, cache_dir,
                 bm25_topk: int = 500):
        from ..retrieval_modules.dense_precomputed import DENSE_PRECOMPUTED
        from ..retrieval_modules import QWEN3_MUSIC_INSTRUCT
        from ..retrieval_modules.bm25 import BM25_MODEL
        self.dense = DENSE_PRECOMPUTED(
            dataset_name, split_types, corpus_types, cache_dir,
            embed_col="metadata-qwen3_embedding_0.6b",
            instruct=QWEN3_MUSIC_INSTRUCT, instruct_label="instruct-music-v1")
        self.catalog_norm = _l2_normalize_rows(np.asarray(self.dense.track_mat))
        self.tid_to_idx = {t: i for i, t in enumerate(self.dense.track_ids)}
        self.bm25 = BM25_MODEL(dataset_name, split_types, corpus_types, cache_dir)
        self.bm25_topk = int(bm25_topk)

    @staticmethod
    def _bm25_score_maps(res, track_ids, n_queries) -> list[dict]:
        """Parse a bm25s tuple-format retrieve result -> per-query {tid: score}.

        Matches BM25_MODEL: res.documents[i] is a list of {'id': corpus_idx}
        dicts, res.scores[i] the parallel score array, and track_ids[idx] the tid.
        """
        out = []
        for i in range(n_queries):
            out.append({track_ids[item["id"]]: float(s)
                        for item, s in zip(res.documents[i], res.scores[i])})
        return out

    def _bm25_maps(self, queries):
        """tid -> raw bm25 score per query (top-`bm25_topk`). Mirrors
        BM25_MODEL.batch_text_to_item_retrieval (lowercase + tokenize + retrieve)."""
        import bm25s
        toks = bm25s.tokenize([q.lower() for q in queries])
        res = self.bm25.bm25_model.retrieve(
            toks, k=self.bm25_topk, return_as="tuple")
        return self._bm25_score_maps(res, self.bm25.track_ids, len(queries))

    def feats_for_batch(self, queries, cand_tids_per_query) -> list[list[dict]]:
        q_emb = np.asarray(self.dense._encode_queries(list(queries)), dtype=np.float64)
        q_emb = _l2_normalize_rows(q_emb)
        bm25_maps = self._bm25_maps(list(queries))
        return [
            relevance_feats_for_candidates(
                q_emb[i], self.catalog_norm, self.tid_to_idx, bm25_maps[i], cands)
            for i, cands in enumerate(cand_tids_per_query)
        ]
