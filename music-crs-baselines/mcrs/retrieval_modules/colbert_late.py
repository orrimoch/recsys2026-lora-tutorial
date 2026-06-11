"""ColBERT late-interaction retrieval channel.

Single-vector bi-encoders average a multi-facet query into one vector and lose
the rare-but-decisive facets (era / region / artist-style). ColBERT keeps one
vector per token and matches each facet independently via MaxSim
(`score = Σ_i max_j q_i · d_j`) — removing the averaging that caps our recall.

This module has two layers:
  * Pure scoring core (`maxsim_score`, `rerank_pool`) — no model/pylate
    dependency; token embeddings are passed in directly. Unit-tested.
  * `ColbertRetriever` — wraps a PyLate ColBERT model for encoding. The
    pool-rerank path (P1 zero-shot kill-test) needs no index. The full
    PLAID-indexed recall-channel path (P2) is added once P1 passes.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

# Module-level cache so repeated ColbertRetriever instantiations share one loaded
# PyLate model (mirrors the _SHARED_* singletons in dense_precomputed / clap).
_SHARED_COLBERT_MODEL: dict[str, object] = {}

# Default off-the-shelf checkpoint for the P1 zero-shot probe.
DEFAULT_COLBERT_MODEL = "colbert-ir/colbertv2.0"


def maxsim_score(query_emb: np.ndarray, doc_emb: np.ndarray) -> float:
    """ColBERT late-interaction score between one query and one document.

    `score = Σ_i max_j (q_i · d_j)` — for each query token, take its single best
    similarity to any doc token, then sum across query tokens. Embeddings are
    assumed L2-normalized, so the dot product is cosine similarity.

    Args:
        query_emb: (n_q, dim) query token embeddings.
        doc_emb:   (n_d, dim) document token embeddings.

    Returns:
        Scalar MaxSim score (higher = better match).
    """
    sim = np.asarray(query_emb, dtype=np.float32) @ np.asarray(doc_emb, dtype=np.float32).T
    return float(sim.max(axis=1).sum())


def rerank_pool(
    query_emb: np.ndarray,
    pool_embs: Sequence[np.ndarray],
    pool_tids: Sequence[str],
    topk: int,
) -> list[str]:
    """Rerank a candidate pool by MaxSim and return the top-`topk` track ids.

    This is the zero-shot probe path: ColBERT acts as a pool reranker over an
    existing recall pool (e.g. the union `cs` pool), needing no PLAID index.

    Args:
        query_emb: (n_q, dim) query token embeddings.
        pool_embs: per-candidate (n_d, dim) document token embeddings, aligned to
            `pool_tids`.
        pool_tids: candidate track ids, aligned to `pool_embs`.
        topk: number of ids to return.

    Returns:
        Track ids ordered by descending MaxSim, truncated to `topk`.
    """
    scores = np.array([maxsim_score(query_emb, d) for d in pool_embs], dtype=np.float32)
    order = np.argsort(-scores)[:topk]
    return [pool_tids[i] for i in order]


def _load_pylate_encoder(model_name: str, q_len: int, d_len: int):
    """Lazy-load a PyLate ColBERT model and return (query_encoder, doc_encoder)
    callables, each mapping list[str] -> list[(n_tok, dim) np.ndarray].

    Imported lazily so the pure scoring core and the unit tests have no hard
    dependency on `pylate` (which is GPU/notebook-only).
    """
    cache_key = f"{model_name}|q{q_len}|d{d_len}"
    if cache_key not in _SHARED_COLBERT_MODEL:
        from pylate import models  # lazy: pylate is notebook/GPU-only

        _SHARED_COLBERT_MODEL[cache_key] = models.ColBERT(
            model_name_or_path=model_name,
            query_length=q_len,
            document_length=d_len,
        )
    model = _SHARED_COLBERT_MODEL[cache_key]

    def _encode(texts, is_query: bool):
        embs = model.encode(
            list(texts),
            is_query=is_query,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [np.asarray(e, dtype=np.float32) for e in embs]

    return (lambda qs: _encode(qs, True), lambda ds: _encode(ds, False))


class ColbertRetriever:
    """ColBERT late-interaction retriever.

    The zero-shot probe path (`batch_rerank_pool`) reranks an existing candidate
    pool by MaxSim — no PLAID index required. Document token embeddings are
    precomputed once (keyed by track id) and the query encoder runs online.

    The encoder is injectable so the orchestration is unit-testable without
    `pylate`. In production, omit `query_encoder` and pass `model_name`/lengths to
    lazy-load a PyLate ColBERT model.
    """

    def __init__(
        self,
        doc_embs: dict[str, np.ndarray],
        query_encoder: Optional[Callable[[Sequence[str]], list[np.ndarray]]] = None,
        doc_encoder: Optional[Callable[[Sequence[str]], list[np.ndarray]]] = None,
        model_name: str = DEFAULT_COLBERT_MODEL,
        q_len: int = 32,
        d_len: int = 96,
    ):
        self.doc_embs = doc_embs
        if query_encoder is None:
            query_encoder, lazy_doc_encoder = _load_pylate_encoder(model_name, q_len, d_len)
            if doc_encoder is None:
                doc_encoder = lazy_doc_encoder
        self._query_encoder = query_encoder
        self._doc_encoder = doc_encoder

    def encode_docs(
        self, tids: Sequence[str], texts: Sequence[str]
    ) -> dict[str, np.ndarray]:
        """Encode catalog document texts to ColBERT token embeddings and merge
        them into `self.doc_embs` (keyed by track id). Run once to build the
        lookup the probe reranks against. Returns the updated `doc_embs`.
        """
        embs = self._doc_encoder(list(texts))
        for tid, emb in zip(tids, embs):
            self.doc_embs[tid] = np.asarray(emb, dtype=np.float32)
        return self.doc_embs

    def batch_rerank_pool(
        self,
        queries: Sequence[str],
        pools: Sequence[Sequence[str]],
        topk: int,
    ) -> list[list[str]]:
        """Rerank each query's own candidate pool by MaxSim (the P1 probe).

        Pool track ids without a precomputed doc embedding are skipped (cannot be
        scored). Returns a per-query ranked list of track ids, truncated to `topk`.
        """
        query_embs = self._query_encoder(list(queries))
        results: list[list[str]] = []
        for q_emb, pool in zip(query_embs, pools):
            tids = [t for t in pool if t in self.doc_embs]
            pool_embs = [self.doc_embs[t] for t in tids]
            results.append(rerank_pool(q_emb, pool_embs, tids, topk))
        return results
