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

# Query/doc token budgets. q_len=96 (was 32) because our raw turn-1 query is the
# full dialog + 'goal: <listener_goal>' (mean ~55 tok, p99 ~93) — q_len=32
# right-truncated the goal (the highest-value turn-1 signal) off ~95% of queries.
DEFAULT_Q_LEN = 96
DEFAULT_D_LEN = 96


def strip_track_id_prefix(doc_text: str) -> str:
    """Drop the leading 'track_id: <uuid>, ' segment from a catalog doc string.

    `format_catalog_track_text` prepends the track_id (a UUID) as the first field;
    it tokenizes to ~30 content-free hex subwords that dilute ColBERT MaxSim (every
    query token can max onto noise). The UUID never contains ', ', so the first
    ', ' reliably ends the track_id field. Non-prefixed or field-less text is
    returned unchanged (never empty).
    """
    if doc_text.startswith("track_id: ") and ", " in doc_text:
        return doc_text.split(", ", 1)[1]
    return doc_text


# --------------------------------------------------------------------------- #
# EXP-217: curated-tag doc enrichment (doc-side, deterministic). Raw tag_list is
# noisy folksonomy + long (mean +111 tok); these three pure helpers curate it to a
# short clean genre/mood suffix. SHARED by the train-data builder + the index builder
# so the enriched doc text is byte-identical (doc-side parity, like the q_len rule).
# --------------------------------------------------------------------------- #
def build_tag_vocab(tag_rows, min_freq: int = 1) -> dict[str, int]:
    """Normalized-tag -> catalog frequency, keeping tags with freq >= min_freq.

    Computed ONCE over the catalog. The frequency floor is the noise filter: real
    genres/moods recur across many tracks; idiosyncratic junk ('goeiepoep', '3 of 10
    stars') appears once and is dropped. Tags are lowercased/stripped before counting.
    """
    from collections import Counter
    counts: Counter = Counter()
    for row in tag_rows:
        for t in (row or []):
            n = str(t).strip().lower()
            if n:
                counts[n] += 1
    return {t: f for t, f in counts.items() if f >= min_freq}


def curate_tags(raw_tags, vocab: dict[str, int], top_k: int = 15) -> list[str]:
    """One track's raw tags -> curated list: normalize+dedup, keep only vocab tags,
    sort by catalog frequency desc (alpha tie-break for determinism), cap at top_k."""
    seen: set = set()
    cand: list[str] = []
    for t in (raw_tags or []):
        n = str(t).strip().lower()
        if n and n in vocab and n not in seen:
            seen.add(n)
            cand.append(n)
    cand.sort(key=lambda t: (-vocab[t], t))
    return cand[:top_k]


def enrich_doc_text(base_text: str, tags: list[str]) -> str:
    """Append ', tags: t1, t2, ...' to the base doc when tags present; else unchanged.
    Tags go LAST (after title/artist/album); d_len must be sized so they survive."""
    if not tags:
        return base_text
    return f"{base_text}, tags: {', '.join(tags)}"


def colbert_doc_text(track_id, id_to_metadata, metadata_dict, vocab, top_k: int = 15) -> str:
    """Per-track enriched ColBERT doc — the convenience SHARED by the train-data + index
    builders (single source of truth for doc-side parity). Strips the track_id prefix off
    the base metadata text, curates the track's tag_list against `vocab`, and appends them.
    Falls back to the bare stripped doc when the track/tags are absent."""
    base = strip_track_id_prefix(id_to_metadata(track_id))
    raw_tags = (metadata_dict.get(track_id) or {}).get("tag_list")
    return enrich_doc_text(base, curate_tags(raw_tags, vocab, top_k))


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


def pylate_encoders(model):
    """Wrap an already-loaded PyLate ColBERT model into (query_encoder,
    doc_encoder) callables, each mapping list[str] -> list[(n_tok, dim) np.ndarray].

    Used both by the lazy loader below and by the trainer's in-loop dev eval,
    which must encode with the *live* (mid-training) model object.
    """
    def _encode(texts, is_query: bool):
        embs = model.encode(
            list(texts),
            is_query=is_query,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [np.asarray(e, dtype=np.float32) for e in embs]

    return (lambda qs: _encode(qs, True), lambda ds: _encode(ds, False))


def _load_pylate_encoder(model_name: str, q_len: int, d_len: int):
    """Lazy-load a PyLate ColBERT model and return its (query_encoder, doc_encoder)
    callables. Imported lazily so the pure scoring core and the unit tests have no
    hard dependency on `pylate` (which is GPU/notebook-only).
    """
    cache_key = f"{model_name}|q{q_len}|d{d_len}"
    if cache_key not in _SHARED_COLBERT_MODEL:
        from pylate import models  # lazy: pylate is notebook/GPU-only

        _SHARED_COLBERT_MODEL[cache_key] = models.ColBERT(
            model_name_or_path=model_name,
            query_length=q_len,
            document_length=d_len,
        )
    return pylate_encoders(_SHARED_COLBERT_MODEL[cache_key])


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
        q_len: int = DEFAULT_Q_LEN,
        d_len: int = DEFAULT_D_LEN,
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


# --------------------------------------------------------------------------- #
# Stage B: full-catalog ColBERT recall channel (PyLate PLAID index).
#
# ColbertRetriever (above) only RERANKS a given pool — it can never surface a gold
# the union missed (recall@100 is invariant under pool-reranking). ColbertIndexRetriever
# retrieves over the WHOLE 47k catalog, so it CAN add new golds — the only artifact
# that tests the plan's recall thesis. Build the index with scripts/build_colbert_index.py.
# --------------------------------------------------------------------------- #
def plaid_results_to_tids(results) -> list[list[str]]:
    """Convert PyLate PLAID `retrieve()` output (per-query list of {id, score} dicts,
    ranked) into per-query ranked lists of track ids."""
    return [[hit["id"] for hit in q] for q in results]


class ColbertIndexRetriever:
    """ColBERT full-catalog retriever over a PyLate PLAID index — the Stage-B recall
    channel. Conforms to the `batch_text_to_item_retrieval` contract so the wRRF
    factory can fuse it as a union channel.

    Inject `query_encoder` + `plaid_retriever` for unit tests; in production pass
    `index_folder` / `index_name` / `model_name` to lazy-load PyLate.
    """

    def __init__(
        self,
        index_folder: Optional[str] = None,
        index_name: str = "colbert-index",
        model_name: str = DEFAULT_COLBERT_MODEL,
        q_len: int = DEFAULT_Q_LEN,
        d_len: int = DEFAULT_D_LEN,
        query_encoder: Optional[Callable[[Sequence[str]], object]] = None,
        plaid_retriever: object = None,
    ):
        if query_encoder is None or plaid_retriever is None:
            from pylate import indexes, models, retrieve  # lazy: notebook/GPU-only

            model = models.ColBERT(
                model_name_or_path=model_name, query_length=q_len, document_length=d_len
            )
            index = indexes.PLAID(
                index_folder=index_folder, index_name=index_name, override=False
            )
            plaid_retriever = retrieve.ColBERT(index=index)
            query_encoder = lambda qs: model.encode(  # noqa: E731
                list(qs), is_query=True, show_progress_bar=False
            )
        self._query_encoder = query_encoder
        self._retriever = plaid_retriever

    def batch_text_to_item_retrieval(
        self, queries, topk, user_ids=None, batch_context=None
    ) -> list[list[str]]:
        """Retrieve the top-`topk` catalog track ids per query (full-catalog, not a
        pool). `user_ids`/`batch_context` are accepted for interface parity, unused."""
        q_emb = self._query_encoder(list(queries))
        results = self._retriever.retrieve(queries_embeddings=q_emb, k=topk)
        return plaid_results_to_tids(results)

    def text_to_item_retrieval(self, query, topk, user_id=None) -> list[str]:
        return self.batch_text_to_item_retrieval([query], topk=topk)[0]
