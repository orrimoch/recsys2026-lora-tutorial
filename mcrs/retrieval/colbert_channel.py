"""R8 — ColBERT late-interaction channel (brute-force MaxSim over the full catalog).

A token-level semantic retriever: encode the focused query and each A1-enriched catalog doc
into ONE vector per token (no pooling), score by MaxSim — `S(Q,D)=Σ_i max_j (q_i·d_j)` — and
return the top-`topk` canonical track_ids. It is the *semantic current-intent* channel and
**replaces** R4's query-dense role (the two are redundant; never run both — see 47_R8 §4.1).

Mirrors R4 `DenseChannel`: the query encoder is INJECTED (`encode_query_fn`) and the per-doc
token embeddings are precomputed once (like R4's catalog matrix), so the retrieval logic is
unit-testable without a GPU/`pylate`. At 47k the plan recommends brute-force MaxSim over PLAID
(one matmul, no ANN approximation, no recall lost — plan §3.3 / §3 insight 2); production wires
PyLate in via `from_pylate`. `batch_context`/`user_ids` are accepted for F2 parity and IGNORED
(content-only channel; personalization stays in R5). See 47_R8.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from mcrs.data.ids import canonical_track_id
from mcrs.retrieval._vec import l2_normalize

# Per-query token embeddings: list[str] -> list of (n_query_tokens, dim) arrays (ragged).
QueryEncodeFn = Callable[[Sequence[str]], list]

_PREFIX = "track_id: "


def maxsim_score(query_emb: np.ndarray, doc_emb: np.ndarray) -> float:
    """ColBERT late-interaction score for one (query, doc): `Σ_i max_j (q_i · d_j)`.

    Each query token takes its single best-matching doc token (max cosine), summed over query
    tokens. Embeddings are assumed L2-normalized (so dot == cosine)."""
    sim = np.asarray(query_emb, dtype=np.float32) @ np.asarray(doc_emb, dtype=np.float32).T
    return float(sim.max(axis=1).sum())


def _strip_track_id_prefix(doc_text: str) -> str:
    """Drop a leading 'track_id: <uuid>, ' segment from a doc string if present.

    The id tokenizes to content-free hex subwords that every query token can max onto (MaxSim
    noise). The id never contains ', ', so the first ', ' ends the field. Non-prefixed text is
    returned unchanged. (Doc-TEXT helper only — canonicalize ids via F1, not this; plan §7.3.)"""
    if doc_text.startswith(_PREFIX) and ", " in doc_text:
        return doc_text.split(", ", 1)[1]
    return doc_text


def colbert_doc_text(catalog, track_id: str) -> str:
    """The ColBERT doc text for a track: the A1-ENRICHED doc (47_R8 §4.2, hard constraint),
    with a leading track_id prefix stripped. Channels never build doc text themselves —
    enrichment is layered by A1 and read through F1 `id_to_metadata(enriched=True)`."""
    return _strip_track_id_prefix(catalog.id_to_metadata(track_id, enriched=True))


def count_docs_over_budget(texts: Sequence[str], budget: int,
                           tokenize: Optional[Callable[[str], Sequence]] = None) -> int:
    """How many docs exceed `budget` tokens — the doc-cap-hit counter (47_R8 §4.3/§8).

    A non-zero count is a signal to PRUNE doc text (trim the raw tag dump, keep the doc2query
    expansion), not to silently truncate. Default tokenizer is whitespace split (a cheap proxy);
    the production builder passes the model tokenizer for an exact subword count."""
    tok = tokenize or (lambda s: s.split())
    return sum(1 for t in texts if len(tok(t)) > budget)


def _topk_stable(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest scores, descending, ties broken by ASCENDING index (47_R8 §7
    determinism). lexsort primary key is the last arg (-scores), secondary is the index."""
    k = min(k, scores.shape[0])
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    order = np.lexsort((np.arange(scores.shape[0]), -scores))
    return order[:k]


class ColBERTChannel:
    """F2 `RetrievalChannel` — full-catalog ColBERT late-interaction by brute-force MaxSim."""

    def __init__(
        self,
        index_to_id: Sequence[str],
        doc_token_embeddings: Sequence[np.ndarray],
        encode_query_fn: QueryEncodeFn,
        label: str = "colbert",
        normalize: bool = True,
        chunk_docs: int = 4096,
    ) -> None:
        self.label = label
        self.index_to_id = list(index_to_id)
        self.encode_query_fn = encode_query_fn
        self.normalize = normalize
        self.chunk_docs = max(1, int(chunk_docs))
        if len(self.index_to_id) != len(doc_token_embeddings):
            raise ValueError(
                f"index_to_id ({len(self.index_to_id)}) and doc_token_embeddings "
                f"({len(doc_token_embeddings)}) must align 1:1 (47_R8 §4.6 row alignment)"
            )
        mats = [np.asarray(m, dtype=np.float32) for m in doc_token_embeddings]
        lengths = [m.shape[0] for m in mats]
        if any(n < 1 for n in lengths):
            raise ValueError("every doc must have >= 1 token (empty doc breaks MaxSim/reduceat)")
        self._offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        stacked = np.vstack(mats)               # (total_tokens, dim)
        self._doc_matrix = l2_normalize(stacked) if normalize else stacked
        self._n_docs = len(mats)
        self.n_docs_over_budget = 0             # set by from_catalog when doc_token_budget given

    # ----- builders -----
    @classmethod
    def from_catalog(
        cls,
        catalog,
        encode_query_fn: QueryEncodeFn,
        encode_docs_fn: Callable[[Sequence[str]], Sequence[np.ndarray]],
        *,
        normalize: bool = True,
        chunk_docs: int = 4096,
        label: str = "colbert",
        doc_text_fn: Optional[Callable[[object, str], str]] = None,
        doc_token_budget: Optional[int] = None,
        tokenize: Optional[Callable[[str], Sequence]] = None,
    ) -> "ColBERTChannel":
        """Build the channel from an F1 `Catalog`: encode each track's A1-enriched doc to ColBERT
        token embeddings, indexed over `catalog.index_to_id` (the F1 id space, §4.6 row alignment).
        Encoders are injected so this is testable without `pylate`; `from_pylate` wires the real
        model. When `doc_token_budget` is given, `n_docs_over_budget` records the cap-hit count."""
        ids = list(catalog.index_to_id)
        text_of = doc_text_fn or colbert_doc_text
        texts = [text_of(catalog, t) for t in ids]
        doc_embs = list(encode_docs_fn(texts))
        if len(doc_embs) != len(ids):
            raise ValueError(
                f"doc encoder returned {len(doc_embs)} embeddings for {len(ids)} catalog tracks "
                f"(§4.6 row alignment — index size must equal len(catalog))"
            )
        ch = cls(ids, doc_embs, encode_query_fn, label=label,
                 normalize=normalize, chunk_docs=chunk_docs)
        if doc_token_budget is not None:
            ch.n_docs_over_budget = count_docs_over_budget(texts, doc_token_budget, tokenize)
        return ch

    @classmethod
    def from_pylate(cls, catalog, cfg, *, label: str = "colbert") -> "ColBERTChannel":
        """Production builder: lazy-load a PyLate ColBERT model from `cfg.retrieval.colbert` and
        encode the enriched catalog docs once. `pylate` is GPU/notebook-only, imported lazily so
        the pure core + unit tests have no hard dependency on it (47_R8 §5)."""
        from pylate import models  # lazy: notebook/GPU-only

        c = cfg.retrieval.colbert
        model = models.ColBERT(
            model_name_or_path=c.model,
            query_length=c.query_maxlen,
            document_length=c.doc_maxlen,
        )

        def _encode(texts, is_query):
            embs = model.encode(list(texts), is_query=is_query,
                                convert_to_numpy=True, show_progress_bar=not is_query,
                                batch_size=c.bsize)
            return [np.asarray(e, dtype=np.float32) for e in embs]

        return cls.from_catalog(
            catalog,
            encode_query_fn=lambda qs: _encode(qs, True),
            encode_docs_fn=lambda ds: _encode(ds, False),
            normalize=c.normalize,
            chunk_docs=c.chunk_docs,
            label=label,
            doc_token_budget=c.doc_maxlen,
            tokenize=lambda s: model.tokenize([s], is_query=False)["input_ids"][0],
        )

    def batch_text_to_item_retrieval(
        self, queries: list[str], topk: int,
        batch_context: Optional[list[dict]] = None, user_ids: Optional[list[str]] = None,
    ) -> list[list[str]]:
        # batch_context / user_ids intentionally ignored: content-only channel (no-leak, §4.7).
        if not queries:
            return []
        q_embs = self.encode_query_fn(list(queries))
        out: list[list[str]] = []
        for q in q_embs:
            q = np.asarray(q, dtype=np.float32)
            if q.ndim == 1:
                q = q[None, :]
            if self.normalize:
                q = l2_normalize(q)
            scores = self._scores_for_query(q)
            idx = _topk_stable(scores, topk)
            out.append([canonical_track_id(self.index_to_id[int(i)]) for i in idx])
        return out

    def _scores_for_query(self, q: np.ndarray) -> np.ndarray:
        """MaxSim of one query (n_q, dim) against every catalog doc -> (n_docs,).

        Chunked over docs so the (n_q × tokens) similarity block stays bounded — the brute-force
        'one matmul' done in memory-safe slices. `np.maximum.reduceat` takes the per-doc max over
        each doc's contiguous token span in one vectorized call."""
        scores = np.empty(self._n_docs, dtype=np.float32)
        for a in range(0, self._n_docs, self.chunk_docs):
            b = min(a + self.chunk_docs, self._n_docs)
            tok_a, tok_b = int(self._offsets[a]), int(self._offsets[b])
            sims = q @ self._doc_matrix[tok_a:tok_b].T          # (n_q, tokens_in_chunk)
            local_starts = self._offsets[a:b] - tok_a           # per-doc start within the chunk
            per_doc_max = np.maximum.reduceat(sims, local_starts, axis=1)  # (n_q, b-a)
            scores[a:b] = per_doc_max.sum(axis=0)
        return scores
