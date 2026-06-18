"""Stage 3 — full-catalog PLAID index + retrieval for the fine-tuned ColBERT (the G3 recall gate).

The dev-eval callback only POOL-reranks (can't surface a gold the pool missed). To test whether the
fine-tune actually raises recall we must retrieve over the WHOLE catalog — that's what this builds.
PLAID (not Voyager): end-to-end, no ef_search/pad_sequence or empty-cache footguns (memory:
use-plaid-not-voyager). The cache self-heals via a 1-query smoke test, like the probe.

All functions here are GPU/pylate integration (run on Colab), so pragma: no cover.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Sequence

from mcrs.data.ids import canonical_track_id
from mcrs.training.colbert_finetune import existing_artifact_blocks


def _index_usable(model, retriever) -> bool:  # pragma: no cover
    """A crashed/partial build leaves an EMPTY index that only errors at query time. Smoke-test one
    query so a bad cache self-heals into a rebuild instead of wedging the gate."""
    try:
        q = model.encode(["test"], is_query=True, show_progress_bar=False)
        retriever.retrieve(queries_embeddings=q, k=1)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  cached index failed smoke-test ({type(e).__name__}): {e}")
        return False


def build_or_load_plaid(model, catalog, index_folder: str, index_name: str,
                        doc_text_fn: Callable[[str], str], *, batch_size: int = 256,
                        force: bool = False):  # pragma: no cover
    """Build (or load+validate) a PyLate PLAID index over the full catalog with `model`. `doc_text_fn`
    renders each tid to its ColBERT doc (same recipe as train, e.g. colbert_doc_text(cat, t,
    expansion_first=True)). Returns a `retrieve.ColBERT` retriever. Self-heals an empty/corrupt cache."""
    from pylate import indexes, retrieve

    path = os.path.join(index_folder, index_name)

    def _build():
        ids = list(catalog.index_to_id)
        texts = [doc_text_fn(t) for t in ids]
        print(f"encoding {len(ids)} docs (one-time)...")
        emb = model.encode(texts, batch_size=batch_size, is_query=False, show_progress_bar=True)
        idx = indexes.PLAID(index_folder=index_folder, index_name=index_name, override=True)
        idx.add_documents(documents_ids=ids, documents_embeddings=emb)
        print("built + cached PLAID index ->", path)
        return idx

    index = None
    if os.path.exists(path) and not force:
        try:
            cand = indexes.PLAID(index_folder=index_folder, index_name=index_name, override=False)
            if _index_usable(model, retrieve.ColBERT(index=cand)):
                index = cand
                print("loaded cached PLAID index <-", path)
        except Exception as e:  # noqa: BLE001
            print(f"cached index load failed ({type(e).__name__}): {e}")
    if index is None:
        index = _build()
    return retrieve.ColBERT(index=index)


def colbert_retrieve(model, retriever, queries: Sequence[str], k: int,
                     batch_size: int = 128) -> list[list[str]]:  # pragma: no cover
    """Encode queries and retrieve top-k canonical track ids per query over the full-catalog index."""
    q = model.encode(list(queries), batch_size=batch_size, is_query=True, show_progress_bar=True)
    results = retriever.retrieve(queries_embeddings=q, k=k)
    return [[canonical_track_id(h["id"]) for h in q_hits] for q_hits in results]
