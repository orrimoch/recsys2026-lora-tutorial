"""Reranker factory. Rerankers take a (query, candidate_tids) list and
return a reordered + truncated tids list. Plugged in after the primary
retriever in crs_baseline.batch_chat."""
from __future__ import annotations

from typing import Any, Optional


def load_reranker_module(
    reranker_type: Optional[str],
    item_db_name: str,
    track_split_types: list[str],
    corpus_types: list[str],
    cache_dir: str = "./cache",
) -> Optional[Any]:
    """Return a reranker instance or None if reranker_type is falsy.

    Each reranker exposes:
      rerank(queries: list[str], candidate_tids: list[list[str]], topk: int)
        -> list[list[str]]
    """
    if not reranker_type:
        return None
    if reranker_type == "bge_reranker_v2_m3":
        from .bge_reranker import BGE_RERANKER
        return BGE_RERANKER(
            item_db_name=item_db_name,
            track_split_types=track_split_types,
            corpus_types=corpus_types,
            cache_dir=cache_dir,
        )
    raise ValueError(f"Unsupported reranker type: {reranker_type}")
