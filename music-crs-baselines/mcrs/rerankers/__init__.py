"""Reranker factory. Rerankers take a (query, candidate_tids) list and
return a reordered + truncated tids list. Plugged in after the primary
retriever in crs_baseline.batch_chat.

All rerankers accept the side-channel kwargs user_ids / goal_categories /
goal_specificities / user_profiles_raw in their rerank() method so the
batch_chat call site can forward session context uniformly. Rerankers
that don't use a given channel accept-and-ignore.
"""
from __future__ import annotations

from typing import Any, Optional


def load_reranker_module(
    reranker_type: Optional[str],
    item_db_name: str,
    track_split_types: list[str],
    corpus_types: list[str],
    cache_dir: str = "./cache",
    model_path: Optional[str] = None,
) -> Optional[Any]:
    """Return a reranker instance or None if reranker_type is falsy.

    Each reranker exposes:
      rerank(queries, candidate_tids, topk, user_ids=None,
             goal_categories=None, goal_specificities=None,
             user_profiles_raw=None) -> list[list[str]]
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
    if reranker_type == "lgbm_rerank":
        from .lgbm_rerank import LGBM_RERANKER
        if not model_path:
            raise ValueError(
                "reranker_type=lgbm_rerank requires reranker_model_path in the "
                "yaml (local dir holding booster.txt + metadata.json from "
                "colab/Build_LGBM_Features_And_Train.ipynb)."
            )
        return LGBM_RERANKER(
            item_db_name=item_db_name,
            track_split_types=track_split_types,
            corpus_types=corpus_types,
            cache_dir=cache_dir,
            model_path=model_path,
        )
    if reranker_type == "pro_rank":
        # W3 default reranker — last-token-logit-diff scoring over Qwen-0.5B.
        # `model_path` (optional) loads a LoRA adapter on top of the base model
        # for the trained ProRank policy. Without it, runs inference-only on
        # the base-Qwen-0.5B (paper-equivalent inference behaviour minus the
        # GRPO policy warmup).
        from .pro_rank import ProRankReranker
        return ProRankReranker(
            item_db_name=item_db_name,
            track_split_types=track_split_types,
            corpus_types=corpus_types,
            cache_dir=cache_dir,
            model_path=model_path,
        )
    raise ValueError(f"Unsupported reranker type: {reranker_type}")
