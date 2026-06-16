"""Response-reranker factory.

Response rerankers sit AFTER `lm.batch_response_generation_multi` and BEFORE
the final response is written to prediction.json. They consume K candidate
responses per query (sampled at varied temperatures) and return the single
best response, scored by a domain-specific reward signal.

Used by exp 026+: the reward-model reranker trained on train
goal_progress_assessments. See colab/Train_Reward_Model.ipynb for training.
"""
from __future__ import annotations

from typing import Any, Optional


def load_response_reranker_module(
    reranker_type: Optional[str],
    model_path: Optional[str] = None,
) -> Optional[Any]:
    """Return a response reranker instance, or None if reranker_type is falsy.

    Each reranker exposes:
      rerank(contexts: list[str], candidates: list[list[str]]) -> list[int]
        — per query, returns the index of the chosen candidate.
    """
    if not reranker_type:
        return None
    if reranker_type == "reward_model":
        from .reward_reranker import REWARD_MODEL_RERANKER
        if not model_path:
            raise ValueError(
                "response_reranker_type=reward_model requires "
                "response_reranker_model_path in yaml (local dir with the "
                "fine-tuned cross-encoder)."
            )
        return REWARD_MODEL_RERANKER(model_path=model_path)
    raise ValueError(f"Unsupported response_reranker_type: {reranker_type}")
