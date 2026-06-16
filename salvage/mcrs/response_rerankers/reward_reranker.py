"""Reward-model cross-encoder response reranker.

Loads a CrossEncoder fine-tuned on train's goal_progress_assessments labels
(see colab/Train_Reward_Model.ipynb). At inference time, for each query
scores K candidate responses via `(context, response)` pairs and returns
the index of the highest-scored candidate.

Expected directory layout of `model_path`:
  model_path/
    config.json
    pytorch_model.bin (or safetensors)
    tokenizer files
    sentence_bert_config.json   (from sentence-transformers save)

Usage inside CRS_BASELINE.batch_chat is orchestrated by the response_reranker
plumbing; this class only knows how to score + pick.
"""
from __future__ import annotations

import os
from typing import Optional


class REWARD_MODEL_RERANKER:
    def __init__(self, model_path: str) -> None:
        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"reward-model directory not found: {model_path}. "
                "Train via colab/Train_Reward_Model.ipynb then unzip into "
                "models/reward_model/."
            )
        import torch
        from sentence_transformers import CrossEncoder

        # Device pick matches the rest of the pipeline.
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        self.model = CrossEncoder(model_path, max_length=384, device=device)
        self.device = device
        print(f"[reward-rerank] loaded from {model_path} on {device}")

    def rerank(
        self,
        contexts: list[str],
        candidates: list[list[str]],
    ) -> list[int]:
        """Per query, score each candidate response; return best-index per query.

        contexts[i]: the 'text_a' string the candidates were generated for
            (query + listener_goal + recommended_track_meta + history).
        candidates[i]: K response strings for query i.
        Returns: list of K-indices, one per query (0-based).
        """
        if len(contexts) != len(candidates):
            raise ValueError(f"len(contexts)={len(contexts)} != len(candidates)={len(candidates)}")

        # Flatten into (context, response) pairs; score once.
        pairs: list[list[str]] = []
        spans: list[tuple[int, int]] = []  # per-query (start, end) in flat list
        cursor = 0
        for cand_list in candidates:
            spans.append((cursor, cursor + len(cand_list)))
            for resp in cand_list:
                pairs.append([contexts[len(spans) - 1], resp])
            cursor += len(cand_list)

        # predict() returns raw logits. Higher = more likely to MOVES_TOWARD_GOAL.
        scores = self.model.predict(pairs, batch_size=64, show_progress_bar=False)

        # Per query, pick argmax.
        choices: list[int] = []
        for start, end in spans:
            window = scores[start:end]
            best = int(max(range(len(window)), key=lambda i: window[i]))
            choices.append(best)
        return choices
