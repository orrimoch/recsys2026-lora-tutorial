"""Stage B inference: MULTIMODAL_RERANKER (Phase 7).

Reranks a first-stage retriever's top-K candidates with the trained
``MultiModalCrossEncoder``. Implements the ``mcrs.rerankers`` contract:

    rerank(queries, candidate_tids, topk, user_ids=None, ...) -> list[list[str]]

Per (query, user), it scores every candidate by fusing the query text + user CF
(cold -> mean fallback) with each candidate's text + CLAP/CF/tag/release
modalities (looked up from ``experiments/cache/multimodal`` artifacts), then
returns the top-k tids by descending score. Scores are cached per
(query_text, user_id, candidate-set) so repeated calls are free.

The pure ordering + caching logic is unit-tested; the model-backed scoring
(``_score_candidates``) is an integration path that runs on Colab GPU.
"""
from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np


def rerank_by_scores(candidate_tids: list, scores, topk: int) -> list:
    """Return the top-k candidate tids ordered by descending score.

    Stable on ties (preserves input order), so a no-op reranker returns the
    first-stage order unchanged.
    """
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    return [candidate_tids[i] for i in order[:topk]]


def _pad_tag_ids(tag_id_lists: list, max_tags: int, pad: int = 0) -> list:
    out = []
    for ids in tag_id_lists:
        ids = list(ids)[:max_tags]
        out.append(ids + [pad] * (max_tags - len(ids)))
    return out


class MULTIMODAL_RERANKER:
    """Multi-modal cross-encoder reranker (Stage B inference)."""

    def __init__(
        self,
        model_dir: str,
        multimodal_artifacts: str,
        item_db_name: str,
        track_split_types: list[str],
        corpus_types: list[str],
        cache_dir: str = "./cache",
        batch_size: int = 32,
        max_length: int = 512,
        max_tags: int = 20,
        device: Optional[str] = None,
    ) -> None:
        import sys
        from pathlib import Path

        import torch

        repo_root = Path(__file__).resolve().parents[3]
        sys.path.insert(0, str(repo_root / "scripts"))
        from build_bi_encoder_training_data import (
            _format_history_music_turn,
            _load_multimodal_artifacts,
            _track_to_tag_ids,
        )
        from mcrs.training.multimodal_bi_encoder import MultiModalArtifacts
        from mcrs.training.multimodal_cross_encoder import MultiModalCrossEncoder

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.max_tags = max_tags

        # Trained Stage B model + tokenizer (tokenizer saved under backbone/).
        self.model = MultiModalCrossEncoder.from_pretrained(model_dir, device=device).eval()
        from transformers import AutoTokenizer
        tok_dir = os.path.join(model_dir, "backbone")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tok_dir if os.path.isdir(tok_dir) else model_dir
        )

        # Modality artifacts (CLAP/CF/user_cf) + tag vocab + release-year lookup.
        self.artifacts = MultiModalArtifacts(multimodal_artifacts)
        tag_vocab, release_year_lookup = _load_multimodal_artifacts(multimodal_artifacts)
        self._release_year_lookup = release_year_lookup

        # Per-track text + tag-id lookups (mirror the Stage A builder so the
        # reranker sees the same doc representation it trained on).
        from datasets import concatenate_datasets, load_dataset
        ds = load_dataset(item_db_name)
        concat = concatenate_datasets([ds[s] for s in track_split_types])
        metadata_dict = {row["track_id"]: dict(row) for row in concat}
        self.tid_to_text = {
            tid: _format_history_music_turn(tid, metadata_dict, corpus_types)
            for tid in metadata_dict
        }
        self.tid_to_tag_ids = {
            tid: _track_to_tag_ids(tid, metadata_dict, tag_vocab, max_tags)
            for tid in metadata_dict
        }
        print(f"[mm-rerank] loaded {model_dir} on {device}; "
              f"{len(self.tid_to_text)} tracks", flush=True)

        # Score cache: (query, user_id, sorted(candidate_ids)) -> {tid: score}.
        self._cache: dict = {}

    def _score_candidates(self, query: str, candidate_tids: list, user_id) -> np.ndarray:
        """Score every (query, candidate) pair with the cross-encoder.

        Returns a float score array aligned to ``candidate_tids``. Chunked by
        ``batch_size`` to bound activation memory on the ~570M backbone.
        """
        import torch

        user_cf = np.asarray(self.artifacts.get_user_cf(user_id), dtype=np.float32)
        scores: list[float] = []
        for start in range(0, len(candidate_tids), self.batch_size):
            chunk = candidate_tids[start:start + self.batch_size]
            n = len(chunk)
            q_enc = self.tokenizer([query] * n, padding=True, truncation=True,
                                   max_length=self.max_length, return_tensors="pt")
            doc_texts = [self.tid_to_text.get(t, "") for t in chunk]
            d_enc = self.tokenizer(doc_texts, padding=True, truncation=True,
                                   max_length=self.max_length, return_tensors="pt")
            doc_clap = np.stack([self.artifacts.get_track_clap(t) for t in chunk])
            doc_cf = np.stack([self.artifacts.get_track_cf(t) for t in chunk])
            doc_tags = _pad_tag_ids([self.tid_to_tag_ids.get(t, []) for t in chunk], self.max_tags)
            doc_year = [int(self._release_year_lookup.get(t, -1)) for t in chunk]
            dev = self.device
            with torch.no_grad():
                logits = self.model(
                    query_input_ids=q_enc["input_ids"].to(dev),
                    query_attention_mask=q_enc["attention_mask"].to(dev),
                    query_user_cf=torch.tensor(np.tile(user_cf, (n, 1)),
                                               dtype=torch.float32, device=dev),
                    doc_input_ids=d_enc["input_ids"].to(dev),
                    doc_attention_mask=d_enc["attention_mask"].to(dev),
                    doc_clap=torch.tensor(doc_clap, dtype=torch.float32, device=dev),
                    doc_cf=torch.tensor(doc_cf, dtype=torch.float32, device=dev),
                    doc_tags=torch.tensor(doc_tags, dtype=torch.long, device=dev),
                    doc_year=torch.tensor(doc_year, dtype=torch.long, device=dev),
                )
            scores.extend(logits.float().cpu().numpy().reshape(-1).tolist())
        return np.asarray(scores, dtype=np.float64)

    def rerank(
        self,
        queries: list[str],
        candidate_tids: list[list[str]],
        topk: int,
        user_ids: Optional[list] = None,
        **_kwargs: Any,
    ) -> list[list[str]]:
        """Per query, score its candidates and return the top-k tids.

        Side-channel kwargs (goal_categories, etc.) are accepted for interface
        parity and ignored. Scores are cached by (query, user_id, candidate-set),
        so the cache is independent of candidate input order.
        """
        out: list[list[str]] = []
        for i, (query, tids) in enumerate(zip(queries, candidate_tids)):
            user_id = user_ids[i] if user_ids is not None else None
            key = (query, user_id, tuple(sorted(tids)))
            cached = self._cache.get(key)
            if cached is not None:
                scores = [cached[t] for t in tids]
            else:
                scores = self._score_candidates(query, tids, user_id)
                self._cache[key] = dict(zip(tids, scores))
            out.append(rerank_by_scores(tids, scores, topk))
        return out
