"""BGE-reranker-v2-m3 cross-encoder: scores (query, track-text) pairs,
reorders top-K candidates from the primary retriever.

Model: BAAI/bge-reranker-v2-m3 — ~568M params, works bf16 on CUDA/MPS,
fp32 on CPU. Multilingual, robust, widely-used 2024 SOTA pointwise
cross-encoder.

Track text = pipe-joined metadata (track_name | artist_name | album_name
[| tag_list]). Cached at init-time; reused across queries.
"""
from __future__ import annotations

import os
import pickle
from typing import Any, Optional

import numpy as np


MODEL_NAME = "BAAI/bge-reranker-v2-m3"


def build_tid_text_map(metadata_dict: dict, corpus_types: list[str]) -> dict[str, str]:
    """Pipe-joined CE doc text: "name | artist | album | tag, tag". List fields
    comma-joined; empty/None dropped. SINGLE source of truth shared by serve
    (BGE_RERANKER) and the CE training-data builder so (query,doc) is identical."""
    out: dict[str, str] = {}
    for tid, row in metadata_dict.items():
        parts: list[str] = []
        for f in corpus_types:
            v = row.get(f)
            if isinstance(v, list):
                v = ", ".join(str(x) for x in v if x is not None)
            if v:
                parts.append(str(v))
        out[tid] = " | ".join(parts)
    return out


class BGE_RERANKER:
    def __init__(
        self,
        item_db_name: str,
        track_split_types: list[str],
        corpus_types: list[str],
        cache_dir: str = "./cache",
        model_name: Optional[str] = None,
        max_length: int = 256,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        # Device pick: CUDA > MPS > CPU. bf16 on accelerators, fp32 on CPU.
        if torch.cuda.is_available():
            self.device = "cuda"
            dtype = torch.bfloat16
        elif torch.backends.mps.is_available():
            self.device = "mps"
            dtype = torch.bfloat16
        else:
            self.device = "cpu"
            dtype = torch.float32
        self.dtype = dtype
        self.max_length = max_length

        # Default to MODEL_NAME (public BGE reranker); override via `model_name`
        # to load our fine-tuned Hub weights.
        resolved = model_name or MODEL_NAME
        self.tokenizer = AutoTokenizer.from_pretrained(resolved)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            resolved, torch_dtype=dtype
        ).to(self.device).eval()
        print(f"[bge-rerank] loaded {resolved} on {self.device} dtype={dtype}")

        self.tid_to_text = self._load_or_build_tid_text(
            item_db_name, track_split_types, corpus_types, cache_dir
        )

    def _load_or_build_tid_text(
        self,
        item_db_name: str,
        track_split_types: list[str],
        corpus_types: list[str],
        cache_dir: str,
    ) -> dict[str, str]:
        # Cache key: (item_db, splits, fields). Shared across experiments.
        safe = item_db_name.replace("/", "_")
        fields_tag = "_".join(sorted(corpus_types))
        splits_tag = "_".join(sorted(track_split_types))
        cache_path = os.path.join(
            cache_dir, "rerank", f"{safe}__{splits_tag}__{fields_tag}__tid_text.pkl"
        )
        if os.path.isfile(cache_path):
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            print(f"[bge-rerank] loaded tid-text cache: {len(cache)} entries")
            return cache

        from datasets import concatenate_datasets, load_dataset

        print(f"[bge-rerank] building tid-text map from {item_db_name}[{track_split_types}] fields={corpus_types}")
        ds = load_dataset(item_db_name)
        concat = concatenate_datasets([ds[s] for s in track_split_types])
        metadata_dict = {row["track_id"]: row for row in concat}
        tid_to_text = build_tid_text_map(metadata_dict, corpus_types)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(tid_to_text, f)
        print(f"[bge-rerank] cached {len(tid_to_text)} tid-text entries at {cache_path}")
        return tid_to_text

    def _score_batch(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        import torch

        queries = [p[0] for p in pairs]
        docs = [p[1] for p in pairs]
        inputs = self.tokenizer(
            queries, docs,
            padding=True, truncation=True, max_length=self.max_length, return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits.view(-1)
        return logits.float().cpu().numpy()

    def rerank(
        self, queries: list[str], candidate_tids: list[list[str]], topk: int,
        # Side-channel kwargs (user_ids etc.) accepted for interface parity
        # with LGBM_RERANKER. BGE is query-driven only — we ignore them.
        **_kwargs: object,
    ) -> list[list[str]]:
        """Per query, score its candidates with the cross-encoder and return
        the top-k tids sorted by score descending."""
        out: list[list[str]] = []
        for q, tids in zip(queries, candidate_tids):
            pairs = [(q, self.tid_to_text.get(t, "")) for t in tids]
            scores = self._score_batch(pairs)
            order = np.argsort(-scores)
            keep = order[:topk]
            out.append([tids[i] for i in keep])
        return out
