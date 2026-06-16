"""Sequential retrieve-then-rerank: BM25 first stage → dense cosine rerank.

v4.1 (tid=008). Rather than parallel RRF fusion (v4, tid=007), this
composition lets BM25 produce the candidate pool and dense re-score only
those candidates. BM25's strong tail recall is preserved; dense's +63%
rel precision@1 signal (observed in v4) rearranges the head. No
dilution from dense's weaker tail candidates.

Grounding: Nogueira & Cho 2019 "Passage Re-ranking with BERT";
Thakur et al. 2021 "BEIR" (retrieve-then-rerank outperforms single-stage
fusion on most OOD tasks).

Interface matches BM25_MODEL so the factory can hand back a
`SEQUENTIAL_RERANK` anywhere a retriever is expected.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class SEQUENTIAL_RERANK:
    def __init__(
        self,
        dataset_name: str,
        split_types: list[str],
        corpus_types: list[str],
        cache_dir: str,
        first_stage_spec: dict[str, Any],
        reranker_spec: dict[str, Any],
    ) -> None:
        """
        first_stage_spec: {"type": str, "corpus_types": list, "topk_internal": int}
        reranker_spec:    {"embed_col": str, "instruct": Optional[str], "instruct_label": str}
                          — direct kwargs for DENSE_PRECOMPUTED
        """
        from . import load_retrieval_module
        from .dense_precomputed import DENSE_PRECOMPUTED

        self.first = load_retrieval_module(
            first_stage_spec["type"],
            dataset_name, split_types,
            first_stage_spec.get("corpus_types", corpus_types),
            cache_dir,
        )
        self.first_topk = int(first_stage_spec.get("topk_internal", 100))

        self.dense = DENSE_PRECOMPUTED(
            dataset_name, split_types, corpus_types, cache_dir,
            embed_col=reranker_spec["embed_col"],
            instruct=reranker_spec.get("instruct"),
            instruct_label=reranker_spec.get("instruct_label", "raw"),
        )
        # Precompute track_id → row index for fast candidate lookup during rerank.
        self._tid_to_idx: dict[str, int] = {
            tid: i for i, tid in enumerate(self.dense.track_ids)
        }
        print(
            f"[seq-rerank] first_stage={first_stage_spec['type']} "
            f"topk_internal={self.first_topk} → dense rerank "
            f"({reranker_spec['embed_col']})"
        )

    def batch_text_to_item_retrieval(
        self, queries: list[str], topk: int
    ) -> list[list[str]]:
        # 1. First-stage retrieves a larger candidate pool per query.
        print(f"[seq-rerank] running first stage for {len(queries)} queries, "
              f"topk_internal={self.first_topk}")
        first_results = self.first.batch_text_to_item_retrieval(queries, topk=self.first_topk)

        # 2. Use the dense retriever's own cached encoding. Calling its
        #    batch_text_to_item_retrieval populates the query cache; we just
        #    discard its result and pull the embeddings directly from the cache.
        #    topk=1 is the minimum valid argument; the returned list is thrown
        #    away, but the encoding is preserved in self.dense._query_cache.
        print("[seq-rerank] encoding queries via cached dense retriever")
        _ = self.dense.batch_text_to_item_retrieval(queries, topk=1)
        q_mat = np.stack(
            [self.dense._query_cache[q] for q in queries], axis=0
        ).astype(np.float32)

        # 3. For each query, restrict dense scoring to first-stage candidates.
        results: list[list[str]] = []
        for q_idx, cands in enumerate(first_results):
            rows: list[np.ndarray] = []
            kept_cands: list[str] = []
            for tid in cands:
                idx = self._tid_to_idx.get(tid)
                if idx is None:
                    continue
                rows.append(self.dense.track_mat[idx])
                kept_cands.append(tid)
            if not rows:
                # First-stage produced nothing the dense knows about — fall back
                # to first-stage order (should be impossible given imputation
                # coverage, but stay defensive).
                results.append(cands[:topk])
                continue
            mat = np.stack(rows, axis=0)
            scores = mat @ q_mat[q_idx]
            order = np.argsort(-scores)
            reranked = [kept_cands[int(j)] for j in order[:topk]]
            results.append(reranked)
        return results

    def text_to_item_retrieval(self, query: str, topk: int) -> list[str]:
        return self.batch_text_to_item_retrieval([query], topk=topk)[0]
