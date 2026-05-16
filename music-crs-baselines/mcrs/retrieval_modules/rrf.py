"""Reciprocal Rank Fusion of multiple sub-retrievers.

Cormack, Clarke, Büttcher 2009. For each candidate d and each
sub-retriever r, fuse scores via:

    score(d) = sum_r  1 / (k + rank_r(d))

where rank_r(d) is the 1-indexed rank of d in r's top-K list (infinity
if not in the list, contributing 0). Documents not ranked by any sub
are implicitly excluded.

Standard k=60 per the original paper.

Interface matches `BM25_MODEL` so the factory can return an RRF
retriever wherever any other retriever would go.
"""
from __future__ import annotations

from typing import Any, Optional


class RRF_MODEL:
    def __init__(
        self,
        dataset_name: str,
        split_types: list[str],
        corpus_types: list[str],
        cache_dir: str = "./cache",
        sub_specs: list[dict[str, Any]] | None = None,
        k: int = 60,
    ) -> None:
        """Assemble sub-retrievers from the factory.

        Each spec in `sub_specs`:
          - "type":           retrieval_type key for load_retrieval_module
          - "corpus_types":   (optional) override this sub's corpus_types
                              (useful: BM25 sub wants 5 fields with tag_list
                              while the outer RRF corpus_types may be narrower
                              for MusicCatalogDB compatibility)
          - "topk_internal":  how many candidates to pull from the sub before fusion
          - "weight":         (optional, default 1.0) weight applied to this
                              sub's RRF contribution. Use to give one
                              retriever more mass when its standalone
                              quality differs from another's. Weight==1.0
                              recovers vanilla symmetric RRF.
        """
        from . import load_retrieval_module

        self.k = k
        if sub_specs is None:
            raise ValueError("RRF_MODEL requires sub_specs")
        self.subs: list[dict[str, Any]] = []
        for spec in sub_specs:
            retriever_type = spec["type"]
            sub_corpus = spec.get("corpus_types", corpus_types)
            sub_topk = int(spec.get("topk_internal", 60))
            weight = float(spec.get("weight", 1.0))
            print(
                f"[rrf] building sub retriever={retriever_type} "
                f"topk_internal={sub_topk} weight={weight:.2f}"
            )
            sub = load_retrieval_module(
                retriever_type, dataset_name, split_types, sub_corpus, cache_dir
            )
            self.subs.append({
                "retriever": sub, "topk": sub_topk,
                "weight": weight, "label": retriever_type,
            })
        print(f"[rrf] ready — k={self.k}, {len(self.subs)} sub-retriever(s)")

    def batch_text_to_item_retrieval(
        self, queries: list[str], topk: int, user_ids=None,
        batch_context: Optional[list[dict]] = None,
    ) -> list[list[str]]:
        # user_ids and batch_context threaded through to any sub that uses them.
        # Subs that ignore them (BM25, dense) still accept via try/except back-compat.
        per_sub: list[list[list[str]]] = []
        for sub in self.subs:
            print(f"[rrf] running sub: {sub['label']}")
            try:
                sub_results = sub["retriever"].batch_text_to_item_retrieval(
                    queries, topk=sub["topk"],
                    batch_context=batch_context, user_ids=user_ids,
                )
            except TypeError:
                # Back-compat: sub-retriever doesn't accept batch_context yet.
                try:
                    sub_results = sub["retriever"].batch_text_to_item_retrieval(
                        queries, topk=sub["topk"], user_ids=user_ids,
                    )
                except TypeError:
                    sub_results = sub["retriever"].batch_text_to_item_retrieval(
                        queries, topk=sub["topk"],
                    )
            per_sub.append(sub_results)

        results: list[list[str]] = []
        for q_idx in range(len(queries)):
            fused: dict[str, float] = {}
            for s_idx, sub in enumerate(self.subs):
                w = sub["weight"]
                ranks = per_sub[s_idx][q_idx]
                for rank, tid in enumerate(ranks, start=1):
                    fused[tid] = fused.get(tid, 0.0) + w / (self.k + rank)
            ordered = sorted(fused.items(), key=lambda kv: -kv[1])
            results.append([tid for tid, _ in ordered[:topk]])
        return results

    def text_to_item_retrieval(self, query: str, topk: int, user_id=None) -> list[str]:
        return self.batch_text_to_item_retrieval(
            [query], topk=topk, user_ids=[user_id] if user_id is not None else None,
        )[0]
