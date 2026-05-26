"""HyDE recall channel: conversation -> pseudo-tracks -> dense retrieval -> RRF.

Composes a HydeGenerator (text) with a dense inner retriever (DENSE_PRECOMPUTED
over metadata-qwen3) so pseudo-track descriptions match in the catalog's own
embedding space. Implements the standard retriever interface so the wRRF
factory can use it as a union channel.
"""
from __future__ import annotations


def rrf_fuse(ranked_lists: list[list[str]], rrf_k: int, topk: int) -> list[str]:
    scores: dict[str, float] = {}
    for lst in ranked_lists:
        for rank, tid in enumerate(lst):
            scores[tid] = scores.get(tid, 0.0) + 1.0 / (rrf_k + rank)
    ordered = sorted(scores.items(), key=lambda kv: -kv[1])
    return [tid for tid, _ in ordered[:topk]]


class HydeQwen3Retriever:
    def __init__(self, generator, inner_retriever,
                 topk_per_doc: int = 100, rrf_k: int = 60):
        self.generator = generator
        self.inner = inner_retriever
        self.topk_per_doc = int(topk_per_doc)
        self.rrf_k = int(rrf_k)

    def batch_text_to_item_retrieval(self, queries, topk, user_ids=None):
        gen = self.generator.generate_batch(queries)
        flat_docs: list[str] = []
        spans: list[tuple[int, int]] = []
        for g in gen:
            docs = g["hyde_docs"] or [g["intent_query"]]
            start = len(flat_docs)
            flat_docs.extend(docs)
            spans.append((start, len(flat_docs)))
        if not flat_docs:
            return [[] for _ in queries]
        hits = self.inner.batch_text_to_item_retrieval(flat_docs, topk=self.topk_per_doc)
        return [rrf_fuse(hits[s:e], self.rrf_k, topk) for (s, e) in spans]

    def text_to_item_retrieval(self, query, topk, user_id=None):
        return self.batch_text_to_item_retrieval([query], topk=topk)[0]
