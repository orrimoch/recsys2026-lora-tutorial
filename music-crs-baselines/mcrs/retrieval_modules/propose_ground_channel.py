"""RAG propose-then-ground recall channel (Tier-1 #3.5): an LLM proposes real
"Artist - Title" tracks; each is grounded to a catalog track by dense NN; the
per-query grounded lists are RRF-fused. Standard retriever interface so the wRRF
factory can use it as an opt-in union channel (use_propose_ground). Mirrors
HydeQwen3Retriever's composition shape.
"""
from __future__ import annotations

from .hyde_qwen3 import rrf_fuse


class ProposeGroundRetriever:
    def __init__(self, generator, inner_retriever,
                 topk_per_proposal: int = 100, rrf_k: int = 60):
        self.generator = generator
        self.inner = inner_retriever
        self.topk_per_proposal = int(topk_per_proposal)
        self.rrf_k = int(rrf_k)

    def batch_text_to_item_retrieval(self, queries, topk, user_ids=None,
                                     batch_context=None):
        gen = self.generator.generate_batch(queries)
        flat: list[str] = []
        spans: list[tuple[int, int]] = []
        for g in gen:
            props = g.get("proposals") or []
            start = len(flat)
            flat.extend(props)
            spans.append((start, len(flat)))
        if not flat:
            return [[] for _ in queries]
        hits = self.inner.batch_text_to_item_retrieval(flat, topk=self.topk_per_proposal)
        out = []
        for (s, e) in spans:
            out.append(rrf_fuse(hits[s:e], self.rrf_k, topk) if e > s else [])
        return out

    def text_to_item_retrieval(self, query, topk, user_id=None):
        return self.batch_text_to_item_retrieval([query], topk=topk)[0]
