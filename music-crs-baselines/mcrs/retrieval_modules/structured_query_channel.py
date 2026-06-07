"""Structured-query recall channel (Tier-1 #3.1b): conversation -> LLM content-only
synthetic query -> dense retrieval. Standard retriever interface so the wRRF
factory can use it as an opt-in union channel (use_structured_query).

Composes a StructuredQueryExtractor with a dense inner retriever (DENSE_PRECOMPUTED
over a Qwen3 field) so the synthetic query matches in the catalog's embedding
space — same composition shape as HydeQwen3Retriever.
"""
from __future__ import annotations


class StructuredQueryRetriever:
    def __init__(self, extractor, inner_retriever, topk_internal: int = 100):
        self.extractor = extractor
        self.inner = inner_retriever
        self.topk_internal = int(topk_internal)

    def batch_text_to_item_retrieval(self, queries, topk, user_ids=None,
                                     batch_context=None):
        # Per-query session/turn (for the extractor's cache) when available.
        session_ids = turn_numbers = None
        if batch_context:
            session_ids = [(c or {}).get("session_id") for c in batch_context]
            turn_numbers = [(c or {}).get("turn_number") for c in batch_context]
        synthetic = self.extractor.extract_batch(
            queries, session_ids=session_ids, turn_numbers=turn_numbers)
        return self.inner.batch_text_to_item_retrieval(
            synthetic, topk=topk, user_ids=user_ids)

    def text_to_item_retrieval(self, query, topk, user_id=None):
        return self.batch_text_to_item_retrieval([query], topk=topk)[0]
