"""R3 — BM25 sparse channel over (optionally enriched) catalog docs.

Implements F2 RetrievalChannel. Doc text comes only from Catalog.id_to_metadata (single source;
A1 enrichment via enriched=True). Returns canonical track_ids. CPU, deterministic.
"""
from __future__ import annotations

from typing import Optional

from mcrs.data.catalog import Catalog
from mcrs.data.ids import canonical_track_id


class BM25Channel:
    label = "bm25"

    def __init__(self, catalog: Catalog, enriched: bool = False) -> None:
        import bm25s

        self._catalog = catalog
        self.index_to_id = list(catalog.index_to_id)
        corpus = [catalog.id_to_metadata(t, enriched=enriched).lower() for t in self.index_to_id]
        self._bm = bm25s.BM25()
        self._bm.index(bm25s.tokenize(corpus, show_progress=False), show_progress=False)

    def batch_text_to_item_retrieval(
        self, queries: list[str], topk: int,
        batch_context: Optional[list[dict]] = None, user_ids: Optional[list[str]] = None,
    ) -> list[list[str]]:
        import bm25s

        if not queries:
            return []
        k = min(topk, len(self.index_to_id))
        qtokens = bm25s.tokenize([q.lower() for q in queries], show_progress=False)
        results, _scores = self._bm.retrieve(qtokens, k=k, show_progress=False)
        out: list[list[str]] = []
        for row in results:  # row = array of corpus indices (== index_to_id order)
            out.append([canonical_track_id(self.index_to_id[int(i)]) for i in row])
        return out
