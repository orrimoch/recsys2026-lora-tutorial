"""Two-tower recall channel (Tier-1 #3.3): encode the retrieval query with the
learned query tower, score it against the precomputed item-repr matrix by cosine,
return top-K. Standard retriever interface so the wRRF factory can use it as an
opt-in union channel (use_two_tower). Mirrors SasrecRetriever's shape.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class TwoTowerRetriever:
    def __init__(self, model, item_repr, track_ids, query_encode,
                 batch_size: int = 256):
        """model: TwoTowerModel (eval); item_repr: (N,d) item vectors (L2-
        normalized at build); track_ids: list aligned to item_repr rows;
        query_encode: list[str] -> (B, q_in_dim) frozen query embeddings."""
        self.model = model.eval() if hasattr(model, "eval") else model
        ir = item_repr if torch.is_tensor(item_repr) else torch.as_tensor(
            np.asarray(item_repr), dtype=torch.float32)
        self.item_repr = F.normalize(ir.float(), dim=1)  # cosine-safe
        self.track_ids = track_ids
        self.query_encode = query_encode
        self.batch_size = int(batch_size)

    def batch_text_to_item_retrieval(self, queries, topk, user_ids=None,
                                     batch_context=None):
        results: list[list[str]] = []
        for s in range(0, len(queries), self.batch_size):
            q_batch = queries[s:s + self.batch_size]
            q_emb = torch.as_tensor(np.asarray(self.query_encode(q_batch)),
                                    dtype=torch.float32)
            with torch.no_grad():
                q_vecs = self.model.encode_query(q_emb)            # (B,d)
                scores = q_vecs @ self.item_repr.t()               # (B,N) cosine
                k = min(topk, scores.shape[1])
                top = torch.topk(scores, k=k, dim=1).indices
            for row in top.tolist():
                results.append([self.track_ids[i] for i in row])
        return results

    def text_to_item_retrieval(self, query, topk, user_id=None):
        return self.batch_text_to_item_retrieval([query], topk=topk)[0]
