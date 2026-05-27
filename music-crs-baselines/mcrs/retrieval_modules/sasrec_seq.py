"""SASRec recall channel: dialog + played history -> session state -> rank the
precomputed catalog item-repr matrix. Standard retriever interface so the wRRF
factory can use it as a union channel."""
from __future__ import annotations

import numpy as np
import torch


class SasrecRetriever:
    def __init__(self, model, item_repr, track_ids, item_feats, text_encode,
                 max_len: int = 50, batch_size: int = 256):
        self.model = model.eval()
        self.item_repr = item_repr                      # (N, d) tensor
        self.track_ids = track_ids
        self.item_feats = item_feats                    # (N, item_in_dim) tensor
        self.tid_to_idx = {t: i for i, t in enumerate(track_ids)}
        self.text_encode = text_encode                  # list[str] -> (B, ctx_in_dim) np
        self.max_len = int(max_len)
        self.batch_size = int(batch_size)
        self.item_in_dim = item_feats.shape[1]

    def _histories_to_feats(self, histories):
        """Build (B, L, item_in_dim) item-feature tensor + (B,) lengths from each
        row's history_tids (last max_len, unknown ids dropped)."""
        idx_lists = [
            [self.tid_to_idx[t] for t in (h or []) if t in self.tid_to_idx][-self.max_len:]
            for h in histories
        ]
        L = max((len(x) for x in idx_lists), default=0)
        L = max(L, 1)  # keep a non-zero time dim for the encoder
        feats = torch.zeros(len(histories), L, self.item_in_dim)
        lengths = torch.zeros(len(histories), dtype=torch.long)
        for b, ids in enumerate(idx_lists):
            lengths[b] = len(ids)
            if ids:
                feats[b, :len(ids)] = self.item_feats[ids]
        return feats, lengths

    def batch_text_to_item_retrieval(self, queries, topk, user_ids=None, batch_context=None):
        ctxs = batch_context or [{} for _ in queries]
        results: list[list[str]] = []
        for s in range(0, len(queries), self.batch_size):
            q_batch = queries[s:s + self.batch_size]
            c_batch = ctxs[s:s + self.batch_size]
            # dialog text = user-turns-only from context, falling back to the query
            dialog_batch = [(c.get("user_dialog") or q) for c, q in zip(c_batch, q_batch)]
            h_batch = [c.get("history_tids", []) for c in c_batch]
            ctx_emb = torch.as_tensor(np.asarray(self.text_encode(dialog_batch)), dtype=torch.float32)
            feats, lengths = self._histories_to_feats(h_batch)
            with torch.no_grad():
                state = self.model.encode(ctx_emb, feats, lengths)
                scores = self.model.score(state, self.item_repr)
                top = torch.topk(scores, k=min(topk, scores.shape[1]), dim=1).indices
            for row in top.tolist():
                results.append([self.track_ids[i] for i in row])
        return results

    def text_to_item_retrieval(self, query, topk, user_id=None):
        return self.batch_text_to_item_retrieval([query], topk=topk)[0]
