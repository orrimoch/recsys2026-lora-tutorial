"""Session-level collaborative recall channel. Candidates = catalog tracks whose
cf-bpr vector is nearest to the centroid of the session's played tracks. (Session
CF centroid recall@100 ~0.24 vs near-inert user-level CF.)"""
from __future__ import annotations

import numpy as np

from .session_history import played_tids_from_context


class SessionCFRetriever:
    def __init__(self, dataset_name, split_types, corpus_types, cache_dir="./cache"):
        from .cf_bpr import CF_BPR
        donor = CF_BPR(dataset_name, split_types, corpus_types, cache_dir)
        self.track_ids = donor.track_ids
        self.track_mat = donor.track_mat            # already L2-normalized (47k x 128)
        self.tid_to_idx = {t: i for i, t in enumerate(self.track_ids)}
        self.catalog_tids = set(self.track_ids)

    def batch_text_to_item_retrieval(self, queries, topk, user_ids=None,
                                     batch_context=None):
        out = []
        for i in range(len(queries)):
            ctx = batch_context[i] if batch_context else None
            played = played_tids_from_context(ctx, self.catalog_tids)
            idxs = [self.tid_to_idx[t] for t in played if t in self.tid_to_idx]
            if not idxs:
                out.append([])
                continue
            centroid = self.track_mat[idxs].mean(axis=0)
            n = np.linalg.norm(centroid)
            if n < 1e-9:
                out.append([])
                continue
            centroid = centroid / n
            scores = self.track_mat @ centroid
            order = np.argsort(-scores)
            played_set = set(played)
            ranked = []
            for j in order:
                tid = self.track_ids[j]
                if tid not in played_set:
                    ranked.append(tid)
                    if len(ranked) >= topk:
                        break
            out.append(ranked)
        return out

    def text_to_item_retrieval(self, query, topk, user_id=None):
        return self.batch_text_to_item_retrieval([query], topk=topk)[0]
