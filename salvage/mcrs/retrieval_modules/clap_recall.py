"""CLAP audio->session RECALL channel (wRRF sub-retriever).

For each turn it mean-pools the session's PLAYED tracks' CLAP audio vectors into
one query (clap_session_query) and returns the catalog tracks whose CLAP audio is
nearest — "more tracks that SOUND like what they've been playing." This reaches
new-artist golds the lexical/metadata/session channels miss (nb74 Stage 23 probe:
rescues 269 union-missed NEW-ARTIST golds, +0.035 recall@100 ceiling). A cold turn
(no played track has a CLAP vector) contributes []. DISTINCT from the rejected
clap_session_sim reranker FEATURE — this ADDS candidates rather than reordering an
existing pool, so it is not affected by the in-sample reranker leak.
"""
from __future__ import annotations

import numpy as np

from .clap_similarity import clap_session_query, load_clap_lookup
from .session_history import played_tids_from_context


class ClapRecallRetriever:
    def __init__(self, dataset_name, split_types, corpus_types, cache_dir="./cache"):
        self._build(load_clap_lookup(cache_dir))

    def _build(self, clap_lookup) -> None:
        self.lookup = clap_lookup
        self.tids = list(clap_lookup.keys())
        self.mat = (np.stack([clap_lookup[t] for t in self.tids]).astype(np.float32)
                    if self.tids else np.zeros((0, 1), dtype=np.float32))
        self.catalog_tids = set(self.tids)

    def batch_text_to_item_retrieval(self, queries, topk, user_ids=None,
                                     batch_context=None):
        out = []
        for i in range(len(queries)):
            ctx = batch_context[i] if batch_context else None
            played = played_tids_from_context(ctx, self.catalog_tids)
            q = clap_session_query(played, self.lookup) if played else None
            if q is None or self.mat.shape[0] == 0:
                out.append([])
                continue
            sims = self.mat @ q                      # (Ncat,), cosine (both L2-normed)
            played_set = set(played)
            kf = min(int(topk) + 300, self.mat.shape[0])  # margin to drop played, then take topk
            idx = np.argpartition(-sims, kf - 1)[:kf]
            order = idx[np.argsort(-sims[idx])]
            picks = []
            for j in order:
                t = self.tids[j]
                if t in played_set:
                    continue
                picks.append(t)
                if len(picks) >= int(topk):
                    break
            out.append(picks)
        return out

    def text_to_item_retrieval(self, query, topk, user_id=None):
        return self.batch_text_to_item_retrieval([query], topk=topk)[0]
