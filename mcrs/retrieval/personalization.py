"""R5 — warm-user personalization channels (F2 RetrievalChannel each).

content-kNN (recency-pooled history embeddings → nearest catalog tracks), CF (user_emb · track_emb),
and same-artist (other tracks by the user's history artists). All return [] for cold users
(empty history / missing CF vector) so they contribute 0 to RRF — clean cold fallback.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from mcrs.data.catalog import Catalog
from mcrs.data.embeddings import TrackEmbeddings, UserEmbeddings
from mcrs.data.ids import canonical_track_id
from mcrs.retrieval._vec import l2_normalize, topk_indices


def _history(batch_context, i) -> list[str]:
    if batch_context and i < len(batch_context):
        return list((batch_context[i] or {}).get("history_tids") or [])
    return []


class ContentKNNChannel:
    label = "content_knn"

    def __init__(self, track_emb: TrackEmbeddings, modality: str, normalize: bool = True) -> None:
        self.index_to_id = track_emb.index_to_id
        self.id_to_index = track_emb.id_to_index
        self.normalize = normalize
        m = track_emb.matrix(modality)
        self.matrix = l2_normalize(m) if normalize else m

    def batch_text_to_item_retrieval(self, queries, topk, batch_context=None, user_ids=None):
        out: list[list[str]] = []
        for i in range(len(queries)):
            idxs = [self.id_to_index[h] for h in _history(batch_context, i) if h in self.id_to_index]
            if not idxs:
                out.append([])
                continue
            pool = self.matrix[idxs].mean(axis=0)
            if self.normalize:
                pool = l2_normalize(pool)
            sims = self.matrix @ pool
            out.append([canonical_track_id(self.index_to_id[int(j)]) for j in topk_indices(sims, topk)])
        return out


class CFChannel:
    label = "cf"

    def __init__(self, user_emb: UserEmbeddings, track_emb: TrackEmbeddings,
                 modality: str = "cf-bpr", normalize: bool = True) -> None:
        self.user_emb = user_emb
        self.index_to_id = track_emb.index_to_id
        self.normalize = normalize
        m = track_emb.matrix(modality)
        self.matrix = l2_normalize(m) if normalize else m

    def batch_text_to_item_retrieval(self, queries, topk, batch_context=None, user_ids=None):
        out: list[list[str]] = []
        for i in range(len(queries)):
            uid = None
            if user_ids and i < len(user_ids):
                uid = user_ids[i]
            elif batch_context and i < len(batch_context):
                uid = (batch_context[i] or {}).get("user_id")
            uv = self.user_emb.vector(uid) if uid else None
            if uv is None:
                out.append([])
                continue
            if self.normalize:
                uv = l2_normalize(uv)
            sims = self.matrix @ uv
            out.append([canonical_track_id(self.index_to_id[int(j)]) for j in topk_indices(sims, topk)])
        return out


class SameArtistChannel:
    label = "same_artist"

    def __init__(self, catalog: Catalog) -> None:
        self.track_to_artists: dict[str, list[str]] = {}
        self.artist_to_tracks: dict[str, list[str]] = {}
        for t in catalog.index_to_id:
            arts = catalog.metadata(t).get("artist_name") or []
            if isinstance(arts, str):
                arts = [arts]
            self.track_to_artists[t] = arts
            for a in arts:
                self.artist_to_tracks.setdefault(a, []).append(t)

    def batch_text_to_item_retrieval(self, queries, topk, batch_context=None, user_ids=None):
        out: list[list[str]] = []
        for i in range(len(queries)):
            hist = _history(batch_context, i)
            hist_set = set(hist)
            res: list[str] = []
            seen: set[str] = set()
            for h in hist:
                for a in self.track_to_artists.get(h, []):
                    for t in self.artist_to_tracks.get(a, []):
                        if t in seen or t in hist_set:
                            continue
                        seen.add(t)
                        res.append(canonical_track_id(t))
                        if len(res) >= topk:
                            break
                    if len(res) >= topk:
                        break
                if len(res) >= topk:
                    break
            out.append(res)
        return out
