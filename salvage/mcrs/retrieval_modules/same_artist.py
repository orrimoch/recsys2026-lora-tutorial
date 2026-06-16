"""Session artist-continuity recall channel. 38% of gold next-tracks are by an
artist already in the session. Candidates = catalog tracks by the session's
artists, minus already-played, ranked by (artist session-count, popularity)."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Optional

from .session_history import played_tids_from_context


class SameArtistRetriever:
    def __init__(self, dataset_name, split_types, corpus_types, cache_dir="./cache"):
        from mcrs.db_item.music_catalog import MusicCatalogDB
        catalog = MusicCatalogDB(dataset_name, split_types, corpus_types)
        self._build_index(catalog)

    def _build_index(self, catalog) -> None:
        self.tid_to_artist: dict[str, str] = {}
        artist_tracks: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for tid, m in catalog.metadata_dict.items():
            artist = str(m.get("artist_name") or "").strip().lower()
            if not artist:
                continue
            pop = float(m.get("popularity") or 0.0)
            self.tid_to_artist[tid] = artist
            artist_tracks[artist].append((tid, pop))
        self.artist_to_tids: dict[str, list[str]] = {
            a: [t for t, _ in sorted(v, key=lambda x: -x[1])]
            for a, v in artist_tracks.items()
        }
        self.catalog_tids = set(self.tid_to_artist.keys())

    def batch_text_to_item_retrieval(self, queries, topk, user_ids=None,
                                     batch_context=None):
        out = []
        for i in range(len(queries)):
            ctx = batch_context[i] if batch_context else None
            played = played_tids_from_context(ctx, self.catalog_tids)
            if not played:
                out.append([])
                continue
            played_set = set(played)
            artist_count = Counter(self.tid_to_artist[t] for t in played
                                   if t in self.tid_to_artist)
            ranked: list[str] = []
            for artist, _cnt in artist_count.most_common():
                for tid in self.artist_to_tids.get(artist, []):
                    if tid not in played_set:
                        ranked.append(tid)
                        if len(ranked) >= topk:
                            break
                if len(ranked) >= topk:
                    break
            out.append(ranked[:topk])
        return out

    def text_to_item_retrieval(self, query, topk, user_id=None):
        return self.batch_text_to_item_retrieval([query], topk=topk)[0]
