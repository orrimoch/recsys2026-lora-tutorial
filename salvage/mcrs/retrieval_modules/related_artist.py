"""Related-artist recall channel (Lever 3). Reaches NEW artists — the ~96%-
new-artist recall wall that same_artist (session artists only) and sasrec
structurally cannot.

Mechanism: cross-session artist co-occurrence. From the TRAIN split, count how
often artist B appears in a session that also contains artist A -> per-artist
Counter of co-occurring artists. At serve: take the session's seen artists,
expand to their top co-occurring NEW artists (not already in the session), and
emit those artists' most-popular tracks (minus already-played).

Stage 16 probe (2026-05-30): ~29% of union-missed dev golds are reachable at
top-100 co-occurring artists — an order of magnitude over other recall levers.

The co-occurrence map is built once from the train split and cached to disk
(like cf_bpr embeddings), so serve just loads it.
"""
from __future__ import annotations

import os
import pickle
from collections import Counter, defaultdict
from typing import Optional

from .session_history import played_tids_from_context

TRAIN_DATASET = "talkpl-ai/TalkPlayData-Challenge-Dataset"


class RelatedArtistRetriever:
    def __init__(self, dataset_name, split_types, corpus_types, cache_dir="./cache"):
        from mcrs.db_item.music_catalog import MusicCatalogDB
        catalog = MusicCatalogDB(dataset_name, split_types, corpus_types)
        self._build_catalog_index(catalog)
        self.cooc = self._load_or_build_cooc(cache_dir)
        n_cooc = len(self.cooc)
        print(f"[related-artist] ready — tracks={len(self.catalog_tids)} "
              f"artists_with_cooc={n_cooc}")

    # ------------------------------------------------------------------ catalog
    def _build_catalog_index(self, catalog) -> None:
        self.tid_to_artist: dict[str, str] = {}
        artist_tracks: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for tid, m in catalog.metadata_dict.items():
            artist = str(m.get("artist_name") or "").strip().lower()
            if not artist:
                continue
            pop = float(m.get("popularity") or 0.0)
            self.tid_to_artist[tid] = artist
            artist_tracks[artist].append((tid, pop))
        # popularity-sorted track ids per artist (mirrors same_artist.py)
        self.artist_to_tids: dict[str, list[str]] = {
            a: [t for t, _ in sorted(v, key=lambda x: -x[1])]
            for a, v in artist_tracks.items()
        }
        self.catalog_tids = set(self.tid_to_artist.keys())

    # ------------------------------------------------------------------ co-occ
    def _cooc_cache_path(self, cache_dir: str) -> str:
        return os.path.join(cache_dir, "related_artist", "artist_cooc.pkl")

    def _load_or_build_cooc(self, cache_dir: str) -> dict:
        path = self._cooc_cache_path(cache_dir)
        if os.path.isfile(path):
            with open(path, "rb") as f:
                cooc = pickle.load(f)
            print(f"[related-artist] loaded co-occurrence cache: "
                  f"{len(cooc)} artists from {path}")
            return cooc
        cooc = self._build_cooc_from_train()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(cooc, f)
        print(f"[related-artist] built + cached co-occurrence: {len(cooc)} artists")
        return cooc

    def _build_cooc_from_train(self) -> dict:
        """artist -> Counter(co-occurring artist -> #sessions shared) over TRAIN.

        An artist pair co-occurs once per session they both appear in (a session's
        artist set is de-duplicated first, so repeated plays don't inflate counts).
        Identical to the Stage 16 probe that validated this signal."""
        import pandas as pd
        from datasets import load_dataset

        print("[related-artist] building artist co-occurrence from train split...")
        tr = load_dataset(TRAIN_DATASET, split="train")
        cooc: dict[str, Counter] = defaultdict(Counter)
        for sess in tr:
            df = pd.DataFrame(sess["conversations"])
            arts = []
            for tid in df[df["role"] == "music"]["content"]:
                a = self.tid_to_artist.get(str(tid))
                if a:
                    arts.append(a)
            uniq = list(dict.fromkeys(arts))  # de-dup, order-preserving
            for a in uniq:
                for b in uniq:
                    if a != b:
                        cooc[a][b] += 1
        return dict(cooc)

    # --------------------------------------------------------------- retrieve
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
            seen_artists = {self.tid_to_artist[t] for t in played
                            if t in self.tid_to_artist}
            # Expand seen artists -> co-occurring NEW artists, summed by count.
            expanded: Counter = Counter()
            for a in seen_artists:
                for b, c in self.cooc.get(a, {}).items():
                    if b not in seen_artists:
                        expanded[b] += c
            ranked: list[str] = []
            for artist, _cnt in expanded.most_common():
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
