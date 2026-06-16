"""F1 — precomputed track & user embeddings.

TrackEmbeddings: one float32 matrix per modality, row-ordered by index_to_id (shared with
Catalog when both load all_tracks in the same order). Lazy per modality (the track-emb arrow
is large). UserEmbeddings: cf-bpr per user; vector()==None => cold/missing signal.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from mcrs.data.ids import canonical_track_id


class TrackEmbeddings:
    def __init__(self, rows: Iterable[dict], modalities: Optional[list[str]] = None) -> None:
        rows = list(rows)
        if modalities is None:
            keys: set[str] = set()
            for r in rows:
                keys.update(k for k in r if k != "track_id")
            modalities = sorted(keys)
        self.modalities = list(modalities)
        self.index_to_id: list[str] = []
        self._vecs: dict[str, dict[str, list]] = {m: {} for m in self.modalities}
        seen: set[str] = set()
        for r in rows:
            tid = canonical_track_id(r["track_id"])
            if tid not in seen:
                seen.add(tid)
                self.index_to_id.append(tid)
            for m in self.modalities:
                v = r.get(m)
                if v is not None:
                    self._vecs[m][tid] = v
        self.id_to_index: dict[str, int] = {t: i for i, t in enumerate(self.index_to_id)}
        self._matrix_cache: dict[str, np.ndarray] = {}

    def vector(self, track_id: str, modality: str) -> np.ndarray:
        return np.asarray(self._vecs[modality][track_id], dtype=np.float32)

    def matrix(self, modality: str) -> np.ndarray:
        """Dense (n_tracks, dim) float32 matrix in index_to_id order. Tracks whose
        vector is missing/empty (e.g. cold tracks with no cf-bpr) are zero-filled to dim."""
        if modality not in self._matrix_cache:
            vecs = self._vecs[modality]
            dim = next((len(v) for v in vecs.values() if v), 0)
            zero = [0.0] * dim
            rows = [(v if (v and len(v) == dim) else zero) for v in
                    (vecs.get(t) for t in self.index_to_id)]
            self._matrix_cache[modality] = np.asarray(rows, dtype=np.float32)
        return self._matrix_cache[modality]

    def present_ids(self, modality: str) -> set[str]:
        """Track ids with a non-empty vector for this modality (for integrity reporting)."""
        return {t for t, v in self._vecs[modality].items() if v}

    @classmethod
    def from_disk(cls, path: str, split: str = "all_tracks",
                  modalities: Optional[list[str]] = None) -> "TrackEmbeddings":
        from datasets import load_from_disk

        ds = load_from_disk(path)
        d = ds[split] if hasattr(ds, "keys") else ds
        if modalities:  # only materialize the requested columns (the arrow is large)
            d = d.select_columns(["track_id", *modalities])
        return cls(d, modalities=modalities)


class UserEmbeddings:
    def __init__(self, rows: Iterable[dict], key: str = "cf-bpr") -> None:
        self.key = key
        self._vecs: dict[str, list] = {r["user_id"]: r[key] for r in rows}

    def vector(self, user_id: str) -> Optional[np.ndarray]:
        v = self._vecs.get(user_id)
        return None if v is None else np.asarray(v, dtype=np.float32)

    def __contains__(self, user_id: str) -> bool:
        return user_id in self._vecs

    @classmethod
    def from_disk(cls, path: str, splits: Optional[list[str]] = None,
                  key: str = "cf-bpr") -> "UserEmbeddings":
        from datasets import load_from_disk

        ds = load_from_disk(path)
        rows: list[dict] = []
        if hasattr(ds, "keys"):
            for sp in (splits or list(ds.keys())):
                rows.extend(ds[sp])
        else:
            rows = list(ds)
        return cls(rows, key=key)
