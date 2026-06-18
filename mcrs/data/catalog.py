"""F1 — Catalog: the all_tracks universe, canonical id set, metadata, doc text.

Accepts in-memory rows (hermetic tests) or loads from disk via `from_disk`. Doc text is the
single source for BM25/dense (A1 layers enrichment on top via `enriched=True`, graceful raw
fallback). See `.claude/documents/features/10_F1_data_access.md`.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional

from mcrs.data.ids import canonical_track_id

_DEFAULT_CORPUS = ["track_name", "artist_name", "album_name", "tag_list", "release_date"]


class Catalog:
    def __init__(
        self,
        rows: Iterable[dict],
        corpus_types: Optional[list[str]] = None,
        enriched_docs: Optional[dict[str, str]] = None,
    ) -> None:
        self.corpus_types = list(corpus_types or _DEFAULT_CORPUS)
        # Canonicalize keys so id_to_metadata(canonical_tid, enriched=True) matches even when the
        # enriched-doc source keys carry a 'track_id:' prefix / whitespace (ML-review finding #3:
        # a mismatch silently falls back to the raw doc, defeating A1 enrichment + expansion-first).
        self._enriched = {canonical_track_id(k): v for k, v in (enriched_docs or {}).items()}
        self._meta: dict[str, dict] = {}
        self.index_to_id: list[str] = []
        for r in rows:
            tid = canonical_track_id(r["track_id"])
            if tid not in self._meta:
                self.index_to_id.append(tid)
            self._meta[tid] = r
        self.id_to_index: dict[str, int] = {t: i for i, t in enumerate(self.index_to_id)}
        self.track_ids: frozenset[str] = frozenset(self.index_to_id)

    def __contains__(self, track_id: str) -> bool:
        return track_id in self.track_ids

    def __len__(self) -> int:
        return len(self.index_to_id)

    def metadata(self, track_id: str) -> dict:
        return self._meta[track_id]

    _FIELD_LABEL = {"tag_list": "tags"}

    def id_to_metadata(self, track_id: str, enriched: bool = False) -> str:
        if enriched and track_id in self._enriched:
            return self._enriched[track_id]
        return self._raw_doc(track_id)

    def is_enriched(self, track_id: str) -> bool:
        return track_id in self._enriched

    def _raw_doc(self, track_id: str) -> str:
        row = self._meta[track_id]
        parts = []
        for field in self.corpus_types:
            v: Any = row.get(field)
            if isinstance(v, list):
                v = ", ".join(str(x) for x in v)
            elif v is None:
                v = ""
            else:
                v = str(v)
            parts.append(f"{self._FIELD_LABEL.get(field, field)}: {v}")
        return ", ".join(parts)

    @classmethod
    def from_disk(
        cls,
        path: str,
        split: str = "all_tracks",
        corpus_types: Optional[list[str]] = None,
        enriched_docs: Optional[dict[str, str]] = None,
    ) -> "Catalog":
        from datasets import load_from_disk

        ds = load_from_disk(path)
        rows = ds[split] if hasattr(ds, "keys") else ds
        return cls(rows, corpus_types=corpus_types, enriched_docs=enriched_docs)
