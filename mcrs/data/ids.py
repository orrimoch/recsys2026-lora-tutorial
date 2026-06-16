"""F1 — the ONE canonical track_id normalizer (plan §7.3 req #1).

Pure normalization only (strip whitespace + a `track_id:` prefix, NFC). Catalog membership
(`tid in catalog`) and the fusion-time ⊆-catalog assert live in Catalog / the integrity gate —
not here — so this stays a pure, dependency-free helper every channel can import.
"""
from __future__ import annotations

import unicodedata

_PREFIX = "track_id:"


def canonical_track_id(raw: str) -> str:
    s = str(raw).strip()
    if s.startswith(_PREFIX):
        s = s[len(_PREFIX):].strip()
    return unicodedata.normalize("NFC", s)


def canonical_track_ids(raws: list[str]) -> list[str]:
    return [canonical_track_id(r) for r in raws]
