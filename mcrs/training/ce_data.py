"""K3b — GPU-free data/label construction for the cross-encoder fine-tune."""
from __future__ import annotations

import re
from typing import Callable, Optional


def build_doc(catalog, track_id: str, *, max_doc_chars: int = 2000) -> str:
    """The SINGLE doc string for train positives, train negatives, AND serve.

    Enriched doc, char-capped. Per-pair token truncation is applied downstream by the shared
    score_fn (cross_encoder.py), so train==serve. Hard-fails if the track has no enriched doc —
    no silent raw fallback during fine-tuning (spec §2/§7).
    """
    if not catalog.is_enriched(track_id):
        raise KeyError(f"build_doc: track {track_id!r} has no enriched doc (100% coverage required)")
    return catalog.id_to_metadata(track_id, enriched=True)[:max_doc_chars]


def normalize_title(t: str) -> str:
    """Lowercase, strip parenthetical qualifiers and punctuation, collapse whitespace."""
    t = (t or "").lower()
    t = re.sub(r"\([^)]*\)", " ", t)                 # drop "(remastered)", "(live)", ...
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return " ".join(t.split())


def is_near_dup(title_a: str, title_b: str) -> bool:
    na, nb = normalize_title(title_a), normalize_title(title_b)
    return bool(na) and na == nb


def is_same_artist(tid_a: str, tid_b: str, artist_fn: Callable[[str], Optional[str]]) -> bool:
    a, b = artist_fn(tid_a), artist_fn(tid_b)
    return bool(a) and bool(b) and a.strip().lower() == b.strip().lower()
