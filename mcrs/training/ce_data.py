"""K3b — GPU-free data/label construction for the cross-encoder fine-tune."""
from __future__ import annotations


def build_doc(catalog, track_id: str, *, max_doc_chars: int = 2000) -> str:
    """The SINGLE doc string for train positives, train negatives, AND serve.

    Enriched doc, char-capped. Per-pair token truncation is applied downstream by the shared
    score_fn (cross_encoder.py), so train==serve. Hard-fails if the track has no enriched doc —
    no silent raw fallback during fine-tuning (spec §2/§7).
    """
    if not catalog.is_enriched(track_id):
        raise KeyError(f"build_doc: track {track_id!r} has no enriched doc (100% coverage required)")
    return catalog.id_to_metadata(track_id, enriched=True)[:max_doc_chars]
