"""Canonical track-text formatter — the SINGLE source of truth for how a catalog
track is rendered as text.

Train/serve parity is this repo's recurring failure mode: a fine-tuned encoder
learns to match positives in one track-text format but is served a catalog
embedded in another, silently destroying the learned alignment (this capped
bge_m3_ft). Every place that renders a track as text — the served catalog
embedding (embed_catalog.py), the [HISTORY] music-turn expansion in training
(build_bi_encoder_training_data._format_history_music_turn), and
MusicCatalogDB.id_to_metadata at serve — MUST go through this one function.

Format (the established id_to_metadata format):
    'track_id: <id>, <ct1>: <vals>, <ct2>: <vals>, ...'
where each <vals> is the corpus field's value list joined by ', ' and lowercased.
Graceful on a missing track (returns the bare id) and on missing/non-list fields.
"""
from __future__ import annotations


def format_catalog_track_text(track_id: str, metadata_dict: dict,
                              corpus_types: list) -> str:
    """Render one track's metadata as the canonical catalog/positive text.

    Byte-identical to MusicCatalogDB.id_to_metadata for valid, list-valued
    tracks; safe (no raise) on missing tracks or missing/non-list fields.
    """
    if track_id not in metadata_dict:
        return track_id
    md = metadata_dict[track_id]
    parts = [f"track_id: {track_id}"]
    for ct in corpus_types:
        val = md.get(ct)
        if val is None:
            joined = ""
        elif isinstance(val, list):
            joined = ", ".join(str(v) for v in val)
        else:
            joined = str(val)
        parts.append(f"{ct}: {joined.lower()}")
    return ", ".join(parts)
