import os
import json
from datasets import load_dataset, concatenate_datasets

# Shared, read-only metadata cache keyed by (dataset_name, split_types). Building
# the {track_id: row} map scans ~47k rows, and several call sites in one run (the
# LGBM feature builder + the same_artist channel) each construct a MusicCatalogDB
# — without sharing they hold independent copies in RAM. Reuse one dict per key.
_SHARED_METADATA: dict[tuple, dict] = {}

class MusicCatalogDB:
    def __init__(self,
            dataset_name: str,
            split_types: list[str],
            corpus_types: list[str],
        ):
        self.corpus_types = corpus_types
        key = (dataset_name, tuple(split_types))
        metadata_dict = _SHARED_METADATA.get(key)
        if metadata_dict is None:
            metadata_dataset = load_dataset(dataset_name)
            metadata_concat_dataset = concatenate_datasets(
                [metadata_dataset[split_type] for split_type in split_types]
            )
            metadata_dict = {item["track_id"]: item for item in metadata_concat_dataset}
            _SHARED_METADATA[key] = metadata_dict
        self.metadata_dict = metadata_dict

    def id_to_metadata(self, track_id: str, use_semantic_id: bool = False):
        # Canonical track-text (single source of truth) — see track_text.py.
        # Keeps the served-history rendering byte-identical to the catalog
        # embedding + the training positives, killing the train/serve format
        # mismatch that capped bge_m3_ft.
        from ..retrieval_modules.track_text import format_catalog_track_text
        return format_catalog_track_text(track_id, self.metadata_dict, self.corpus_types)
