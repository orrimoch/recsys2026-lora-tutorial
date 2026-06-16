"""F1 — integrity helpers (the join-coverage checks channels/fusion rely on).

The full 6-point gate over all data is run in P0; these are the composable, unit-tested
primitives. Channels canonicalize ids; fusion/integrity assert ⊆ catalog.
"""
from __future__ import annotations

from typing import Iterable

from mcrs.data.catalog import Catalog
from mcrs.data.ids import canonical_track_id


def unresolved_ids(ids: Iterable[str], catalog: Catalog) -> set[str]:
    """Canonical ids that are NOT in the catalog universe (0 is the integrity gate)."""
    return {t for t in (canonical_track_id(x) for x in ids) if t not in catalog}


def embeddings_aligned(catalog: Catalog, track_emb) -> bool:
    """TrackEmbeddings rows must be row-aligned to the catalog id order (matmul correctness)."""
    return catalog.index_to_id == track_emb.index_to_id
