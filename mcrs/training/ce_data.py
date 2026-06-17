"""K3b — GPU-free data/label construction for the cross-encoder fine-tune."""
from __future__ import annotations

import random
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


def sample_negatives(pool, *, gold_tid, gold_title, gold_artist,
                     artist_fn, title_fn, n, k_min, seed,
                     sampling="rank_strat", same_artist="soft_downweight",
                     denoise_near_dup=True, skip_top_rank=False,
                     same_artist_weight=0.25):
    """Pick up to `n` negative track_ids from `pool` (list of (tid, rank), rank-ascending).

    Deterministic given `seed`. Near-dup titles are dropped; same-artist is dropped/down-weighted/
    kept per `same_artist`; rank-1 optionally skipped. Returns [] if the eligible set is empty.
    """
    rng = random.Random(seed)
    cands = sorted(pool, key=lambda x: x[1])                       # stable, rank-ascending
    eligible = []                                                  # (tid, rank, weight)
    for tid, rank in cands:
        if tid == gold_tid:
            continue
        if skip_top_rank and rank == 1:
            continue
        title = title_fn(tid) or ""
        if denoise_near_dup and is_near_dup(gold_title, title):
            continue                                               # true false negative
        cand_artist = artist_fn(tid)
        same = (bool(gold_artist) and bool(cand_artist)
                and gold_artist.strip().lower() == cand_artist.strip().lower())
        if same and same_artist == "drop":
            continue
        w = same_artist_weight if (same and same_artist == "soft_downweight") else 1.0
        eligible.append((tid, rank, w))
    if not eligible:
        return []
    if sampling == "rank_strat":
        # split eligible into top-half and bottom-half by rank; draw ~half from each
        mid = len(eligible) // 2 or 1
        top, bot = eligible[:mid], eligible[mid:]
        chosen = (_weighted_sample(top, (n + 1) // 2, rng)
                  + _weighted_sample(bot, n // 2, rng))
        # backfill if a stratum was short
        if len(chosen) < min(n, len(eligible)):
            remaining = [e for e in eligible if e[0] not in set(chosen)]
            chosen += _weighted_sample(remaining, min(n, len(eligible)) - len(chosen), rng)
    else:
        chosen = _weighted_sample(eligible, n, rng)
    return chosen


def _weighted_sample(items, k, rng):
    """Weighted sampling WITHOUT replacement; returns a list of track_ids. items: (tid, rank, w)."""
    pool = list(items)
    out = []
    for _ in range(min(k, len(pool))):
        total = sum(w for _, _, w in pool)
        if total <= 0:
            break
        r = rng.random() * total
        acc = 0.0
        for idx, (tid, _, w) in enumerate(pool):
            acc += w
            if r <= acc:
                out.append(tid)
                pool.pop(idx)
                break
    return out
