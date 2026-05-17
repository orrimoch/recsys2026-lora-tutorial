"""Post-mortem analysis helpers for the SID generator.

Three diagnostics, all built on pure functions over token-id arrays + lookup dicts:
  Q1 — per-position teacher-forced top-k accuracy (which SID position is wrong?)
  Q2 — generated-SID popularity collapse stats (entropy, gini, top-k coverage)
  Q3 — unconstrained-generation validity (does the model know the SID space?)

Kept separate from `mcrs.sid.eval` because eval is on the hot-path for the W3
gate; diagnostics are post-failure forensics with different aggregation needs.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Iterable


# ---------------------------------------------------------------------------
# Q1 — per-position accuracy
# ---------------------------------------------------------------------------

def topk_accuracy(
    rankings: list[list[int]],
    golds: list[int],
    k: int,
) -> float:
    """Fraction of rows where the gold token id is among the first `k` ranked predictions.

    Args:
        rankings: rankings[i] = list of token ids, most likely first.
        golds: golds[i] = the gold token id for row i.
        k: cutoff.

    Returns:
        float in [0, 1]. Empty input → 0.0.
    """
    if not rankings:
        return 0.0
    hits = sum(1 for r, g in zip(rankings, golds) if g in r[:k])
    return hits / len(rankings)


def build_level_token_ids(
    sid_lookup: dict[tuple[int, int], int],
) -> dict[int, list[int]]:
    """Group SID token ids by level — used to restrict argmax to a single level's vocab.

    Returns {level: sorted list of token ids}.
    """
    by_level: dict[int, list[int]] = {}
    for (level, _code), tok_id in sid_lookup.items():
        by_level.setdefault(level, []).append(tok_id)
    for level in by_level:
        by_level[level].sort()
    return by_level


# ---------------------------------------------------------------------------
# Q2 — popularity collapse statistics
# ---------------------------------------------------------------------------

def shannon_entropy_bits(counter: dict | Counter) -> float:
    """Shannon entropy in bits of the distribution implied by the counter values.

    A uniform 2-outcome distribution returns 1.0; a degenerate 1-outcome distribution
    returns 0.0. Empty counter returns 0.0.
    """
    total = sum(counter.values())
    if total == 0:
        return 0.0
    h = 0.0
    for c in counter.values():
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log2(p)
    return h


def gini(values: list[int] | list[float]) -> float:
    """Gini coefficient over a list of non-negative counts.

    Returns 0.0 for empty input or perfect equality, approaches 1.0 as concentration
    increases. Computed via the standard sorted-cumsum formula.
    """
    n = len(values)
    if n == 0:
        return 0.0
    sorted_vals = sorted(values)
    cum = 0.0
    weighted = 0.0
    for i, v in enumerate(sorted_vals, start=1):
        weighted += i * v
        cum += v
    if cum == 0:
        return 0.0
    return (2 * weighted) / (n * cum) - (n + 1) / n


def popularity_stats(
    counter: Counter,
    top_ks: Iterable[int] = (1, 10, 50, 100),
) -> dict:
    """Distributional summary of a SID frequency counter.

    Returns:
        n_unique          — number of distinct keys in the counter
        total             — sum of counts
        top_1_frequency   — most-frequent key's share of the total
        top_1_key         — that key (e.g. an (c1,c2,c3) tuple)
        top_k_coverage    — {k: cumulative share covered by the top-k keys}
        entropy_bits      — Shannon entropy in bits
        gini              — Gini coefficient
    """
    total = sum(counter.values())
    n_unique = len(counter)
    if total == 0 or n_unique == 0:
        return {
            "n_unique": 0, "total": 0,
            "top_1_frequency": 0.0, "top_1_key": None,
            "top_k_coverage": {k: 0.0 for k in top_ks},
            "entropy_bits": 0.0, "gini": 0.0,
        }
    most_common = counter.most_common()
    top_1_count = most_common[0][1]
    cumulative = 0
    cum_by_k: dict[int, float] = {}
    sorted_ks = sorted(set(top_ks))
    next_k_idx = 0
    for i, (_key, c) in enumerate(most_common, start=1):
        cumulative += c
        while next_k_idx < len(sorted_ks) and i == sorted_ks[next_k_idx]:
            cum_by_k[sorted_ks[next_k_idx]] = cumulative / total
            next_k_idx += 1
    # For any k > n_unique, coverage saturates at 1.0.
    for k in sorted_ks:
        if k not in cum_by_k:
            cum_by_k[k] = 1.0 if k > n_unique else cum_by_k.get(k, 0.0)
    return {
        "n_unique": n_unique,
        "total": total,
        "top_1_frequency": top_1_count / total,
        "top_1_key": most_common[0][0],
        "top_k_coverage": cum_by_k,
        "entropy_bits": shannon_entropy_bits(counter),
        "gini": gini([c for _, c in most_common]),
    }


# ---------------------------------------------------------------------------
# Q3 — unconstrained-generation validity
# ---------------------------------------------------------------------------

def is_triplet_valid(
    token_ids: list[int],
    sid_inverse_lookup: dict[int, tuple[int, int]],
    sid_to_tracks: dict[tuple[int, int, int], list[str]],
) -> dict:
    """Classify a 3-token unconstrained generation.

    Args:
        token_ids: the 3 tokens the model emitted at positions [0, 1, 2].
        sid_inverse_lookup: token_id → (level, code), built from `sid_lookup`.
        sid_to_tracks: codebook — only triplets in this map correspond to real tracks.

    Returns dict with:
        level_in_range[i]   — True iff token at position i is a SID token at level i
        all_levels_in_range — True iff all three positions had the right level
        triplet_in_codebook — True iff (c1, c2, c3) appears in `sid_to_tracks`
    """
    if len(token_ids) != 3:
        raise ValueError(f"expected 3 token ids, got {len(token_ids)}")
    level_in_range = [False, False, False]
    codes = [None, None, None]
    for pos, tid in enumerate(token_ids):
        info = sid_inverse_lookup.get(tid)
        if info is None:
            continue
        level, code = info
        if level == pos:
            level_in_range[pos] = True
            codes[pos] = code
    all_in_range = all(level_in_range)
    triplet_in_codebook = False
    if all_in_range:
        triplet_in_codebook = (codes[0], codes[1], codes[2]) in sid_to_tracks
    return {
        "level_in_range": level_in_range,
        "all_levels_in_range": all_in_range,
        "triplet_in_codebook": triplet_in_codebook,
    }


def aggregate_validity(records: list[dict]) -> dict:
    """Aggregate per-query validity records into rates.

    Each input record matches the dict returned by `is_triplet_valid`.
    """
    n = len(records)
    if n == 0:
        return {
            "n": 0,
            "level_0_in_range_rate": 0.0,
            "level_1_in_range_rate": 0.0,
            "level_2_in_range_rate": 0.0,
            "all_levels_in_range_rate": 0.0,
            "triplet_in_codebook_rate": 0.0,
        }
    l0 = sum(1 for r in records if r["level_in_range"][0])
    l1 = sum(1 for r in records if r["level_in_range"][1])
    l2 = sum(1 for r in records if r["level_in_range"][2])
    all_in = sum(1 for r in records if all(r["level_in_range"]))
    in_cb = sum(1 for r in records if r["triplet_in_codebook"])
    return {
        "n": n,
        "level_0_in_range_rate": l0 / n,
        "level_1_in_range_rate": l1 / n,
        "level_2_in_range_rate": l2 / n,
        "all_levels_in_range_rate": all_in / n,
        "triplet_in_codebook_rate": in_cb / n,
    }
