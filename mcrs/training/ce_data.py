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


GP_WEIGHTS = {"MOVES_TOWARD_GOAL": 1.0, "DOES_NOT_MOVE_TOWARD_GOAL": 0.3, None: 1.0}


def build_ce_training_groups(query_builder, fusion, turns, gold_fn, *, catalog, cross_encoder_k,
                             n_negatives=15, sampling="rank_strat", same_artist="soft_downweight",
                             denoise_near_dup=True, skip_top_rank=False, k_min=4, seed,
                             gp_fn=None, w_low=0.3, report=None):
    """Build [(query_text, [pos_doc, neg_doc...], group_weight)] for gold-in-pool turns.

    `gp_fn(turn) -> goal_progress_label | None` supplies the per-group weight (None => uniform).
    Drops turns whose gold is not in the top-`cross_encoder_k` pool, or that have < k_min negatives.
    """
    gp_weights = dict(GP_WEIGHTS); gp_weights["DOES_NOT_MOVE_TOWARD_GOAL"] = w_low
    queries = [query_builder.build(t).text for t in turns]
    bc = [{"history_tids": t.history_tids, "user_id": t.user_id} for t in turns]
    uids = [t.user_id for t in turns]
    pools = fusion.fuse(queries, cross_encoder_k, topk_internal=cross_encoder_k,
                        batch_context=bc, user_ids=uids)
    artist_fn = lambda tid: catalog.metadata(tid).get("artist_name") if tid in catalog._meta else None
    title_fn = lambda tid: catalog.metadata(tid).get("track_name") if tid in catalog._meta else None
    groups, dropped_no_gold, dropped_few_neg = [], 0, 0
    for turn, qtext, pool in zip(turns, queries, pools):
        gold = gold_fn(turn)
        top = pool[:cross_encoder_k]
        ids = [c.track_id for c in top]
        if gold is None or gold not in ids:
            dropped_no_gold += 1
            continue
        ranked = [(c.track_id, min(c.channel_ranks.values()) if c.channel_ranks else (i + 1))
                  for i, c in enumerate(top)]
        negs = sample_negatives(ranked, gold_tid=gold, gold_title=(title_fn(gold) or ""),
                                gold_artist=(artist_fn(gold) or ""), artist_fn=artist_fn,
                                title_fn=title_fn, n=n_negatives, k_min=k_min,
                                seed=hash((seed, turn.session_id, turn.turn_number)) & 0xFFFFFFFF,
                                sampling=sampling, same_artist=same_artist,
                                denoise_near_dup=denoise_near_dup, skip_top_rank=skip_top_rank)
        if len(negs) < k_min:
            dropped_few_neg += 1
            continue
        docs = [build_doc(catalog, gold)] + [build_doc(catalog, t) for t in negs]
        gw = gp_weights.get(gp_fn(turn)) if gp_fn else 1.0
        groups.append((qtext, docs, gw))
    if report is not None:
        report.update(dropped_no_gold=dropped_no_gold, dropped_few_neg=dropped_few_neg, kept=len(groups))
    return groups


def assign_session_folds(session_ids, *, k, seed):
    """Deterministic session-disjoint fold id per row. All rows of a session share a fold."""
    uniq = sorted(set(session_ids))
    rng = random.Random(seed)
    rng.shuffle(uniq)
    fold_of = {s: i % k for i, s in enumerate(uniq)}
    return [fold_of[s] for s in session_ids]


def drop_cross_fold_near_dups(items):
    """items: list[(dedup_key, fold)]. Keep the first occurrence of each key; drop later folds'
    copies so a near-duplicate (query->gold) never straddles the fold boundary. Returns kept indices."""
    seen, keep = set(), []
    for i, (key, _fold) in enumerate(items):
        if key in seen:
            continue
        seen.add(key)
        keep.append(i)
    return keep
