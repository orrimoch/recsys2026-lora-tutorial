"""K3b — GPU-free data/label construction for the cross-encoder fine-tune."""
from __future__ import annotations

import hashlib
import random
import re
from typing import Callable, Optional


def _stable_seed(seed: int, session_id: str, turn_number: int) -> int:
    """Process-stable 32-bit seed for per-turn negative sampling. Python's built-in hash() is
    randomized per interpreter (PYTHONHASHSEED), which would break OOF reproducibility across runs;
    md5 of the joined key is deterministic everywhere."""
    digest = hashlib.md5(f"{seed}:{session_id}:{turn_number}".encode()).hexdigest()
    return int(digest, 16) & 0xFFFFFFFF


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


def false_negative_drop_set(tids, scores: dict, quantile: float) -> set:
    """The set of candidate negatives to DROP as likely unlabeled positives (T1.3).

    A frozen cross-encoder scores each (query, candidate) pair; the highest-scoring negatives are the
    ones the teacher thinks are relevant — i.e. probable unlabeled positives. Training against them
    teaches the model to push apart true matches. Drop the top `quantile` fraction by teacher score.

    Deterministic: ties broken by ascending tid so the cut is stable. `quantile<=0` (default) drops
    nothing — a pure no-op, so this lever is off unless explicitly enabled."""
    if quantile <= 0 or not tids:
        return set()
    ranked = sorted(tids, key=lambda t: (scores.get(t, float("-inf")), t))   # ascending teacher score
    n_drop = int(len(ranked) * quantile)                                     # floor; <1 frac -> 0
    return set(ranked[len(ranked) - n_drop:]) if n_drop > 0 else set()


def sample_negatives(pool, *, gold_tid, gold_title, gold_artist,
                     artist_fn, title_fn, n, k_min, seed,
                     sampling="rank_strat", same_artist="soft_downweight",
                     denoise_near_dup=True, skip_top_rank=False,
                     same_artist_weight=0.25,
                     neg_scores=None, fp_quantile=0.0):
    """Pick up to `n` negative track_ids from `pool` (list of (tid, rank), rank-ascending).

    Deterministic given `seed`. Near-dup titles are dropped; same-artist is dropped/down-weighted/
    kept per `same_artist`; rank-1 optionally skipped. When `neg_scores` (a {tid: teacher_score} map)
    and `fp_quantile>0` are given, the top `fp_quantile` of eligible negatives by teacher score are
    dropped as likely unlabeled positives (T1.3 false-negative denoise). Returns [] if empty.
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
            continue                                               # true false negative (title)
        cand_artist = artist_fn(tid)
        same = (bool(gold_artist) and bool(cand_artist)
                and gold_artist.strip().lower() == cand_artist.strip().lower())
        if same and same_artist == "drop":
            continue
        w = same_artist_weight if (same and same_artist == "soft_downweight") else 1.0
        eligible.append((tid, rank, w))
    # T1.3: drop teacher-scored false negatives (likely unlabeled positives) before sampling.
    if neg_scores and fp_quantile > 0:
        drop = false_negative_drop_set([e[0] for e in eligible], neg_scores, fp_quantile)
        eligible = [e for e in eligible if e[0] not in drop]
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
                             gp_fn=None, w_low=0.3, fusion_query_builder=None,
                             fusion_chunk=0, show_progress=False, report=None,
                             teacher_score_fn=None, fp_quantile=0.0):
    """Build [(ce_query_text, [pos_doc, neg_doc...], group_weight)] for gold-in-pool turns.

    `query_builder` builds the CROSS-ENCODER pair query (e.g. the enriched/markered query).
    `fusion_query_builder` (defaults to `query_builder`) builds the RETRIEVAL query used to fuse the
    candidate pool — at serve, retrieval uses the plain query while K3 re-scores with the enriched one,
    so pass the plain builder here to make the negative pool match serve (spec train==serve).
    `gp_fn(turn) -> goal_progress_label | None` supplies the per-group weight (None => uniform). Drops
    turns whose gold is not in the top-`cross_encoder_k` pool, or that have < k_min negatives. When a
    `report` dict is given it gets `dropped_no_gold`, `dropped_few_neg`, `kept`, and `kept_keys`
    (the (session_id, turn_number) of each kept group, in order — use this to align folds to groups).
    """
    gp_weights = dict(GP_WEIGHTS); gp_weights["DOES_NOT_MOVE_TOWARD_GOAL"] = w_low
    fusion_qb = fusion_query_builder or query_builder
    ce_queries = [query_builder.build(t).text for t in turns]          # cross-encoder pair query
    fusion_queries = [fusion_qb.build(t).text for t in turns]          # pool-retrieval query (matches serve)
    bc = [{"history_tids": t.history_tids, "user_id": t.user_id} for t in turns]
    uids = [t.user_id for t in turns]
    # Fusion is the slow step (full retrieval over the catalog per turn). With `fusion_chunk` we fuse
    # in chunks of turns — gives a tqdm progress bar (show_progress) AND bounds peak memory (avoids
    # one giant similarity matrix). Default (0) keeps the single batched call (unchanged behavior).
    if fusion_chunk and fusion_chunk > 0:
        steps = range(0, len(turns), fusion_chunk)
        if show_progress:
            try:
                from tqdm.auto import tqdm
                steps = tqdm(list(steps), desc="fusion (turns)", unit="chunk")
            except Exception:
                pass
        pools = []
        for s in steps:
            e = s + fusion_chunk
            pools.extend(fusion.fuse(fusion_queries[s:e], cross_encoder_k,
                                     topk_internal=cross_encoder_k,
                                     batch_context=bc[s:e], user_ids=uids[s:e]))
    else:
        pools = fusion.fuse(fusion_queries, cross_encoder_k, topk_internal=cross_encoder_k,
                            batch_context=bc, user_ids=uids)
    def _field(tid, key):                                   # real catalog fields can be LISTS -> coerce to str
        if tid not in catalog._meta:
            return None
        v = catalog.metadata(tid).get(key)
        if isinstance(v, list):
            return ", ".join(str(x) for x in v)
        return None if v is None else str(v)
    artist_fn = lambda tid: _field(tid, "artist_name")
    title_fn = lambda tid: _field(tid, "track_name")
    groups, kept_keys, dropped_no_gold, dropped_few_neg = [], [], 0, 0
    for turn, qtext, pool in zip(turns, ce_queries, pools):
        gold = gold_fn(turn)
        top = pool[:cross_encoder_k]
        ids = [c.track_id for c in top]
        if gold is None or gold not in ids:
            dropped_no_gold += 1
            continue
        ranked = [(c.track_id, i + 1) for i, c in enumerate(top)]      # fused-pool rank (RRF-sorted position)
        # T1.3: if a frozen-CE teacher is supplied, score the pool once so sample_negatives can drop
        # the top-`fp_quantile` likely-unlabeled-positives. Off by default (teacher_score_fn=None).
        neg_scores = None
        if teacher_score_fn is not None and fp_quantile > 0:
            neg_scores = dict(zip(ids, teacher_score_fn(qtext, ids)))
        negs = sample_negatives(ranked, gold_tid=gold, gold_title=(title_fn(gold) or ""),
                                gold_artist=(artist_fn(gold) or ""), artist_fn=artist_fn,
                                title_fn=title_fn, n=n_negatives, k_min=k_min,
                                seed=_stable_seed(seed, turn.session_id, turn.turn_number),
                                sampling=sampling, same_artist=same_artist,
                                denoise_near_dup=denoise_near_dup, skip_top_rank=skip_top_rank,
                                neg_scores=neg_scores, fp_quantile=fp_quantile)
        if len(negs) < k_min:
            dropped_few_neg += 1
            continue
        docs = [build_doc(catalog, gold)] + [build_doc(catalog, t) for t in negs]
        gw = gp_weights.get(gp_fn(turn)) if gp_fn else 1.0
        groups.append((qtext, docs, gw))
        kept_keys.append((turn.session_id, turn.turn_number))
    if report is not None:
        report.update(dropped_no_gold=dropped_no_gold, dropped_few_neg=dropped_few_neg,
                      kept=len(groups), kept_keys=kept_keys)
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


def normalize_within_pool(tid_to_score: dict) -> dict:
    """Min-max normalize scores within one turn's candidate pool (fold-scale-invariant feature)."""
    if not tid_to_score:
        return {}
    vals = list(tid_to_score.values())
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    return {tid: (s - lo) / rng for tid, s in tid_to_score.items()}
