"""Stage 1 — GPU-free data construction for the ColBERT fine-tune.

Ports `scripts/build_colbert_train_data.py` (recall-union-lgbm branch) onto the fresh-start
contracts. The feedback signal is the ORIGINAL way: PyLate `losses.Contrastive` over
(query, positive, negatives) triples, with a turn-1 dev-recall callback selecting the checkpoint
(see `colbert_finetune`, stage 2). So the JSONL row shape (query/positive/negatives + *_tids) is
preserved verbatim.

Pure functions only here (no I/O, no model) — unit-tested. The notebook/CLI wiring (load HF, build
the fusion pool, write JSONL) lives in `build_colbert_train_data` at the bottom and is exercised on
Colab, mirroring how `ce_data.build_ce_training_groups` is driven.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Optional, Sequence

from mcrs.training.ce_data import false_negative_drop_set, is_near_dup, is_same_artist

LABEL_POS = "MOVES_TOWARD_GOAL"


def goal_progress_label(session_row: dict, turn_number: int) -> Optional[str]:
    """Goal-progress label for `turn_number` from the raw HF row, or None if unlabeled.

    Lives on the raw row (`goal_progress_assessments`), NOT on TurnContext — TalkPlay's turn-1
    has no assessment, and the train/test distribution shifts (memory: data facts)."""
    for x in (session_row.get("goal_progress_assessments") or []):
        if int(x.get("turn_number")) == int(turn_number):
            return x.get("goal_progress_assessment")
    return None


def select_hard_negatives(pool: Sequence[str], gold_tid: str, k: int) -> Optional[list[str]]:
    """Top-k non-gold pool items (pool order = hardest first). Returns None when the gold is not
    in the pool — a recall miss the ranker can't recover, so the row is skipped (plan §6)."""
    if gold_tid not in pool:
        return None
    return [t for t in pool if t != gold_tid][:k]


def build_colbert_triple(query: str, pos_text: str, neg_texts: list[str], pos_tid: str,
                         neg_tids: list[str], session_id: Optional[str],
                         turn_number: int, *, pos_score: Optional[float] = None,
                         neg_scores: Optional[list[float]] = None) -> dict[str, Any]:
    """One JSONL training row. `negatives` are doc texts (what ColBERT trains on); `*_tids` are
    kept for diagnostics / val nDCG. When a distillation teacher is used, `pos_score`/`neg_scores`
    carry the teacher's relevance for the positive and each negative (soft labels for KD); they are
    omitted entirely for the plain contrastive path so those JSONL rows are byte-identical."""
    row = {
        "query": query,
        "positive": pos_text,
        "negatives": list(neg_texts),
        "pos_tid": pos_tid,
        "neg_tids": list(neg_tids),
        "session_id": session_id,
        "turn_number": turn_number,
    }
    if pos_score is not None:
        row["pos_score"] = float(pos_score)
    if neg_scores is not None:
        row["neg_scores"] = [float(s) for s in neg_scores]
    return row


def build_triples_from_pools(
    positive_rows: Sequence[dict],
    pools: Sequence[Sequence[str]],
    doc_text_fn: Callable[[str], str],
    k_negs: int,
    min_negs: int = 1,
    *,
    title_fn: Optional[Callable[[str], Optional[str]]] = None,
    artist_fn: Optional[Callable[[str], Optional[str]]] = None,
    denoise_near_dup: bool = True,
    drop_same_artist: bool = False,
    teacher_score_fn: Optional[Callable[[str, list[str]], Sequence[float]]] = None,
    fp_quantile: float = 0.0,
) -> tuple[list[dict], dict]:
    """Assemble triples from positive rows + their aligned candidate pools.

    `pools[i]` is the hardest-first candidate tids for `positive_rows[i]` (e.g. an RRF fusion pool).
    For each row: take the gold + top-`k_negs` non-gold pool items, render to doc text, emit a
    `build_colbert_triple`. Drops the row when the gold is not in the pool (unrecoverable recall miss)
    or it yields < `min_negs` negatives.

    `title_fn`/`artist_fn(tid) -> str|None`: when given, denoise the hard negatives the SAME cheap way
    the CE path does (ML-review T2 #2) BEFORE the top-`k_negs` cut — drop any candidate whose
    normalized title equals the gold's (a different release of the same song = a true false negative the
    contrastive loss would wrongly push away). Same-artist tracks are legitimate hard negatives and are
    KEPT by default; pass `drop_same_artist=True` to also remove them. `denoise_near_dup=False` disables
    the title filter even when `title_fn` is given.

    `teacher_score_fn(query, tids) -> scores` (a frozen/fine-tuned cross-encoder, T2.1/T2.4): when given,
    scores [gold] + candidate negatives so the triple carries teacher soft labels (`pos_score`/
    `neg_scores`) for KD, AND — when `fp_quantile>0` — drops the top fraction of negatives the teacher
    scores as relevant (likely unlabeled positives, T2.4) BEFORE taking the top-`k_negs`.
    Returns (triples, stats) where stats = {kept, dropped_no_gold, dropped_no_neg, dropped_fp, dropped_dup}."""
    triples: list[dict] = []
    dropped_no_gold = dropped_no_neg = dropped_fp = dropped_dup = 0
    for row, pool in zip(positive_rows, pools):
        gold = row["gold_tid"]
        if gold not in pool:
            dropped_no_gold += 1
            continue
        cand = [t for t in pool if t != gold]              # hardest-first non-gold candidates
        # Cheap false-negative denoise (mirrors ce_data.sample_negatives) before any cut/teacher pass.
        if title_fn is not None or (artist_fn is not None and drop_same_artist):
            gold_title = title_fn(gold) if title_fn is not None else None
            kept = []
            for t in cand:
                if denoise_near_dup and title_fn is not None and is_near_dup(gold_title or "", title_fn(t) or ""):
                    dropped_dup += 1
                    continue
                if drop_same_artist and artist_fn is not None and is_same_artist(gold, t, artist_fn):
                    dropped_dup += 1
                    continue
                kept.append(t)
            cand = kept
        pos_score = neg_scores = None
        if teacher_score_fn is not None:
            # The teacher scores relevance with its OWN query (`teacher_query` = the full query the
            # cross-encoder serves with), NOT the student's focused ColBERT query — else the soft
            # labels come from a query distribution the CE never saw. Falls back to row["query"].
            tq = row.get("teacher_query") or row["query"]
            tids = [gold] + cand
            score_of = dict(zip(tids, teacher_score_fn(tq, tids)))
            if fp_quantile > 0:                             # T2.4: drop teacher-flagged false negatives
                drop = false_negative_drop_set(cand, score_of, fp_quantile)
                dropped_fp += len(drop)
                cand = [t for t in cand if t not in drop]
            negs = cand[:k_negs]
            pos_score = score_of[gold]
            neg_scores = [score_of[t] for t in negs]
        else:
            negs = cand[:k_negs]
        if len(negs) < min_negs:
            dropped_no_neg += 1
            continue
        triples.append(build_colbert_triple(
            query=row["query"], pos_text=doc_text_fn(gold), neg_texts=[doc_text_fn(t) for t in negs],
            pos_tid=gold, neg_tids=negs, session_id=row.get("session_id"),
            turn_number=row["turn_number"], pos_score=pos_score, neg_scores=neg_scores))
    stats = {"kept": len(triples), "dropped_no_gold": dropped_no_gold,
             "dropped_no_neg": dropped_no_neg, "dropped_fp": dropped_fp, "dropped_dup": dropped_dup}
    return triples, stats


def dev_eval_pack_from_pools(
    rows: Sequence[dict],
    pools: Sequence[Sequence[str]],
    doc_text_fn: Callable[[str], str],
    *,
    wall_value: Optional[bool] = None,
) -> dict:
    """Assemble the dev-eval pack the fine-tune selection callback consumes, from dev rows + their
    aligned candidate pools. Each row has {query, gold_tid, turn_number}. Drops rows with no gold
    (can't score recall). `wall[i]` marks the rows the checkpoint is SELECTED on (the subset
    `wall_recall_at_k` scores); `tid_to_text` covers the union of all pool tids so the live model can
    re-encode + rerank them.

    `wall_value`: when None (legacy), `wall[i] = (turn_number == 1)` — selects on the turn-1 gate.
    Pass `wall_value=True` when every row IS already a served (final-turn) row, so the whole pack is
    on the selection wall regardless of each row's turn_number (BUG #1 fix: select on what we serve)."""
    queries, golds, out_pools, wall = [], [], [], []
    tid_to_text: dict[str, str] = {}
    for row, pool in zip(rows, pools):
        gold = row["gold_tid"]
        if gold is None:
            continue
        queries.append(row["query"])
        golds.append(gold)
        out_pools.append(list(pool))
        wall.append(bool(wall_value) if wall_value is not None else int(row["turn_number"]) == 1)
        for t in pool:
            if t not in tid_to_text:
                tid_to_text[t] = doc_text_fn(t)
    return {"queries": queries, "golds": golds, "pools": out_pools,
            "wall": wall, "tid_to_text": tid_to_text}


def iter_colbert_positives(
    turns: Sequence[Any],
    gold_fn: Callable[[Any], Optional[str]],
    query_builder: Any,
    gp_fn: Callable[[Any], Optional[str]],
    label_pos: str = LABEL_POS,
    teacher_query_builder: Any = None,
) -> list[dict[str, Any]]:
    """One positive-training row per qualifying TurnContext: {query, gold_tid, turn_number,
    session_id, user_id, history_tids}.

    `query_builder.build(ctx).text` is the per-channel ColBERT query — pass the SAME builder used at
    serve so train query == serve query (no skew). `gold_fn(ctx)` resolves the turn's gold tid;
    `gp_fn(ctx)` the goal-progress label.

    TURN-1 IS ALWAYS KEPT regardless of label (real turn-1 has no assessment, yet the gate is 100%
    turn-1 — RCA #1). Turns >1 require `label_pos` (strips noisy off-goal mid-conversation targets).
    Turns with no resolvable gold are skipped (no positive => nothing to train on)."""
    rows: list[dict[str, Any]] = []
    for ctx in turns:
        tn = int(ctx.turn_number)
        if tn != 1 and gp_fn(ctx) != label_pos:
            continue
        gold = gold_fn(ctx)
        if gold is None:
            continue
        row = {
            "query": query_builder.build(ctx).text,
            "gold_tid": gold,
            "turn_number": tn,
            "session_id": ctx.session_id,
            "user_id": ctx.user_id,
            "history_tids": list(ctx.history_tids),
            "segment": ctx.segment,
        }
        if teacher_query_builder is not None:           # full-query the teacher CE serves with (H2)
            row["teacher_query"] = teacher_query_builder.build(ctx).text
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# I/O orchestration (Colab-driven; needs Conversations + a fusion pool + a catalog).
# Exercised on GPU like ce_data's build_ce_training_groups — not unit-tested.
# --------------------------------------------------------------------------- #
def goal_progress_lookup(raw_rows: Sequence[dict]) -> dict[tuple[str, int], Optional[str]]:
    """Map (session_id, turn_number) -> goal-progress label from the raw HF rows. The label is not
    on TurnContext, so we index it once here and resolve per turn in `build_colbert_train_data`."""
    gp: dict[tuple[str, int], Optional[str]] = {}
    for row in raw_rows:
        sid = row.get("session_id") or row.get("id")
        for x in (row.get("goal_progress_assessments") or []):
            gp[(sid, int(x["turn_number"]))] = x.get("goal_progress_assessment")
    return gp


def write_jsonl(rows: Sequence[dict], path: str) -> int:
    """Write one JSON object per line; returns the count written."""
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return len(rows)


def read_jsonl(path: str) -> list[dict]:
    """Read a JSONL file back into a list of dicts (for the notebook's triple cache)."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_colbert_train_data(
    conversations: Any,
    raw_rows: Sequence[dict],
    query_builder: Any,
    fusion: Any,
    doc_text_fn: Callable[[str], str],
    *,
    pool_size: int = 100,
    k_negs: int = 15,
    min_negs: int = 1,
    show_progress: bool = False,
    report: Optional[dict] = None,
    title_fn: Optional[Callable[[str], Optional[str]]] = None,
    artist_fn: Optional[Callable[[str], Optional[str]]] = None,
    drop_same_artist: bool = False,
    teacher_score_fn: Optional[Callable[[str, list[str]], Sequence[float]]] = None,
    fp_quantile: float = 0.0,
    teacher_query_builder: Any = None,
    fusion_chunk: int = 2000,
) -> list[dict]:
    """End-to-end (TRAIN SPLIT ONLY): TurnContext stream -> contrastive triples.

    `conversations` is a fresh-start `Conversations` for the train split (provides `.turns()` and
    `.gold()`); `raw_rows` are the matching raw HF rows (for goal-progress). `query_builder` MUST be
    the SAME builder ColBERT serves with (train==serve). `fusion` is the RRF pool used to mine hard
    negatives; `doc_text_fn` renders a tid to its ColBERT doc (e.g. `colbert_doc_text(cat, t,
    expansion_first=True)`). Returns the triples (also fills `report` with the drop stats)."""
    gp = goal_progress_lookup(raw_rows)
    gold_fn = lambda ctx: conversations.gold(ctx.session_id, ctx.turn_number)
    gp_fn = lambda ctx: gp.get((ctx.session_id, ctx.turn_number))

    turns = list(conversations.turns())
    positives = iter_colbert_positives(turns, gold_fn, query_builder, gp_fn,
                                       teacher_query_builder=teacher_query_builder)

    queries = [r["query"] for r in positives]
    # segment plumbed through so the hard-neg pool matches serve once segment-weighted RRF is on (#4)
    bc = [{"history_tids": r["history_tids"], "user_id": r["user_id"], "segment": r["segment"]}
          for r in positives]
    uids = [r["user_id"] for r in positives]
    # Chunk the fusion so the progress bar advances per-chunk (retrieval is per-query independent, so
    # chunked == one batched call) AND peak memory is bounded. fusion_chunk<=0 = one call (old behavior).
    chunk = fusion_chunk if (fusion_chunk and fusion_chunk > 0) else (len(queries) or 1)
    pbar = None
    if show_progress:
        try:
            from tqdm.auto import tqdm
            pbar = tqdm(total=len(queries), desc="fusion pools (turns)", unit="turn")
        except Exception:
            pbar = None
    pools: list = []
    for s in range(0, len(queries), chunk):
        e = s + chunk
        pools.extend(fusion.batch_text_to_item_retrieval(
            queries[s:e], pool_size, batch_context=bc[s:e], user_ids=uids[s:e]))
        if pbar is not None:
            pbar.update(min(chunk, len(queries) - s))
    if pbar is not None:
        pbar.close()

    triples, stats = build_triples_from_pools(positives, pools, doc_text_fn, k_negs, min_negs=min_negs,
                                              title_fn=title_fn, artist_fn=artist_fn,
                                              drop_same_artist=drop_same_artist,
                                              teacher_score_fn=teacher_score_fn, fp_quantile=fp_quantile)
    if report is not None:
        report.update(positives=len(positives), **stats)
    return triples


def build_dev_eval_pack(
    conversations: Any,
    query_builder: Any,
    fusion: Any,
    doc_text_fn: Callable[[str], str],
    *,
    pool_size: int = 100,
    final_turn: bool = True,
) -> dict:
    """Build the dev-eval pack (DEV/TEST split) for in-loop checkpoint selection. SAME `query_builder`
    + `fusion` + `doc_text_fn` as train, so the dev re-probe matches serve. Returns the pack for
    `colbert_finetune.make_dev_eval_callback`.

    BUG #1 fix (selection/serve skew): Blind-A scores ONE warm FINAL turn per session, so the
    checkpoint MUST be selected on that SERVED distribution. With `final_turn=True` (default) the pack
    is built from `conversations.gold_target_turns()` — each session's trailing gold-bearing turn (the
    final-turn warm proxy) — and EVERY such row is on the selection wall. The previous behavior selected
    on a turn-1-COLD filter (`turns()` restricted to turn 1), capping the recall channel on the wrong
    curve. Pass `final_turn=False` to restore that legacy turn-1 gate (kept for ablation/back-compat)."""
    if final_turn:
        turns = list(conversations.gold_target_turns())     # served final-turn warm proxy (Blind-A curve)
        wall_value: Optional[bool] = True                   # every served row is on the selection wall
    else:
        turns = [t for t in conversations.turns() if int(t.turn_number) == 1]   # legacy turn-1 gate
        wall_value = None
    rows = []
    for t in turns:
        rows.append({
            "query": query_builder.build(t).text,
            "gold_tid": conversations.gold(t.session_id, t.turn_number),
            "turn_number": int(t.turn_number),
            "history_tids": list(t.history_tids),
            "user_id": t.user_id,
            "segment": t.segment,
        })
    queries = [r["query"] for r in rows]
    bc = [{"history_tids": r["history_tids"], "user_id": r["user_id"], "segment": r["segment"]}
          for r in rows]
    uids = [r["user_id"] for r in rows]
    pools = fusion.batch_text_to_item_retrieval(queries, pool_size, batch_context=bc, user_ids=uids)
    return dev_eval_pack_from_pools(rows, pools, doc_text_fn, wall_value=wall_value)
