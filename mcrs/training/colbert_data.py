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
                         turn_number: int) -> dict[str, Any]:
    """One JSONL training row. `negatives` are doc texts (what ColBERT trains on); `*_tids` are
    kept for diagnostics / val nDCG."""
    return {
        "query": query,
        "positive": pos_text,
        "negatives": list(neg_texts),
        "pos_tid": pos_tid,
        "neg_tids": list(neg_tids),
        "session_id": session_id,
        "turn_number": turn_number,
    }


def build_triples_from_pools(
    positive_rows: Sequence[dict],
    pools: Sequence[Sequence[str]],
    doc_text_fn: Callable[[str], str],
    k_negs: int,
    min_negs: int = 1,
) -> tuple[list[dict], dict]:
    """Assemble contrastive triples from positive rows + their aligned candidate pools.

    `pools[i]` is the hardest-first candidate tids for `positive_rows[i]` (e.g. an RRF fusion pool).
    For each row: take the gold from the pool's negatives (`select_hard_negatives`), render gold +
    negatives to doc text, emit a `build_colbert_triple`. Drops the row when the gold is not in the
    pool (unrecoverable recall miss) or it yields < `min_negs` negatives (no pair to contrast).
    Returns (triples, stats) where stats = {kept, dropped_no_gold, dropped_no_neg}."""
    triples: list[dict] = []
    dropped_no_gold = dropped_no_neg = 0
    for row, pool in zip(positive_rows, pools):
        gold = row["gold_tid"]
        negs = select_hard_negatives(pool, gold, k_negs)
        if negs is None:
            dropped_no_gold += 1
            continue
        if len(negs) < min_negs:
            dropped_no_neg += 1
            continue
        triples.append(build_colbert_triple(
            query=row["query"], pos_text=doc_text_fn(gold), neg_texts=[doc_text_fn(t) for t in negs],
            pos_tid=gold, neg_tids=negs, session_id=row.get("session_id"),
            turn_number=row["turn_number"]))
    stats = {"kept": len(triples), "dropped_no_gold": dropped_no_gold, "dropped_no_neg": dropped_no_neg}
    return triples, stats


def dev_eval_pack_from_pools(
    rows: Sequence[dict],
    pools: Sequence[Sequence[str]],
    doc_text_fn: Callable[[str], str],
) -> dict:
    """Assemble the dev-eval pack the fine-tune selection callback consumes, from dev rows + their
    aligned candidate pools. Each row has {query, gold_tid, turn_number}. Drops rows with no gold
    (can't score recall). `wall[i]` marks the turn-1 gate (the metric the checkpoint is selected on);
    `tid_to_text` covers the union of all pool tids so the live model can re-encode + rerank them."""
    queries, golds, out_pools, wall = [], [], [], []
    tid_to_text: dict[str, str] = {}
    for row, pool in zip(rows, pools):
        gold = row["gold_tid"]
        if gold is None:
            continue
        queries.append(row["query"])
        golds.append(gold)
        out_pools.append(list(pool))
        wall.append(int(row["turn_number"]) == 1)
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
        rows.append({
            "query": query_builder.build(ctx).text,
            "gold_tid": gold,
            "turn_number": tn,
            "session_id": ctx.session_id,
            "user_id": ctx.user_id,
            "history_tids": list(ctx.history_tids),
        })
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
    positives = iter_colbert_positives(turns, gold_fn, query_builder, gp_fn)

    queries = [r["query"] for r in positives]
    bc = [{"history_tids": r["history_tids"], "user_id": r["user_id"]} for r in positives]
    uids = [r["user_id"] for r in positives]
    rng = range(len(queries))
    if show_progress:
        try:
            from tqdm.auto import tqdm
            rng = tqdm(rng, desc="fusion pools (turns)")  # noqa: F841 (kept for symmetry/UX)
        except Exception:
            pass
    pools = fusion.batch_text_to_item_retrieval(queries, pool_size, batch_context=bc, user_ids=uids)

    triples, stats = build_triples_from_pools(positives, pools, doc_text_fn, k_negs, min_negs=min_negs)
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
) -> dict:
    """Build the dev-eval pack (DEV/TEST split) for in-loop checkpoint selection. Gate = turn-1
    recall (the probe metric), so we evaluate turn-1 dev turns only. SAME `query_builder` + `fusion`
    + `doc_text_fn` as train, so the dev re-probe matches serve. Returns the pack for
    `colbert_finetune.make_dev_eval_callback`."""
    turns = [t for t in conversations.turns() if int(t.turn_number) == 1]
    rows = []
    for t in turns:
        rows.append({
            "query": query_builder.build(t).text,
            "gold_tid": conversations.gold(t.session_id, t.turn_number),
            "turn_number": 1,
            "history_tids": list(t.history_tids),
            "user_id": t.user_id,
        })
    queries = [r["query"] for r in rows]
    bc = [{"history_tids": r["history_tids"], "user_id": r["user_id"]} for r in rows]
    uids = [r["user_id"] for r in rows]
    pools = fusion.batch_text_to_item_retrieval(queries, pool_size, batch_context=bc, user_ids=uids)
    return dev_eval_pack_from_pools(rows, pools, doc_text_fn)
