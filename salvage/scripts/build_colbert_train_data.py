"""ColBERT fine-tune training-data builder (W2.a of the ColBERT plan).

Walks the conversation dataset TRAIN split → per (session, music-turn) filtered to
MOVES_TOWARD_GOAL → builds the canonical dialog query (nb74/nb82 parity), the gold
track doc text, and K hard negatives mined from the SASRec-free recall pool's
non-gold items → writes JSONL triples consumable by `scripts/train_colbert.py`.

Leak discipline (plan §11): TRAIN SPLIT ONLY. `goal_progress_assessments` is used
to *filter training targets* offline; it is NEVER a serve input.

Train/serve parity: `build_dialog_query` reproduces nb74 cell-4 / nb82 `#82-probe`
query construction EXACTLY, so the fine-tuned model sees the same query
distribution the re-probe reranks with.

Usage:
  python scripts/build_colbert_train_data.py \
    --output experiments/cache/retrieval_v2/colbert_train.jsonl \
    --cache-dir experiments/cache --pool-size 100 --k-negs 15 --max-rows 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Optional

LABEL_POS = "MOVES_TOWARD_GOAL"


# --------------------------------------------------------------------------- #
# Pure functions (unit-tested; no I/O, no model).
# --------------------------------------------------------------------------- #
def goal_progress_label(session: dict, turn_number: int) -> Optional[str]:
    """Return the goal-progress label for `turn_number`, or None if unlabeled."""
    for x in (session.get("goal_progress_assessments") or []):
        if int(x.get("turn_number")) == int(turn_number):
            return x.get("goal_progress_assessment")
    return None


def build_dialog_query(
    prior_records: list[dict], goal_txt: str, id_to_metadata: Callable[[str], str]
) -> str:
    """Reproduce nb74 cell-4 / nb82 probe query EXACTLY: newline-joined
    'role: content' lines (music role -> 'assistant', content -> id_to_metadata),
    then '\\ngoal: <listener_goal>' appended when a goal is present.
    """
    lines = []
    for t in prior_records:
        role = "assistant" if t["role"] == "music" else t["role"]
        content = id_to_metadata(t["content"]) if t["role"] == "music" else t["content"]
        lines.append(f"{role}: {content}")
    q = "\n".join(lines)
    if goal_txt:
        q = q + "\n" + "goal: " + goal_txt
    return q


def _prior_slice(conversations: list[dict], tn: int) -> list[dict]:
    """All turns before `tn` (any role) + the current-turn user request — the
    `prior_turns` semantics, in plain python (order-preserving)."""
    return [
        t for t in conversations
        if int(t["turn_number"]) < tn
        or (int(t["turn_number"]) == tn and t["role"] == "user")
    ]


def iter_positive_turns(
    session: dict, id_to_metadata: Callable[[str], str],
    compact_query: bool = False,
) -> list[dict[str, Any]]:
    """Yield one positive-training row per qualifying music turn: {query, gold_tid,
    turn_number, session_id, user_id, played_tids}.

    Query construction:
      compact_query=False (default): legacy nb74-parity full-dialog query
        (build_dialog_query — music turns rendered as metadata, goal appended).
      compact_query=True: the SHARED compact ColBERT query (goal + culture + last
        user turn) via build_retrieval_query(mode="compact_colbert") — byte-identical
        to serve/Blind-A, so the retrain has no train/serve skew. Needs the session's
        user_profile (for culture); both train and Blind-A carry it.

    TURN-1 IS ALWAYS INCLUDED (gold-direct), regardless of label — real turn-1 has
    NO goal_progress_assessment, so the MOVES_TOWARD_GOAL filter used to drop 100%
    of turn-1 while the gate is 100% turn-1 (RCA #1). Turns >1 still require
    MOVES_TOWARD_GOAL (strips noisy mid-conversation targets).
    """
    goal_txt = ((session.get("conversation_goal") or {}).get("listener_goal") or "").strip()
    convos = session.get("conversations") or []
    user_profile = session.get("user_profile")
    if compact_query:
        # Lazy import: keeps the script's pure functions import-light (mirrors the
        # in-main() mcrs imports); only paid when actually building compact data.
        from mcrs.crs_baseline import build_retrieval_query
    rows: list[dict[str, Any]] = []
    for t in convos:
        if t["role"] != "music":
            continue
        tn = int(t["turn_number"])
        if tn != 1 and goal_progress_label(session, tn) != LABEL_POS:
            continue
        prior = _prior_slice(convos, tn)
        if compact_query:
            query = build_retrieval_query(
                prior, mode="compact_colbert", goal_text=goal_txt,
                user_profile=user_profile)
        else:
            query = build_dialog_query(prior, goal_txt, id_to_metadata)
        rows.append({
            "query": query,
            "gold_tid": t["content"],
            "turn_number": tn,
            "session_id": session.get("session_id") or session.get("id"),
            "user_id": session.get("user_id"),
            "played_tids": [
                x["content"] for x in convos
                if x["role"] == "music" and int(x["turn_number"]) < tn
            ],
        })
    return rows


def select_hard_negatives(
    pool: list[str], gold_tid: str, k: int
) -> Optional[list[str]]:
    """Top-k non-gold pool items (pool order = hardest first). Returns None when
    the gold is not in the pool — a recall miss the ranker can't recover, so the
    row is skipped (plan §6 gold-not-in-pool=skip)."""
    if gold_tid not in pool:
        return None
    return [t for t in pool if t != gold_tid][:k]


def build_colbert_triple(
    query: str, pos_text: str, neg_texts: list[str],
    pos_tid: str, neg_tids: list[str], session_id: Optional[str], turn_number: int,
) -> dict[str, Any]:
    """One JSONL training row. `negatives` are doc texts (what ColBERT trains on);
    `*_tids` are kept for diagnostics / val nDCG."""
    return {
        "query": query,
        "positive": pos_text,
        "negatives": neg_texts,
        "pos_tid": pos_tid,
        "neg_tids": neg_tids,
        "session_id": session_id,
        "turn_number": turn_number,
    }


# --------------------------------------------------------------------------- #
# CLI wiring (I/O; exercised in the notebook, not unit-tested).
# --------------------------------------------------------------------------- #
def main():  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-conv-hf", default="talkpl-ai/TalkPlayData-Challenge-Dataset")
    ap.add_argument("--track-meta-hf", default="talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
    ap.add_argument("--output", required=True, help="Output JSONL path")
    ap.add_argument("--cache-dir", default="experiments/cache")
    ap.add_argument("--pool-size", type=int, default=100, help="SASRec-free recall pool size")
    ap.add_argument("--k-negs", type=int, default=15, help="Hard negatives per query")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-rows", type=int, default=0, help="Smoke cap; 0 = all")
    ap.add_argument("--compact-query", action="store_true",
                    help="Build the compact ColBERT query (goal+culture+last user turn, "
                         "shared build_retrieval_query mode='compact_colbert') instead of "
                         "the legacy full-dialog query. MUST match serve (colbert_compact_query).")
    ap.add_argument("--enrich-tags", action="store_true",
                    help="EXP-217: append curated genre/mood tags to each DOC (positives + "
                         "negatives). MUST match build_colbert_index --enrich-tags + same "
                         "--tag-min-freq/--tag-top-k (doc-side train/index parity).")
    ap.add_argument("--tag-min-freq", type=int, default=50)
    ap.add_argument("--tag-top-k", type=int, default=15)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "music-crs-baselines"))
    from datasets import load_dataset
    from tqdm import tqdm
    from mcrs.db_item.music_catalog import MusicCatalogDB
    from mcrs.retrieval_modules import load_retrieval_module
    from mcrs.retrieval_modules.colbert_late import make_colbert_doc_text_fn

    corpus = ["track_name", "artist_name", "album_name"]
    item_db = MusicCatalogDB(args.track_meta_hf, ["all_tracks"], corpus)
    id2meta = item_db.id_to_metadata  # raw (with track_id) — for the query/dialog rendering
    # ColBERT DOC text via the SHARED factory (same recipe must be used by
    # build_colbert_index + nb82 dev-eval/reprobe — print the recipe so a mismatch shows).
    doc_text, _doc_recipe = make_colbert_doc_text_fn(
        item_db, enrich_tags=args.enrich_tags,
        tag_min_freq=args.tag_min_freq, tag_top_k=args.tag_top_k)
    print(f"[colbert-data] DOC RECIPE: {_doc_recipe}", file=sys.stderr)

    # 1) Collect MOVES_TOWARD_GOAL rows from the TRAIN split.
    conv = load_dataset(args.train_conv_hf, split="train")
    rows: list[dict] = []
    for sess in tqdm(conv, desc="sessions"):
        rows.extend(iter_positive_turns(sess, id2meta, compact_query=args.compact_query))
        if args.max_rows and len(rows) >= args.max_rows:
            rows = rows[: args.max_rows]
            break
    print(f"[colbert-data] {len(rows)} MOVES_TOWARD_GOAL turns (train)", file=sys.stderr)

    # 2) SASRec-free recall pool per query (bm25 + dense + same_artist; no SASRec).
    union = load_retrieval_module("wrrf_union_v1", args.track_meta_hf, ["all_tracks"],
                                  corpus, args.cache_dir, extra_config={})
    queries = [r["query"] for r in rows]
    ctx = [{"history_tids": r["played_tids"]} for r in rows]
    pools: list[list[str]] = []
    for s in tqdm(range(0, len(queries), args.batch_size), desc="pool"):
        pools.extend(union.batch_text_to_item_retrieval(
            queries[s:s + args.batch_size], topk=args.pool_size,
            batch_context=ctx[s:s + args.batch_size]))

    # 3) Build triples (skip gold-not-in-pool) and write JSONL.
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    n_written = n_skipped = 0
    with open(args.output, "w") as f:
        for r, pool in zip(rows, pools):
            neg_tids = select_hard_negatives(pool, r["gold_tid"], args.k_negs)
            if neg_tids is None:
                n_skipped += 1
                continue
            triple = build_colbert_triple(
                query=r["query"], pos_text=doc_text(r["gold_tid"]),
                neg_texts=[doc_text(t) for t in neg_tids],
                pos_tid=r["gold_tid"], neg_tids=neg_tids,
                session_id=r["session_id"], turn_number=r["turn_number"])
            f.write(json.dumps(triple) + "\n")
            n_written += 1
    print(f"[colbert-data] DONE -> {args.output} "
          f"(wrote {n_written}; skipped_no_gold={n_skipped})", file=sys.stderr)


if __name__ == "__main__":  # pragma: no cover
    main()
