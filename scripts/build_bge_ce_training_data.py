# scripts/build_bge_ce_training_data.py  (top of file)
"""Plain bge cross-encoder training-data builder (Path A v2).
Walks HF train conversations, mines hard negatives from THE canonical serve union
(UNION_EC: sasrec+colbert+bm25+dense+same_artist, clap off, PG off) with real
per-turn history ctx, and writes the EXACT compact query + tag-enriched pipe doc
text the serve bge_reranker uses. One JSONL row per music turn."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "music-crs-baselines")); sys.path.insert(0, str(REPO_ROOT / "scripts"))


def select_ce_negatives(gold_tid: str, candidate_tids: list[str], max_negatives: int) -> list[str]:
    """Pool minus the gold, hardest-first (pool rank order), capped."""
    return [t for t in candidate_tids if t != gold_tid][:max_negatives]


def build_ce_row(query: str, gold_tid: str, neg_tids: list[str], text_map: dict[str, str],
                 user_id: Optional[str], session_id: Optional[str]) -> dict[str, Any]:
    """One JSONL triple; negs absent from text_map dropped from BOTH neg & neg_tids."""
    kept = [t for t in neg_tids if t in text_map]
    return {"query": query, "pos": text_map[gold_tid], "neg": [text_map[t] for t in kept],
            "pos_tid": gold_tid, "neg_tids": kept, "user_id": user_id, "session_id": session_id}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-conv-hf", default="talkpl-ai/TalkPlayData-Challenge-Dataset")
    p.add_argument("--track-meta-hf", default="talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
    p.add_argument("--output", required=True)
    p.add_argument("--cache-dir", default="../experiments/cache")
    p.add_argument("--pool-size", type=int, default=300)
    p.add_argument("--max-negatives", type=int, default=63)
    p.add_argument("--doc-corpus", default="track_name,artist_name,album_name,tag_list")
    p.add_argument("--retrieval-query-mode", default="raw_enriched")
    p.add_argument("--ce-query-mode", default="compact_colbert")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-rows", type=int, default=0)
    args = p.parse_args()

    import pandas as pd
    from datasets import load_dataset
    from mcrs.retrieval_modules import load_retrieval_module
    from mcrs.crs_baseline import build_retrieval_query
    from mcrs.retrieval_modules.sasrec_model import build_user_dialog, prior_turns
    from mcrs.rerankers.bge_reranker import build_tid_text_map
    from mcrs.db_item.music_catalog import MusicCatalogDB

    doc_corpus = [c.strip() for c in args.doc_corpus.split(",") if c.strip()]
    base_corpus = ["track_name", "artist_name", "album_name"]

    # 1) catalog -> full metadata rows -> CE doc text (shared helper = serve parity)
    item_db = MusicCatalogDB(args.track_meta_hf, ["all_tracks"], base_corpus)
    text_map = build_tid_text_map(item_db.metadata_dict, doc_corpus)

    # 2) THE canonical serve union (UNION_EC). Must match nb90 gate + config 224.
    union = load_retrieval_module(
        "wrrf_union_v1", args.track_meta_hf, ["all_tracks"], base_corpus, args.cache_dir,
        extra_config={"use_sasrec": True, "w_sasrec": 1.0, "use_colbert": True, "w_colbert": 1.0,
                      "colbert_index_name": "colbert-music-v1", "colbert_model": None,
                      "colbert_q_len": 96, "colbert_compact_query": True,
                      "use_clap_text": False, "use_propose_ground": False})

    # 3) walk conversations -> per-music-turn rows with REAL history (fixes BLOCKER-1)
    conv = load_dataset(args.train_conv_hf, split="train")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    buf_raw, buf_cmp, buf_ctx, buf_uid, buf_gold, buf_sess = [], [], [], [], [], []
    # Open once in "w" (truncate on fresh run) so a Colab restart re-writes rather
    # than silently appending duplicate rows; handle held across all flush batches.
    out_f = open(args.output, "w")

    def flush():
        nonlocal n_written
        if not buf_raw:
            return
        cands = union.batch_text_to_item_retrieval(
            buf_raw, topk=args.pool_size, user_ids=buf_uid, batch_context=buf_ctx)
        for j in range(len(buf_raw)):
            gold = buf_gold[j]
            if gold not in text_map:
                continue
            negs = select_ce_negatives(gold, list(cands[j]), args.max_negatives)
            row = build_ce_row(buf_cmp[j], gold, negs, text_map, buf_uid[j], buf_sess[j])
            if len(row["neg"]) < 2:
                continue
            out_f.write(json.dumps(row) + "\n"); n_written += 1
        buf_raw.clear(); buf_cmp.clear(); buf_ctx.clear(); buf_uid.clear(); buf_gold.clear(); buf_sess.clear()

    n_rows = 0
    for sess in conv:
        df = pd.DataFrame(sess["conversations"])
        goal = (sess.get("conversation_goal") or {}).get("listener_goal", "") or ""
        up = sess.get("user_profile"); uid = sess.get("user_id"); sid = sess.get("session_id")
        for _, music in df[df["role"] == "music"].iterrows():
            tn = int(music["turn_number"])
            # Single source of truth for the conditioning slice (train/serve parity).
            prior = prior_turns(df, tn)
            prior_rows = [{"role": ("assistant" if t["role"] == "music" else t["role"]),
                           "content": (item_db.id_to_metadata(t["content"]) if t["role"] == "music"
                                       else t["content"])} for _, t in prior.iterrows()]
            played = list(df[(df["role"] == "music") & (df["turn_number"] < tn)]["content"])
            raw_q = build_retrieval_query(prior_rows, mode=args.retrieval_query_mode,
                                          goal_text=goal, user_profile=up)
            cmp_q = build_retrieval_query(prior_rows, mode=args.ce_query_mode,
                                          goal_text=goal, user_profile=up)
            buf_raw.append(raw_q); buf_cmp.append(cmp_q)
            buf_ctx.append({"history_tids": played, "user_dialog": build_user_dialog(prior.to_dict("records")),
                            "colbert_query": cmp_q})
            buf_uid.append(uid); buf_gold.append(music["content"]); buf_sess.append(sid)
            if len(buf_raw) >= args.batch_size:
                flush()
            n_rows += 1
            if args.max_rows and n_rows >= args.max_rows:
                flush(); out_f.close()
                print(f"[ce-build] DONE (max-rows) -> {args.output} ({n_written})", file=sys.stderr)
                return
    flush()
    out_f.close()
    print(f"[ce-build] DONE -> {args.output} ({n_written} rows from {n_rows} turns)", file=sys.stderr)


if __name__ == "__main__":
    main()
