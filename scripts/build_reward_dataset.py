"""Build the training dataset for the response-quality reward model.

For each train conversation, for each turn N >= 2 with a goal_progress_assessment
label, emit ONE training row:
    text_a: "query + goal + recommended_track_meta" (the *context* being scored)
    text_b: "response" (the assistant message being scored)
    label : 1 if goal_progress_assessment == MOVES_TOWARD_GOAL, else 0

This powers a cross-encoder that at inference time will score K sampled
responses per query and pick the one most likely to advance the user's goal.
Unlike a generic semantic reranker (which exp 023 showed is misaligned),
this reward model is trained on the TASK-SPECIFIC signal — the same kind
of user-preference feedback the Blind-A Gemini judge approximates.

Output: parquet with columns (text_a, text_b, label, session_id, turn_number)
Default out: data/reward_train.parquet  (~60–80k rows across 15k sessions)

Usage:
    python scripts/build_reward_dataset.py
    python scripts/build_reward_dataset.py --n-sessions 15000 --out data/reward_train.parquet
    python scripts/build_reward_dataset.py --split-val 0.1  # hold out 10% for validation
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINES_DIR = REPO_ROOT / "music-crs-baselines"
sys.path.insert(0, str(BASELINES_DIR))

import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402
from datasets import load_dataset  # noqa: E402

from mcrs.db_item import MusicCatalogDB  # noqa: E402


LABEL_POS = "MOVES_TOWARD_GOAL"
LABEL_NEG = "DOES_NOT_MOVE_TOWARD_GOAL"
MAX_HISTORY_TURNS = 4  # cap history context; reward model input tokens are precious
MAX_RESPONSE_CHARS = 800  # ~200 tokens; responses are ~150 words typically


def _first(v):
    if isinstance(v, list):
        return v[0] if v else None
    return v


def track_summary(track_id: str, item_db: MusicCatalogDB) -> str:
    """Short metadata string for the recommended track."""
    try:
        meta = item_db.metadata_dict.get(track_id, {})
    except Exception:
        return "(unknown track)"
    name = _first(meta.get("track_name")) or ""
    artist = _first(meta.get("artist_name")) or ""
    tags = meta.get("tag_list") or []
    if isinstance(tags, list):
        tags_str = ", ".join(tags[:5])
    else:
        tags_str = str(tags)[:80]
    parts = []
    if name:
        parts.append(name)
    if artist:
        parts.append(f"by {artist}")
    if tags_str:
        parts.append(f"[{tags_str}]")
    return " ".join(parts)


def history_summary(df, turn_n: int, item_db: MusicCatalogDB) -> str:
    """Last MAX_HISTORY_TURNS of dialog, with music turns expanded to metadata."""
    prior = df[df["turn_number"] < turn_n]
    if prior.empty:
        return ""
    prior = prior.tail(MAX_HISTORY_TURNS * 3)  # 3 roles per turn (user/music/assistant)
    lines = []
    for _, t in prior.iterrows():
        role = "assistant" if t["role"] == "music" else t["role"]
        content = t["content"]
        if t["role"] == "music":
            content = track_summary(content, item_db)
        content_str = str(content)[:200]
        lines.append(f"{role}: {content_str}")
    return "\n".join(lines)


def build(n_sessions: int, seed: int, out_path: str, split_val: float) -> None:
    print(f"[reward-data] loading train split")
    tr = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="train")
    rng = random.Random(seed)
    if n_sessions < len(tr):
        indices = rng.sample(range(len(tr)), n_sessions)
        sessions = tr.select(indices).to_list()
    else:
        sessions = tr.to_list()
    print(f"[reward-data] using {len(sessions)} sessions (seed={seed})")

    print(f"[reward-data] loading item_db")
    item_db = MusicCatalogDB(
        "talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
        ["all_tracks"],
        ["track_name", "artist_name", "album_name"],
    )

    rows = []
    skipped_none = 0
    skipped_missing = 0
    for sess in tqdm(sessions, desc="sessions"):
        convos = sess.get("conversations") or []
        if not convos:
            continue
        df = pd.DataFrame(convos)
        gpa_map = {
            int(x["turn_number"]): x.get("goal_progress_assessment")
            for x in (sess.get("goal_progress_assessments") or [])
        }
        goal = sess.get("conversation_goal") or {}
        goal_listener = (goal.get("listener_goal") or "").strip()
        goal_category = goal.get("category") or "?"

        for turn_n in sorted(df["turn_number"].unique()):
            tn = int(turn_n)
            label_raw = gpa_map.get(tn)
            if label_raw == LABEL_POS:
                label = 1
            elif label_raw == LABEL_NEG:
                label = 0
            else:
                skipped_none += 1
                continue

            turn_df = df[df["turn_number"] == tn]
            user_row = turn_df[turn_df["role"] == "user"]
            music_row = turn_df[turn_df["role"] == "music"]
            asst_row = turn_df[turn_df["role"] == "assistant"]
            if user_row.empty or music_row.empty or asst_row.empty:
                skipped_missing += 1
                continue

            user_query = str(user_row.iloc[0]["content"])[:500]
            rec_tid = str(music_row.iloc[0]["content"])
            rec_meta = track_summary(rec_tid, item_db)
            response = str(asst_row.iloc[0]["content"])[:MAX_RESPONSE_CHARS]
            history = history_summary(df, tn, item_db)

            # text_a = the "context" the response is being judged against.
            # text_b = the response itself. Cross-encoder scores (a, b) pairs.
            text_a_parts = [
                f"User query: {user_query}",
                f"Listener goal: {goal_listener[:250]}" if goal_listener else "",
                f"Goal category: {goal_category}",
                f"Recommended track: {rec_meta}",
                f"Prior dialog: {history}" if history else "",
            ]
            text_a = "\n".join(p for p in text_a_parts if p)

            rows.append({
                "text_a": text_a,
                "text_b": response,
                "label": label,
                "session_id": sess["session_id"],
                "user_id": sess["user_id"],
                "turn_number": tn,
                "goal_category": goal_category,
            })

    df_out = pd.DataFrame(rows)
    print(f"[reward-data] rows built: {len(df_out)}  (skipped: {skipped_none} None-labels, {skipped_missing} missing-roles)")
    print(f"[reward-data] positives: {int(df_out['label'].sum())}  negatives: {len(df_out) - int(df_out['label'].sum())}")

    # Split into train/val by session_id to avoid leakage.
    if split_val > 0:
        all_sess = sorted(df_out["session_id"].unique())
        rng.shuffle(all_sess)
        n_val = int(len(all_sess) * split_val)
        val_sess = set(all_sess[:n_val])
        df_out["split"] = df_out["session_id"].map(lambda s: "val" if s in val_sess else "train")
        print(f"[reward-data] split: train={int((df_out['split']=='train').sum())} val={int((df_out['split']=='val').sum())}")
    else:
        df_out["split"] = "train"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_out.to_parquet(out_path, index=False)
    print(f"[reward-data] wrote {out_path}  ({os.path.getsize(out_path) / 1e6:.1f} MB)")


def main() -> int:
    p = argparse.ArgumentParser(description="Build cross-encoder reward-model training data from train conversations.")
    p.add_argument("--n-sessions", type=int, default=15000,
                   help="Max train sessions (default: 15000 = all).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default=str(REPO_ROOT / "data" / "reward_train.parquet"))
    p.add_argument("--split-val", type=float, default=0.1,
                   help="Fraction of sessions held out for validation (default 0.1).")
    args = p.parse_args()

    origin_cwd = os.getcwd()
    os.chdir(BASELINES_DIR)
    try:
        build(args.n_sessions, args.seed, args.out, args.split_val)
    finally:
        os.chdir(origin_cwd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
