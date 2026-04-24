"""Compact exploration of all TalkPlay datasets under data/.

Run from repo root:
    python documents/analysis/explore_data.py

Output is structured into labeled sections so results can be lifted into
documents/data_exploration.md without editing.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from datasets import load_from_disk

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"

DATASETS = {
    "sessions": DATA_ROOT / "TalkPlayData-Challenge-Dataset",
    "blind_a": DATA_ROOT / "TalkPlayData-Challenge-Blind-A",
    "track_meta": DATA_ROOT / "TalkPlayData-Challenge-Track-Metadata",
    "track_emb": DATA_ROOT / "TalkPlayData-Challenge-Track-Embeddings",
    "user_meta": DATA_ROOT / "TalkPlayData-Challenge-User-Metadata",
    "user_emb": DATA_ROOT / "TalkPlayData-Challenge-User-Embeddings",
}


def banner(title: str) -> None:
    print(f"\n{'=' * 8} {title} {'=' * 8}")


def percentiles(values: Iterable[int | float], qs=(0, 50, 95, 100)) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {f"p{q}": float("nan") for q in qs}
    return {f"p{q}": float(np.percentile(arr, q)) for q in qs}


def topk_counter(values: Iterable[Any], k: int = 6) -> list[tuple[Any, int]]:
    c = Counter(values)
    return c.most_common(k)


def describe_table(tbl: pa.Table, label: str) -> None:
    print(f"[{label}] rows={tbl.num_rows}  cols={tbl.num_columns}")
    print("  schema:")
    for f in tbl.schema:
        print(f"    - {f.name}: {f.type}")


# ---------------------------------------------------------------------------
# Sessions (train + test) and Blind-A
# ---------------------------------------------------------------------------

def explore_sessions(path: Path, split: str) -> None:
    banner(f"SESSIONS {path.name}/{split}")
    ds = load_from_disk(str(path))[split]
    tbl = ds.data.table
    describe_table(tbl, f"{path.name}/{split}")

    # IDs
    n = ds.num_rows
    uniq_session = len(set(ds["session_id"]))
    uniq_user = len(set(ds["user_id"]))
    print(f"  unique session_id={uniq_session}  unique user_id={uniq_user}")

    # session_date range
    dates = [d for d in ds["session_date"] if d]
    print(f"  session_date range: {min(dates)} .. {max(dates)}  (non-null={len(dates)}/{n})")

    # user_profile (nested struct): pull fields via pyarrow
    prof = tbl.column("user_profile")
    # Flatten the struct to Python list of dicts lazily via column access.
    for fname in ["age", "age_group", "country_code", "country_name", "gender",
                   "preferred_language", "preferred_musical_culture", "user_split"]:
        col = prof.combine_chunks().field(fname).to_pylist()
        nulls = sum(1 for x in col if x is None or x == "")
        if fname == "age":
            vals = [int(x) for x in col if x is not None]
            print(f"  user_profile.age: min={min(vals)} median={median(vals):.0f} max={max(vals)} mean={mean(vals):.1f} nulls={nulls}")
        else:
            top = topk_counter([x for x in col if x is not None])
            print(f"  user_profile.{fname}: nulls={nulls}  top={top}")

    # conversation_goal
    goal = tbl.column("conversation_goal").combine_chunks()
    for fname in ["category", "listener_goal", "specificity"]:
        col = goal.field(fname).to_pylist()
        nulls = sum(1 for x in col if x is None or x == "")
        top = topk_counter([x for x in col if x is not None])
        print(f"  conversation_goal.{fname}: nulls={nulls}  top={top}")

    # conversations list
    convs = ds["conversations"]
    turns_per_session = [len(c) for c in convs]
    print(f"  conversations[] length: {percentiles(turns_per_session)}")

    # role distribution across ALL turns
    roles: list[str] = []
    content_lens: list[int] = []
    thought_lens: list[int] = []
    thought_empty = 0
    thought_total = 0
    for c in convs:
        for turn in c:
            roles.append(turn.get("role"))
            content = turn.get("content") or ""
            content_lens.append(len(content))
            th = turn.get("thought")
            thought_total += 1
            if th is None or th == "":
                thought_empty += 1
            else:
                thought_lens.append(len(th))
    print(f"  conversations.role: {topk_counter(roles)}")
    print(f"  conversations.content len chars: {percentiles(content_lens)}")
    print(f"  conversations.thought len chars: {percentiles(thought_lens)}  empty={thought_empty}/{thought_total}")

    # goal_progress_assessments length
    gpa = ds["goal_progress_assessments"]
    gpa_lens = [len(g) for g in gpa]
    print(f"  goal_progress_assessments[] length: {percentiles(gpa_lens)}")

    # Detect track_ids referenced in assistant responses (rough proxy)
    # Look at first session: what keys does a turn have?
    print(f"  sample turn keys: {sorted(convs[0][0].keys())}")
    print(f"  sample turn[0] role={convs[0][0]['role']!r}  content[:120]={convs[0][0]['content'][:120]!r}")


# ---------------------------------------------------------------------------
# Track metadata
# ---------------------------------------------------------------------------

def explore_track_meta(path: Path) -> tuple[set[str], set[str]]:
    banner("TRACK METADATA")
    out_ids: dict[str, set[str]] = {}
    for split in ["all_tracks", "test_tracks"]:
        ds = load_from_disk(str(path))[split]
        tbl = ds.data.table
        describe_table(tbl, split)
        tids = ds["track_id"]
        out_ids[split] = set(tids)
        print(f"  unique track_id: {len(out_ids[split])}")

        if split == "all_tracks":
            # Popularity
            pop = [p for p in ds["popularity"] if p is not None]
            nulls = ds.num_rows - len(pop)
            print(f"  popularity: nulls={nulls}  min={min(pop):.4f}  median={median(pop):.4f}  "
                  f"mean={mean(pop):.4f}  max={max(pop):.4f}  std={pstdev(pop):.4f}")
            # Duration
            dur = [d for d in ds["duration"] if d is not None]
            print(f"  duration (sec): min={min(dur)}  median={median(dur)}  max={max(dur)}  mean={mean(dur):.1f}")
            # release_date
            rd = [r for r in ds["release_date"] if r]
            print(f"  release_date: non-null={len(rd)}/{ds.num_rows}  range={min(rd)} .. {max(rd)}")

            # List-valued fields: size distribution + samples
            for fname in ["ISRC", "track_name", "artist_name", "album_name",
                          "tag_list", "artist_id", "album_id"]:
                col = ds[fname]
                sizes = [len(x) if x is not None else 0 for x in col]
                empties = sum(1 for s in sizes if s == 0)
                print(f"  {fname}: len-pcts={percentiles(sizes)}  empty={empties}/{len(sizes)}")
                # show a sample non-empty
                sample = next((x for x in col if x), None)
                print(f"    sample: {sample!r}"[:180])

    return out_ids["all_tracks"], out_ids["test_tracks"]


# ---------------------------------------------------------------------------
# Track embeddings — handle memory carefully
# ---------------------------------------------------------------------------

EMB_COLS = [
    "audio-laion_clap", "image-siglip2", "cf-bpr",
    "attributes-qwen3_embedding_0.6b", "lyrics-qwen3_embedding_0.6b",
    "metadata-qwen3_embedding_0.6b",
]


def explore_track_embeddings(path: Path) -> tuple[set[str], set[str]]:
    banner("TRACK EMBEDDINGS")
    out_ids: dict[str, set[str]] = {}
    for split in ["all_tracks", "test_tracks"]:
        ds = load_from_disk(str(path))[split]
        tbl = ds.data.table
        describe_table(tbl, split)
        tids = ds["track_id"]
        out_ids[split] = set(tids)
        print(f"  unique track_id: {len(out_ids[split])}")

        for col_name in EMB_COLS:
            col = tbl.column(col_name)
            # list<float64> — use compute.list_value_length for O(n) without materializing floats
            lengths = pc.list_value_length(col).to_numpy(zero_copy_only=False)
            dims = set(int(x) for x in lengths if x is not None and x > 0)
            empty_rows = int(np.sum((lengths == 0) | (lengths == None)))
            # All-zero check on a sample to avoid materializing the full matrix
            # Sample up to 5000 non-empty rows.
            nonempty_idx = np.where(lengths > 0)[0]
            sample_n = min(5000, len(nonempty_idx))
            if sample_n > 0:
                rng = np.random.default_rng(0)
                picks = rng.choice(nonempty_idx, size=sample_n, replace=False)
                # Pull only sampled rows via .take
                sub = ds.select(picks.tolist())
                arr = np.asarray(sub[col_name], dtype=np.float64)
                norms = np.linalg.norm(arr, axis=1)
                all_zero_sample = int(np.sum(norms == 0))
                norm_p = np.percentile(norms, [5, 50, 95])
                print(f"  {col_name}: dim={sorted(dims)}  empty_rows={empty_rows}/{ds.num_rows}  "
                      f"norm[p5,p50,p95]=({norm_p[0]:.3f},{norm_p[1]:.3f},{norm_p[2]:.3f})  "
                      f"all_zero_in_{sample_n}_sample={all_zero_sample}")
            else:
                print(f"  {col_name}: dim=?  empty_rows={empty_rows}/{ds.num_rows}  (no non-empty rows)")
    return out_ids["all_tracks"], out_ids["test_tracks"]


# ---------------------------------------------------------------------------
# User metadata
# ---------------------------------------------------------------------------

def explore_user_meta(path: Path) -> set[str]:
    banner("USER METADATA")
    ds = load_from_disk(str(path))["all_users"]
    tbl = ds.data.table
    describe_table(tbl, "all_users")
    uids = set(ds["user_id"])
    print(f"  unique user_id: {len(uids)}")

    ages = [a for a in ds["age"] if a is not None]
    print(f"  age: min={min(ages)} median={median(ages):.0f} max={max(ages)} mean={mean(ages):.1f}")

    for fname in ["age_group", "country_code", "country_name", "gender"]:
        col = ds[fname]
        nulls = sum(1 for x in col if x is None or x == "")
        print(f"  {fname}: nulls={nulls}  top={topk_counter([x for x in col if x])}")
    return uids


# ---------------------------------------------------------------------------
# User embeddings
# ---------------------------------------------------------------------------

def explore_user_embeddings(path: Path) -> dict[str, set[str]]:
    banner("USER EMBEDDINGS")
    out: dict[str, set[str]] = {}
    for split in ["train", "test_warm", "test_cold"]:
        ds = load_from_disk(str(path))[split]
        tbl = ds.data.table
        describe_table(tbl, split)
        out[split] = set(ds["user_id"])
        print(f"  unique user_id: {len(out[split])}")

        col = tbl.column("cf-bpr")
        lengths = pc.list_value_length(col).to_numpy(zero_copy_only=False)
        dims = set(int(x) for x in lengths if x is not None and x > 0)
        empty_rows = int(np.sum(lengths == 0))
        # Norm only on non-empty rows (test_cold has empties)
        nonempty_idx = np.where(lengths > 0)[0]
        if len(nonempty_idx) > 0:
            sub = ds.select(nonempty_idx.tolist())
            arr = np.asarray(sub["cf-bpr"], dtype=np.float64)
            norms = np.linalg.norm(arr, axis=1)
            all_zero = int(np.sum(norms == 0))
            norm_p = np.percentile(norms, [5, 50, 95])
            print(f"  cf-bpr: dim={sorted(dims)}  empty_rows={empty_rows}/{ds.num_rows}  "
                  f"all_zero={all_zero}  norm[p5,p50,p95]=({norm_p[0]:.3f},{norm_p[1]:.3f},{norm_p[2]:.3f})")
        else:
            print(f"  cf-bpr: dim={sorted(dims)}  empty_rows={empty_rows}/{ds.num_rows}  (all empty)")
    return out


# ---------------------------------------------------------------------------
# Cross-dataset relationships
# ---------------------------------------------------------------------------

def cross_dataset(sessions_train_users: set[str], sessions_test_users: set[str],
                  blind_users: set[str], user_meta_ids: set[str],
                  user_emb_ids: dict[str, set[str]],
                  tracks_all: set[str], tracks_test: set[str],
                  track_emb_all: set[str], track_emb_test: set[str]) -> None:
    banner("CROSS-DATASET OVERLAPS")
    # Users
    print(f"  sessions train users: {len(sessions_train_users)}")
    print(f"  sessions test  users: {len(sessions_test_users)}")
    print(f"  blind-A users:        {len(blind_users)}")
    print(f"  user-metadata:        {len(user_meta_ids)}")
    print(f"  user-emb train:       {len(user_emb_ids['train'])}")
    print(f"  user-emb test_warm:   {len(user_emb_ids['test_warm'])}")
    print(f"  user-emb test_cold:   {len(user_emb_ids['test_cold'])}")

    # Every session user in user-metadata?
    all_session_users = sessions_train_users | sessions_test_users | blind_users
    print(f"  union of session users: {len(all_session_users)}")
    print(f"  session users not in user-metadata: {len(all_session_users - user_meta_ids)}")
    print(f"  user-metadata users not in any session: {len(user_meta_ids - all_session_users)}")

    # Sessions-train users ⊆ user-emb train?
    overlap = sessions_train_users & user_emb_ids["train"]
    print(f"  sessions_train ∩ user_emb_train: {len(overlap)}")
    print(f"  sessions_train \\ user_emb_train: {len(sessions_train_users - user_emb_ids['train'])}")

    # test split composition
    test_warm_u = user_emb_ids["test_warm"]
    test_cold_u = user_emb_ids["test_cold"]
    test_union = test_warm_u | test_cold_u
    # Sessions test
    print(f"  sessions_test ∩ user_emb_test_warm: {len(sessions_test_users & test_warm_u)}")
    print(f"  sessions_test ∩ user_emb_test_cold: {len(sessions_test_users & test_cold_u)}")
    print(f"  sessions_test \\ (warm∪cold): {len(sessions_test_users - test_union)}")
    # Blind-A
    print(f"  blind_A ∩ user_emb_test_warm: {len(blind_users & test_warm_u)}")
    print(f"  blind_A ∩ user_emb_test_cold: {len(blind_users & test_cold_u)}")
    print(f"  blind_A ∩ sessions_test: {len(blind_users & sessions_test_users)}")

    # Test cold definition check: should have NO overlap with user_emb train
    print(f"  user_emb_test_cold ∩ user_emb_train: {len(test_cold_u & user_emb_ids['train'])}")
    print(f"  user_emb_test_warm ∩ user_emb_train: {len(test_warm_u & user_emb_ids['train'])}")

    # Tracks
    print(f"  track-meta all_tracks: {len(tracks_all)}")
    print(f"  track-meta test_tracks: {len(tracks_test)}")
    print(f"  track-emb all_tracks:  {len(track_emb_all)}")
    print(f"  track-emb test_tracks: {len(track_emb_test)}")
    print(f"  track-meta vs track-emb (all) symmetric diff: {len(tracks_all.symmetric_difference(track_emb_all))}")
    print(f"  track-meta vs track-emb (test) symmetric diff: {len(tracks_test.symmetric_difference(track_emb_test))}")
    print(f"  test_tracks ⊆ all_tracks (meta): {tracks_test.issubset(tracks_all)}")


def main() -> None:
    # Sessions main dataset (train + test)
    sessions_path = DATASETS["sessions"]
    explore_sessions(sessions_path, "train")
    # Capture user_ids for cross checks
    ds_train = load_from_disk(str(sessions_path))["train"]
    ds_test = load_from_disk(str(sessions_path))["test"]
    sessions_train_users = set(ds_train["user_id"])
    sessions_test_users = set(ds_test["user_id"])
    explore_sessions(sessions_path, "test")

    # Blind-A
    explore_sessions(DATASETS["blind_a"], "test")
    blind_users = set(load_from_disk(str(DATASETS["blind_a"]))["test"]["user_id"])

    # Tracks
    tracks_all, tracks_test = explore_track_meta(DATASETS["track_meta"])
    track_emb_all, track_emb_test = explore_track_embeddings(DATASETS["track_emb"])

    # Users
    user_meta_ids = explore_user_meta(DATASETS["user_meta"])
    user_emb_ids = explore_user_embeddings(DATASETS["user_emb"])

    # Cross-dataset
    cross_dataset(sessions_train_users, sessions_test_users, blind_users,
                  user_meta_ids, user_emb_ids,
                  tracks_all, tracks_test, track_emb_all, track_emb_test)


if __name__ == "__main__":
    main()
