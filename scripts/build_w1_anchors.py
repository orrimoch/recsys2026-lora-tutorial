"""Build the W1 reward-correlation anchor dataset.

Three sources of anchors (per RecSys_Challenge_Plan §6.2):

  Source A — train-data GPA rows (~20k):
      For every train turn with a goal_progress_assessment label, emit one row.
      response = gold assistant response.
      predicted_track_ids = [gold_track_id]  → R_retr ≡ 1.0 by construction
      judge_anchor       = 5 if MOVES_TOWARD_GOAL else 1
      → Tests whether mechanical response-side reward (R_rule + R_user_prof)
        predicts the GPA judge label.

  Source B — cached devset rollouts (~80k):
      For every cached prediction in music-crs-baselines/exp/inference/devset/,
      join with music-crs-evaluator/exp/ground_truth/devset.json.
      response = predicted_response (varies)
      predicted_track_ids = experiment's top-20 (varies)
      judge_anchor = NaN (unavailable)
      → R_retr correlation sanity (must mirror nDCG by construction);
        also produces R_rule on real-experiment responses so we can spot
        distribution shifts across experiments.

  Source C — 8 aggregate Blind-A submissions (no per-turn predictions on disk):
      Pulled from the historical submissions_log table (commit a81696b).
      Tuple: (exp_id, nDCG@20, CatDiv, LexDiv, Gemini_judge, composite_blind).
      → Sanity-verifies our composite_ref formula reproduces the leaderboard
        composite from (nDCG, CatDiv, LexDiv, LLM) inputs.

Output:
    data/reward_calibration_anchors.parquet   (rows: A + B)
    data/reward_calibration_anchors_sourceC.json  (the 8 Blind-A aggregate rows)

Then `scripts/run_w1_gate.py` consumes these and computes the Spearman
correlations + bootstrap CI for the W1 hard gate.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_DIR = REPO_ROOT / "music-crs-evaluator"
BASELINES_DIR = REPO_ROOT / "music-crs-baselines"
DATA_DIR = REPO_ROOT / "data"

# Reuse our shared module so reward terms stay in one place.
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
from reward_fns import dump_json  # noqa: E402

LABEL_POS = "MOVES_TOWARD_GOAL"
LABEL_NEG = "DOES_NOT_MOVE_TOWARD_GOAL"
MAX_HISTORY_TURNS = 4
MAX_RESPONSE_CHARS = 1200


def _first(v):
    if isinstance(v, list):
        return v[0] if v else None
    return v


# ---------------------------------------------------------------------------
# Lightweight loaders (no mcrs imports — keeps Colab footprint small)
# ---------------------------------------------------------------------------

def load_track_meta() -> dict[str, dict]:
    print("[anchors] loading track metadata")
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
    concat = concatenate_datasets([ds[s] for s in ["all_tracks"]])
    return {
        item["track_id"]: {
            "track_name": _first(item.get("track_name")) or "",
            "artist_name": _first(item.get("artist_name")) or "",
            "album_name": _first(item.get("album_name")) or "",
            "release_date": item.get("release_date"),
            "tag_list": item.get("tag_list") or [],
        }
        for item in concat
    }


def load_user_profiles() -> dict[str, dict]:
    print("[anchors] loading user metadata")
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-User-Metadata")
    concat = concatenate_datasets([ds[s] for s in ["all_users"]])
    return {
        item["user_id"]: {
            "age_group": item.get("age_group") or "",
            "gender": item.get("gender") or "",
            "country_name": item.get("country_name") or "",
        }
        for item in concat
    }


def history_text(df: pd.DataFrame, turn_n: int, track_meta: dict[str, dict]) -> str:
    """Concatenate prior turns (capped at MAX_HISTORY_TURNS turns)."""
    prior = df[df["turn_number"] < turn_n]
    if prior.empty:
        return ""
    prior = prior.tail(MAX_HISTORY_TURNS * 3)  # 3 roles per turn
    lines = []
    for _, t in prior.iterrows():
        role = "assistant" if t["role"] == "music" else t["role"]
        content = t["content"]
        if t["role"] == "music":
            tm = track_meta.get(content, {})
            content = f'{tm.get("track_name", "")} by {tm.get("artist_name", "")}'.strip()
        content_str = str(content)[:200]
        lines.append(f"{role}: {content_str}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Source A — train GPA rows
# ---------------------------------------------------------------------------

def build_source_a(
    n_sessions: int,
    seed: int,
    track_meta: dict,
    user_profiles: dict,
    pad_to_topk: int = 20,
) -> list[dict]:
    """Build Source A. predicted_track_ids = [gold] padded to `pad_to_topk` (Gap 9).

    Padding rationale: the original Source A produced 1-element lists which
    made downstream CatDiv / top-k metrics meaningless and inconsistent with
    Source B / production (always 20). We pad with random distinct catalog
    track_ids (seeded for reproducibility); R_retr stays 1.0 because gold is
    at rank 1, but any consumer that aggregates across rows can now compute
    meaningful diversity stats.
    """
    print(f"[anchors] Source A: loading train split (n_sessions={n_sessions})")
    tr = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="train")
    rng = random.Random(seed)
    if n_sessions < len(tr):
        idx = rng.sample(range(len(tr)), n_sessions)
        sessions = tr.select(idx).to_list()
    else:
        sessions = tr.to_list()
    print(f"[anchors] Source A: using {len(sessions)} sessions")

    # Catalog track ids — used to pad predicted_track_ids to `pad_to_topk`.
    catalog_ids = list(track_meta.keys()) if track_meta else []
    if not catalog_ids:
        # Defensive: track_meta missing → can't pad; downstream gracefully sees 1-element lists.
        pad_to_topk = 1

    rows = []
    skipped_none = 0
    skipped_missing = 0
    for sess in tqdm(sessions, desc="Source A: train sessions"):
        convos = sess.get("conversations") or []
        if not convos:
            continue
        df = pd.DataFrame(convos)
        gpa_map = {
            int(x["turn_number"]): x.get("goal_progress_assessment")
            for x in (sess.get("goal_progress_assessments") or [])
        }
        user_id = sess["user_id"]
        profile = user_profiles.get(user_id, {})
        goal = sess.get("conversation_goal") or {}
        goal_listener = (goal.get("listener_goal") or "").strip()

        for turn_n in sorted(df["turn_number"].unique()):
            tn = int(turn_n)
            label_raw = gpa_map.get(tn)
            if label_raw == LABEL_POS:
                judge_anchor = 5
            elif label_raw == LABEL_NEG:
                judge_anchor = 1
            else:
                skipped_none += 1
                continue

            tdf = df[df["turn_number"] == tn]
            user_row = tdf[tdf["role"] == "user"]
            music_row = tdf[tdf["role"] == "music"]
            asst_row = tdf[tdf["role"] == "assistant"]
            if user_row.empty or music_row.empty or asst_row.empty:
                skipped_missing += 1
                continue

            user_query = str(user_row.iloc[0]["content"])[:500]
            gold_track_id = str(music_row.iloc[0]["content"])
            gold_response = str(asst_row.iloc[0]["content"])[:MAX_RESPONSE_CHARS]
            t1m = track_meta.get(gold_track_id, {"track_name": "", "artist_name": ""})
            hist = history_text(df, tn, track_meta)

            # Synthetic prediction: gold-at-rank-1 → R_retr will be 1.0.
            # That's the point — we hold R_retr constant so the correlation
            # test on Source A isolates the response-side mechanical reward.
            # Padding (Gap 9): top-1 = gold, top-(2..K) = random distinct
            # catalog ids drawn from a per-row RNG (seeded by gold_track_id
            # so pad order is reproducible). Stays distinct from gold.
            if pad_to_topk > 1 and catalog_ids:
                pad_rng = random.Random(f"{seed}::{gold_track_id}")
                # Sample one extra in case the first random pick collides with gold.
                sample = pad_rng.sample(catalog_ids, k=min(pad_to_topk, len(catalog_ids)))
                # Drop gold if it appears in the sample, then take pad_to_topk - 1.
                pad_ids = [t for t in sample if t != gold_track_id][: pad_to_topk - 1]
                # If we somehow didn't get enough, top up by walking forward.
                if len(pad_ids) < pad_to_topk - 1:
                    extras = [t for t in catalog_ids if t != gold_track_id and t not in pad_ids]
                    pad_ids.extend(extras[: (pad_to_topk - 1 - len(pad_ids))])
                predicted_track_ids = [gold_track_id] + pad_ids
            else:
                predicted_track_ids = [gold_track_id]

            rows.append({
                "source": "A",
                "exp_id": "train_gpa",
                "session_id": sess["session_id"],
                "user_id": user_id,
                "turn_number": tn,
                "user_query": user_query,
                "gold_track_id": gold_track_id,
                "predicted_track_ids": predicted_track_ids,
                "predicted_response": gold_response,
                "track_name": t1m.get("track_name", ""),
                "artist_name": t1m.get("artist_name", ""),
                "country_name": profile.get("country_name", ""),
                "age_group": profile.get("age_group", ""),
                "gender": profile.get("gender", ""),
                "history_text": hist,
                "goal_listener": goal_listener[:300],
                "judge_anchor": judge_anchor,
            })

    print(f"[anchors] Source A: {len(rows)} rows  "
          f"(skipped {skipped_none} no-label, {skipped_missing} missing-roles)")
    return rows


# ---------------------------------------------------------------------------
# Source B — cached devset rollouts
# ---------------------------------------------------------------------------

def build_source_b(track_meta: dict, user_profiles: dict) -> list[dict]:
    devset_dir = BASELINES_DIR / "exp" / "inference" / "devset"
    gt_path = EVALUATOR_DIR / "exp" / "ground_truth" / "devset.json"
    if not gt_path.exists():
        print(f"[anchors] Source B: GT not found at {gt_path}, skipping")
        return []
    with gt_path.open("r", encoding="utf-8") as f:
        gt_list = json.load(f)
    gt_map = {(r["session_id"], int(r["turn_number"])): r["ground_truth_track_id"]
              for r in gt_list}

    pred_files = sorted(devset_dir.glob("*.json"))
    print(f"[anchors] Source B: {len(pred_files)} cached devset prediction files")

    # Load dev session metadata (user_id) so we can look up profile.
    print("[anchors] Source B: loading dev session metadata for user_id")
    dev = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="test")
    sid_to_uid = {s["session_id"]: s["user_id"] for s in dev}

    rows = []
    for pf in pred_files:
        exp_id = pf.stem
        try:
            with pf.open("r", encoding="utf-8") as f:
                preds = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[anchors] Source B: skipping {exp_id} ({e})")
            continue
        if not isinstance(preds, list) or not preds:
            continue

        for r in preds:
            sid = r.get("session_id")
            tn = int(r.get("turn_number", -1))
            tids = r.get("predicted_track_ids") or []
            resp = r.get("predicted_response") or ""
            gold = gt_map.get((sid, tn))
            if gold is None or not tids:
                continue
            user_id = sid_to_uid.get(sid, "")
            profile = user_profiles.get(user_id, {})
            top1 = tids[0] if tids else None
            t1m = track_meta.get(top1, {"track_name": "", "artist_name": ""}) if top1 else {}
            rows.append({
                "source": "B",
                "exp_id": exp_id,
                "session_id": sid,
                "user_id": user_id,
                "turn_number": tn,
                "user_query": "",  # not stored in cached prediction; OK for R_rule history field
                "gold_track_id": gold,
                "predicted_track_ids": tids,
                "predicted_response": resp,
                "track_name": t1m.get("track_name", ""),
                "artist_name": t1m.get("artist_name", ""),
                "country_name": profile.get("country_name", ""),
                "age_group": profile.get("age_group", ""),
                "gender": profile.get("gender", ""),
                "history_text": "",
                "goal_listener": "",
                "judge_anchor": None,  # unknown for devset cached rollouts
            })
    print(f"[anchors] Source B: {len(rows)} rows")
    return rows


# ---------------------------------------------------------------------------
# Source D — synthetic perturbations of Source A responses (Gap 5)
#
# Why this exists:
#   Source A by construction has R_retr ≡ 1.0 and R_format ≡ 0 — so the W1 G1
#   gate's variance is driven entirely by R_rule, the same as G2. The two
#   gates collapse into one, weakening the signal. Source D varies R_rule
#   meaningfully WHILE keeping the GPA label fixed, producing a fairer test of
#   whether R_rule predicts well-formedness vs the GPA-derived judge.
#
# Perturbations:
#   POS rows  (judge_anchor=5): emit a "corrupted" sibling with low R_rule.
#                               Variants:
#                                 - drop_track_name : strip the gold track name from response
#                                 - inject_banned   : append "absolutely fantastic, perfectly amazing"
#                                 - truncate_5      : keep only the first 5 words
#                                 - drop_why        : strip the music-detail vocabulary
#   NEG rows  (judge_anchor=1): emit a "rescued" sibling with high R_rule.
#                               Variants:
#                                 - replace_banned  : replace banned phrases with neutral synonyms
#                                 - extend_neutral  : append a clean why-clause + track-name mention
#
# Each Source A row produces ONE Source D variant (round-robin across types
# so we don't tilt the distribution). judge_anchor copies the parent's label.
# ---------------------------------------------------------------------------

PERTURB_BANNED_INJECTION = " That song is absolutely fantastic, perfectly amazing, sorry I can't say more."
PERTURB_NEUTRAL_REPLACEMENTS = [
    (re.compile(r"\babsolutely\b", re.I), "really"),
    (re.compile(r"\bfantastic\b", re.I), "great"),
    (re.compile(r"\bperfectly\b", re.I), "well"),
    (re.compile(r"\bamazing\b", re.I), "good"),
    (re.compile(r"\bsorry\b", re.I), "consider"),
    (re.compile(r"\bunfortunately\b", re.I), "however"),
    (re.compile(r"\bi (cannot|can't)\b", re.I), "you might"),
]
PERTURB_WHY_TOKENS = re.compile(
    r"\b(because|since|features|leans|driven by|atmosphere|tempo|groove|"
    r"arrangement|released|from \d{4}|era|decade|vibe|texture|timbre|harmony|melody|rhythm)\b",
    re.I,
)


def _perturb_drop_track_name(resp: str, track_name: str, artist_name: str) -> str:
    out = resp
    if track_name:
        out = re.sub(re.escape(track_name), "this song", out, flags=re.IGNORECASE)
    if artist_name:
        out = re.sub(re.escape(artist_name), "the artist", out, flags=re.IGNORECASE)
    return out


def _perturb_inject_banned(resp: str) -> str:
    return resp.rstrip() + PERTURB_BANNED_INJECTION


def _perturb_truncate_5(resp: str) -> str:
    return " ".join(resp.split()[:5])


def _perturb_drop_why(resp: str) -> str:
    return PERTURB_WHY_TOKENS.sub("[X]", resp)


def _perturb_replace_banned(resp: str) -> str:
    out = resp
    for rx, repl in PERTURB_NEUTRAL_REPLACEMENTS:
        out = rx.sub(repl, out)
    return out


def _perturb_extend_neutral(resp: str, track_name: str, artist_name: str) -> str:
    """Tack on a clean why-clause + track-name mention (raises R_rule)."""
    extra_bits = []
    if track_name:
        extra_bits.append(f"\"{track_name}\"")
    if artist_name:
        extra_bits.append(f"by {artist_name}")
    why = " features a layered arrangement and reflective tempo that fits the moment."
    if extra_bits:
        return resp.rstrip() + " I'd lean " + " ".join(extra_bits) + " — it" + why
    return resp.rstrip() + " The track" + why


POS_VARIANTS = ["drop_track_name", "inject_banned", "truncate_5", "drop_why"]
NEG_VARIANTS = ["replace_banned", "extend_neutral"]


def build_source_d(rows_a: list[dict], seed: int = 42) -> list[dict]:
    """Generate one Source D row per Source A row, round-robin across variants.

    Source D rows have `source="D"` and the same judge_anchor as the parent.
    """
    if not rows_a:
        return []
    print(f"[anchors] Source D: deriving from {len(rows_a)} Source A rows")
    out: list[dict] = []
    pos_i = 0
    neg_i = 0
    for r in rows_a:
        ja = r.get("judge_anchor")
        resp = r.get("predicted_response") or ""
        track_name = r.get("track_name") or ""
        artist_name = r.get("artist_name") or ""
        if ja == 5:
            variant = POS_VARIANTS[pos_i % len(POS_VARIANTS)]
            pos_i += 1
            if variant == "drop_track_name":
                new_resp = _perturb_drop_track_name(resp, track_name, artist_name)
            elif variant == "inject_banned":
                new_resp = _perturb_inject_banned(resp)
            elif variant == "truncate_5":
                new_resp = _perturb_truncate_5(resp)
            else:  # drop_why
                new_resp = _perturb_drop_why(resp)
        elif ja == 1:
            variant = NEG_VARIANTS[neg_i % len(NEG_VARIANTS)]
            neg_i += 1
            if variant == "replace_banned":
                new_resp = _perturb_replace_banned(resp)
            else:  # extend_neutral
                new_resp = _perturb_extend_neutral(resp, track_name, artist_name)
        else:
            continue  # no label — skip

        d = dict(r)  # copy parent
        d["source"] = "D"
        d["exp_id"] = f"train_gpa_perturbed::{variant}"
        d["predicted_response"] = new_resp
        d["perturb_variant"] = variant
        out.append(d)
    print(f"[anchors] Source D: {len(out)} perturbed rows  "
          f"(pos_variants={pos_i}, neg_variants={neg_i})")
    return out


# ---------------------------------------------------------------------------
# Source C — historical Blind-A aggregate scores (commit a81696b)
# ---------------------------------------------------------------------------

# Recovered from `git show a81696b:documents/submissions_log.md`.
# Aggregate per-submission stats only (per-turn predictions never committed).
SOURCE_C_BLIND_A = [
    # exp_id, nDCG@20, CatDiv, LexDiv, LLM_judge (1-5), composite_reported
    ("021-two-step-wrrf-lyrics-qwen15b",       0.19, 0.03, 0.67, 3.15, 0.33),
    ("022-persona-qwen15b",                    0.14, 0.03, 0.80, 2.15, 0.24),
    ("023-rerank-qwen3b",                      0.07, 0.03, 0.61, 2.20, 0.19),
    ("024-qwen3b-longresp",                    0.14, 0.03, 0.62, 3.00, 0.29),
    ("026-reward-rerank-qwen15b",              0.14, 0.03, 0.78, 2.60, 0.27),
    ("027-wrrf-lgbm-qwen15b",                  0.11, 0.03, 0.77, 2.20, 0.23),
    ("028-top3-wrrf-qwen15b",                  0.14, 0.03, 0.77, 2.10, 0.23),
    ("029-cot-user-state-qwen15b",             0.14, 0.03, 0.80, 2.25, 0.25),
]


def build_source_c() -> list[dict]:
    print(f"[anchors] Source C: {len(SOURCE_C_BLIND_A)} historical Blind-A aggregates")
    rows = []
    for exp_id, ndcg20, catdiv, lexdiv, judge, composite in SOURCE_C_BLIND_A:
        composite_ref = (
            0.50 * ndcg20
            + 0.10 * catdiv
            + 0.10 * lexdiv
            + 0.30 * (judge - 1) / 4
        )
        rows.append({
            "exp_id": exp_id,
            "ndcg_at_20": ndcg20,
            "cat_div": catdiv,
            "lex_div": lexdiv,
            "llm_judge": judge,
            "composite_reported": composite,
            "composite_ref": round(composite_ref, 4),
            "delta_vs_reported": round(composite_ref - composite, 4),
        })
    return rows


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-sessions", type=int, default=15000,
                   help="Cap train sessions for Source A (default: 15000 = all available).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default=str(DATA_DIR / "reward_calibration_anchors.parquet"))
    p.add_argument("--out-source-c", type=str,
                   default=str(DATA_DIR / "reward_calibration_anchors_sourceC.json"))
    p.add_argument("--source", choices=["A", "B", "C", "D", "ABC", "ABCD", "AD"], default="ABC",
                   help="Which sources to build. Source D = perturbed Source A "
                        "(Gap 5 — breaks R_rule near-zero variance).")
    args = p.parse_args(argv)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Source C is cheap; do it first.
    if "C" in args.source:
        c_rows = build_source_c()
        dump_json({"rows": c_rows, "note": "From git a81696b documents/submissions_log.md"},
                  args.out_source_c)
        print(f"[anchors] wrote Source C → {args.out_source_c}")

    # A and B both need the metadata + profiles.
    if "A" in args.source or "B" in args.source:
        track_meta = load_track_meta()
        user_profiles = load_user_profiles()
    else:
        track_meta = {}
        user_profiles = {}

    rows: list[dict] = []
    rows_a_only: list[dict] = []
    if "A" in args.source:
        rows_a_only = build_source_a(args.n_sessions, args.seed, track_meta, user_profiles)
        rows.extend(rows_a_only)
    if "B" in args.source:
        rows.extend(build_source_b(track_meta, user_profiles))
    if "D" in args.source:
        # Source D needs Source A as input — build it on-the-fly if not in `rows_a_only`.
        if not rows_a_only:
            print("[anchors] Source D requested but Source A not in --source; building A internally.")
            rows_a_only = build_source_a(args.n_sessions, args.seed, track_meta, user_profiles)
        rows.extend(build_source_d(rows_a_only, seed=args.seed))

    if rows:
        df = pd.DataFrame(rows)
        df.to_parquet(args.out, index=False)
        size_mb = os.path.getsize(args.out) / 1e6
        print(f"[anchors] wrote {len(df)} rows → {args.out} ({size_mb:.1f} MB)")
        # Source breakdown
        for src, n in df["source"].value_counts().items():
            print(f"            source={src}  n={n}")
        if "judge_anchor" in df.columns:
            ja = df.dropna(subset=["judge_anchor"])
            if len(ja):
                pos = int((ja["judge_anchor"] >= 4).sum())
                neg = int((ja["judge_anchor"] <= 2).sum())
                print(f"            judge anchored rows: {len(ja)}  (pos={pos}, neg={neg})")
    else:
        print("[anchors] no Source A/B rows to write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
