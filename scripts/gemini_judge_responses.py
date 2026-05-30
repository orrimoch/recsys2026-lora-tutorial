"""Phase 0 responder eval: score predicted_response with the REAL Gemini API
(the same judge family the Blind-A leaderboard uses) on Personalization +
Explanation Quality. Lets us A/B responder changes offline without burning
Blind-A submissions.

Input : exp/inference/devset/{tid}.json (from run_inference_devset.py) — list of
        {session_id, user_id, turn_number, predicted_track_ids, predicted_response}.
        We re-join the dev 'test' split to give the judge the conversation context
        + the recommended track metadata (what a personalized/grounded reply
        should reference).
Output: exp/judge/{tid}_gemini.json with per-row scores + a printed mean.

Key/rubric:
- GEMINI_API_KEY read from env (set it from Colab secrets before running).
- The rubric below is a BEST-EFFORT reconstruction of the challenge's
  Personalization + Explanation Quality axes (0-5 each). RECONCILE against the
  official rubric when available (the judge prompt is the single place to edit).
  See project_blind_judge_gemini + project_responder_workstream_plan memories.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import pandas as pd
from datasets import load_dataset

MODEL = os.environ.get("GEMINI_JUDGE_MODEL", "gemini-1.5-flash")
DEV_DATASET = "talkpl-ai/TalkPlayData-Challenge-Dataset"
ITEM_DB = "talkpl-ai/TalkPlayData-Challenge-Track-Metadata"

JUDGE_INSTRUCTIONS = """You are evaluating a music recommender's reply to a user in a conversation.
Score TWO axes from 0 to 5 (integers or one decimal).

PERSONALIZATION (0-5): Does the reply reflect THIS user's stated intent, mood,
and taste from the conversation? 5 = clearly tailored to the user's request and
context; 0 = generic, could be sent to anyone.

EXPLANATION_QUALITY (0-5): Does the reply give a concrete, accurate reason for
the recommendation, citing real attributes of the recommended track (artist,
title, genre, mood, instrumentation, era)? 5 = specific and grounded; 0 = vague,
empty, or hallucinated.

Return ONLY a JSON object, no prose:
{"personalization": <0-5>, "explanation_quality": <0-5>}"""


def build_context(conversations, item_db_meta, target_turn):
    """User-visible conversation up to (and including) the target user turn."""
    df = pd.DataFrame(conversations)
    hist = df[df["turn_number"] < target_turn]
    lines = []
    for _, t in hist.iterrows():
        role = t["role"]
        content = t["content"]
        if role == "music":
            role = "assistant"
            m = item_db_meta.get(str(content), {})
            content = f"[recommended: {m.get('track_name')} by {m.get('artist_name')}]"
        lines.append(f"{role}: {content}")
    cur = df[(df["turn_number"] == target_turn) & (df["role"] == "user")]
    if len(cur):
        lines.append(f"user: {cur.iloc[0]['content']}")
    return "\n".join(lines)


def track_meta_str(tids, item_db_meta, n=3):
    out = []
    for tid in (tids or [])[:n]:
        m = item_db_meta.get(str(tid), {})
        out.append(f"{m.get('track_name')} by {m.get('artist_name')}"
                   f" (album {m.get('album_name')})")
    return "; ".join(out) if out else "(none)"


def parse_score(text):
    m = re.search(r"\{[^}]*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        return (float(d["personalization"]), float(d["explanation_quality"]))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tid", required=True, help="config tid; reads exp/inference/devset/{tid}.json")
    ap.add_argument("--limit", type=int, default=400, help="max rows to judge (cost control)")
    ap.add_argument("--sleep", type=float, default=0.5, help="seconds between API calls (rate limit)")
    args = ap.parse_args()

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        sys.exit("ERROR: set GEMINI_API_KEY (Colab secret) before running.")
    import google.generativeai as genai
    genai.configure(api_key=key)
    model = genai.GenerativeModel(MODEL)

    pred_path = f"exp/inference/devset/{args.tid}.json"
    preds = json.load(open(pred_path))
    preds = preds[: args.limit]
    print(f"[judge] {len(preds)} rows from {pred_path} via {MODEL}")

    # dev context + track metadata lookups
    dev = load_dataset(DEV_DATASET, split="test")
    sess_by_id = {s["session_id"]: s for s in dev}
    idb = load_dataset(ITEM_DB)
    import datasets as _d
    idb = _d.concatenate_datasets([idb[s] for s in idb])
    item_db_meta = {}
    for r in idb:
        a = r.get("artist_name"); a = (a[0] if isinstance(a, list) and a else a)
        item_db_meta[str(r["track_id"])] = {
            "track_name": (r.get("track_name") or [None])[0] if isinstance(r.get("track_name"), list) else r.get("track_name"),
            "artist_name": a,
            "album_name": (r.get("album_name") or [None])[0] if isinstance(r.get("album_name"), list) else r.get("album_name"),
        }

    rows, p_sum, e_sum, n_ok = [], 0.0, 0.0, 0
    for i, p in enumerate(preds):
        sess = sess_by_id.get(p["session_id"])
        if sess is None:
            continue
        ctx = build_context(sess["conversations"], item_db_meta, p["turn_number"])
        recs = track_meta_str(p.get("predicted_track_ids"), item_db_meta)
        prompt = (f"{JUDGE_INSTRUCTIONS}\n\n=== CONVERSATION ===\n{ctx}\n\n"
                  f"=== RECOMMENDED TRACKS ===\n{recs}\n\n"
                  f"=== ASSISTANT REPLY TO JUDGE ===\n{p['predicted_response']}\n\n"
                  f"Return only the JSON.")
        try:
            resp = model.generate_content(prompt)
            sc = parse_score(resp.text)
        except Exception as e:
            print(f"  row {i}: API error {e!r}"); sc = None
        if sc:
            pp, ee = sc
            p_sum += pp; e_sum += ee; n_ok += 1
            rows.append({**{k: p[k] for k in ("session_id", "turn_number")},
                         "personalization": pp, "explanation_quality": ee})
        if args.sleep:
            time.sleep(args.sleep)
        if (i + 1) % 25 == 0:
            print(f"  judged {i+1}/{len(preds)} (ok={n_ok})")

    out_path = f"exp/judge/{args.tid}_gemini.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(rows, open(out_path, "w"), ensure_ascii=False, indent=1)
    if n_ok:
        pm, em = p_sum / n_ok, e_sum / n_ok
        print(f"\n=== Gemini judge: {args.tid} (n={n_ok}) ===")
        print(f"  personalization     : {pm:.3f} / 5")
        print(f"  explanation_quality : {em:.3f} / 5")
        print(f"  mean (LLM-axis proxy): {(pm+em)/2:.3f} / 5")
        print(f"  wrote {out_path}")
    else:
        print("[judge] no rows scored — check API key / model / rate limits")


if __name__ == "__main__":
    main()
