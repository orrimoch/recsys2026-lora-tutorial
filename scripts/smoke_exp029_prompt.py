"""Smoke-test exp 029 CoT user-state prompt on 3 train rows.

Tiny local probe (≤40-row M4 rule) to verify, before queuing the full
Colab dev-set run, that:

  1. The CRS pipeline loads the new prompt without errors.
  2. Qwen 1.5B actually emits the <user_state>...</user_state> +
     <response>...</response> structure under the new prompt.
  3. crs_baseline.extract_cot_response strips the user_state envelope
     so predicted_response (what Gemini sees) is the response prose only.

Skips full inference machinery — calls the LM directly with one pre-formed
session memory + recommend_item per row.

Usage:
    source recsys26/bin/activate
    python scripts/smoke_exp029_prompt.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINES_DIR = REPO_ROOT / "music-crs-baselines"
sys.path.insert(0, str(BASELINES_DIR))

import torch  # noqa: E402
from mcrs.lm_modules import load_lm_module  # noqa: E402
from mcrs.crs_baseline import extract_cot_response  # noqa: E402


PROMPT_PATH = BASELINES_DIR / "mcrs/system_prompts/response_generation_cot_user_state.txt"
ROLEPLAY_PATH = BASELINES_DIR / "mcrs/system_prompts/roleplay.txt"

SAMPLES = [
    {
        "session_memory": [
            {"role": "user", "content": "I want something melancholic and acoustic, like Bon Iver or Sufjan Stevens."},
        ],
        "recommend_item": "track_name: Mystery of Love\nartist_name: Sufjan Stevens\nalbum_name: Call Me By Your Name (Original Motion Picture Soundtrack)",
    },
    {
        "session_memory": [
            {"role": "user", "content": "Need high-energy workout music, electronic, no lyrics."},
            {"role": "assistant", "content": "track_name: Strobe\nartist_name: deadmau5\nalbum_name: For Lack of a Better Name"},
            {"role": "user", "content": "Something faster, more aggressive."},
        ],
        "recommend_item": "track_name: Ghosts 'n' Stuff\nartist_name: deadmau5\nalbum_name: For Lack of a Better Name",
    },
    {
        "session_memory": [
            {"role": "user", "content": "Play me some 90s hip-hop, classic east coast vibes."},
        ],
        "recommend_item": "track_name: C.R.E.A.M.\nartist_name: Wu-Tang Clan\nalbum_name: Enter the Wu-Tang (36 Chambers)",
    },
]


def main() -> int:
    sys_prompt = ROLEPLAY_PATH.read_text(encoding="utf-8") + PROMPT_PATH.read_text(encoding="utf-8")
    print(f"[smoke] system prompt loaded ({len(sys_prompt)} chars)")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[smoke] loading Qwen 2.5-1.5B-Instruct on {device}…")
    lm = load_lm_module(
        "Qwen/Qwen2.5-1.5B-Instruct",
        device=device,
        attn_implementation="eager",
        dtype=torch.bfloat16,
    )
    print("[smoke] LM loaded")

    sys_prompts = [sys_prompt] * len(SAMPLES)
    histories = [s["session_memory"] for s in SAMPLES]
    rec_items = [s["recommend_item"] for s in SAMPLES]

    raw = lm.batch_response_generation(sys_prompts, histories, rec_items, max_new_tokens=192)
    print(f"[smoke] generated {len(raw)} responses\n")

    has_response_tag = 0
    has_user_state_tag = 0
    parser_clean = 0
    for i, (sample, r) in enumerate(zip(SAMPLES, raw)):
        print("=" * 78)
        print(f"[{i}] query: {sample['session_memory'][-1]['content']!r}")
        print(f"[{i}] recommended: {sample['recommend_item'].splitlines()[0]}")
        print(f"\n--- RAW LM OUTPUT ({len(r)} chars) ---")
        print(r)
        if "<response>" in r.lower():
            has_response_tag += 1
        if "<user_state>" in r.lower():
            has_user_state_tag += 1
        parsed = extract_cot_response(r)
        print(f"\n--- AFTER extract_cot_response ({len(parsed)} chars) ---")
        print(parsed)
        leak = ("<user_state>" in parsed.lower() or "mood:" in parsed.lower()
                or "session_intent:" in parsed.lower())
        if not leak and parsed.strip():
            parser_clean += 1
        print(f"\nleak detected: {leak}")
        print()

    n = len(SAMPLES)
    print("=" * 78)
    print(f"SUMMARY ({n} samples)")
    print(f"  raw contains <response> tag : {has_response_tag}/{n}")
    print(f"  raw contains <user_state>   : {has_user_state_tag}/{n}")
    print(f"  parser output clean         : {parser_clean}/{n}")
    print()
    print("PASS criteria: parser output clean ≥ 2/3 AND no exceptions raised.")
    return 0 if parser_clean >= max(1, n - 1) else 1


if __name__ == "__main__":
    sys.exit(main())
