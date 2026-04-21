"""Build the LoRA fine-tuning dataset for Qwen2.5-3B on Blind-A task.

Extracts turn-1 triples from `talkpl-ai/TalkPlayData-Challenge-Dataset` (train
split), formats them to EXACTLY match the v10-champion inference prompt shape
(system = roleplay + response_generation_v4 + "About this user" + "Candidate
tracks" with the gold track at slot 1), and saves a HF-datasets object to disk
ready for `train_lora.py`.

Design decisions (each grounded in a prior experiment's finding):

* Turn-1 only. Blind-A is 80 single-turn rows; multi-turn train responses
  follow "Awesome! Glad you liked X" patterns that inject conversational
  priors into the fine-tuned model. The `_strip_warm_opener` regex strips the
  remaining turn-1 opener so the response body is a cold-start-style
  recommendation. (v17 lesson: NN few-shot with opener-containing responses
  drove Blind-A outputs to 50-word chat-bot replies.)
* Filter out responses shorter than 40 words. v10 champion mean is 81 words;
  training on short responses compresses generation length.
* Filter out responses containing banned AI-speak words: "absolutely",
  "fantastic", "truly", "amazing". v10's LLM-judge +0.40 lift came from
  banning these; training on contaminated responses would undo that gain.
* Gold track metadata goes into slot 1 of the "Candidate tracks" block in
  the SYSTEM prompt. At inference, slot 1 is the BM25 top-1. The model thus
  learns: given any track metadata in slot 1, cite it in critic tone.
* Demographics + conversation_goal injected the same way as inference.

CLI:
    python build_train_dataset.py --output ./data/train_sft
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Any

from datasets import Dataset, load_dataset


BANNED_WORDS = re.compile(r"(?i)\b(absolutely|fantastic|truly|amazing)\b")
# Apologies / failure admissions — v10's prompt explicitly bans these
# ("Never apologize. Never say 'I'm sorry'..."). Lesson from Exp 002→003:
# stock prompt's apology directive cost ~+0.9 LLM-judge once removed.
APOLOGY_PATTERNS = re.compile(
    r"(?i)\b("
    r"i[\u2019']?m\s+(really\s+|so\s+|terribly\s+)?sorry|"
    r"i\s+apologi[sz]e|apologies\b|my\s+apologies|"
    r"unfortunately|"
    r"i\s+don[\u2019']?t\s+have\s+(more|any|the|other|a)|"
    r"i\s+(can[\u2019']?t|cannot)\s+(find|locate|see|provide)|"
    r"i\s+couldn[\u2019']?t\s+find|"
    r"no\s+(luck|results|matches|tracks|songs)\s+(found|in|with)|"
    r"unable\s+to\s+(find|provide|locate|recommend)|"
    r"i\s+wasn[\u2019']?t\s+able"
    r")\b"
)
# Hallucinated user-already-heard patterns. v10's prompt rule 5: "Never claim
# the user has already heard or enjoyed these tracks." v17 confirmed these
# leak in via train data and tank LLM-judge.
HALLUCINATION_PATTERNS = re.compile(
    r"(?i)\b("
    r"glad\s+you\s+(liked|enjoyed|loved)|"
    r"happy\s+you\s+(liked|enjoyed|loved)|"
    r"since\s+you\s+(liked|enjoyed|loved)|"
    r"you[\u2019']?re\s+(loving|enjoying|digging)\s+(this|that|the)"
    r")\b"
)
# Sentence-aware warm-opener detection: any sentence containing "glad" or
# starting with conversational-feedback phrasing is a turn-N follow-up that
# acknowledges what the user said about a previous track. Strip iteratively
# until the next sentence is real critique content.
_FIRST_SENT_END = re.compile(r"[.!?][\"\u201d\u2019']?(?:\s+|$)")
_WARM_PREFIX = re.compile(
    r"^\s*("
    r"awesome|great|nice|dope|alright|ok|okay|sure|yeah|yes|hey|cool|sweet|"
    r"excellent|wonderful|perfect|got\s+it|good\s+(one|call|choice)|gotcha|"
    r"oh|ah|right\s+(on)?|totally|love\s+(it|that)|"
    r"that[\u2019']?s\s+(awesome|great|nice|cool|wonderful|perfect|excellent|a\s+great|the\s+spirit)|"
    r"this\s+is\s+(awesome|great|nice|cool|wonderful|perfect|excellent)|"
    r"happy\s+you|here[\u2019']?s\s+|let[\u2019']?s\s+(dive|go|see|keep|continue|stick|switch|ride)|"
    r"sticking\s+with|since\s+you[\u2019']?re|since\s+you\s+like|since\s+you\s+enjoyed|"
    r"i[\u2019']?ve\s+got|i[\u2019']?m\s+(so\s+)?(glad|happy)|i\s+love\s+(that|it)|"
    r"you[\u2019']?re\s+(in|going|gonna|all)|you\s+(got|nailed)"
    r")\b",
    re.IGNORECASE,
)
_GLAD_ANYWHERE = re.compile(r"\bglad\b", re.IGNORECASE)


def _is_warm_sentence(sent: str) -> bool:
    """A sentence is 'warm' if it contains 'glad' anywhere or starts with a
    common conversational-feedback phrase."""
    return bool(_GLAD_ANYWHERE.search(sent) or _WARM_PREFIX.match(sent))
MIN_RESPONSE_WORDS = 40  # filter out chat-style 1-2 sentence replies
# Required citation pattern: year-in-parens (1990) OR quoted/asterisked title.
# Forces the response to anchor a concrete metadata fact, matching v10 style.
CITATION_PATTERN = re.compile(
    r"\(\d{4}\)|\*[^*]+\*|\"[^\"]{2,}\"|\u201c[^\u201d]{2,}\u201d"
)


def strip_warm_opener(resp: str) -> str:
    """Iteratively strip leading 'warm' sentences. Stops at the first
    sentence that's actual content (no glad/great/that's/etc).

    Caller must filter by length afterward — never fall back to the original
    (v17 showed openers teach the model bad patterns)."""
    resp = resp.strip()
    for _ in range(4):  # cap to avoid stripping entire response on edge cases
        m = _FIRST_SENT_END.search(resp)
        if not m:
            break
        first_sent = resp[:m.end()].strip()
        if not _is_warm_sentence(first_sent):
            break
        resp = resp[m.end():].strip()
    return resp


def first(v: Any) -> str:
    if isinstance(v, list):
        return v[0] if v else ""
    return v or ""


def format_track_for_prompt(meta: dict[str, Any]) -> str:
    name = first(meta.get("track_name"))
    artist = first(meta.get("artist_name"))
    album = first(meta.get("album_name"))
    release = first(meta.get("release_date"))
    year = release.split("-")[0] if release else ""
    tags = meta.get("tag_list") or []
    tag_str = ", ".join(tags[:5]) if isinstance(tags, list) else str(tags)
    parts = [f'"{name}"']
    if artist:
        parts.append(f"by {artist}")
    tail = [t for t in (album, year) if t]
    if tail:
        parts.append(f"({', '.join(tail)})")
    if tag_str:
        parts.append(f"[tags: {tag_str}]")
    return " ".join(parts)


def id_to_profile_str(user_meta: dict[str, Any] | None) -> str:
    """Match v10 inference's UserProfileDB.id_to_profile_str EXACTLY.
    Columns: user_id, age_group, gender, country_name. One per line."""
    if not user_meta:
        return ""
    cols = ["user_id", "age_group", "gender", "country_name"]
    return "\n".join(f"{k}: {user_meta.get(k)}" for k in cols)


def build_sys_prompt(roleplay: str, response_gen: str,
                     user_profile_blob: str, user_meta: dict[str, Any] | None,
                     conversation_goal: str | dict | None,
                     gold_track_meta: dict[str, Any]) -> str:
    """Reproduces v10 inference's build_sys_prompt_v2 exactly. CRITICAL for
    train-test parity — if the LoRA learns a different prompt format than
    inference uses, it produces garbage at inference time."""
    sections = [roleplay.strip(), response_gen.strip()]

    person_bits = []
    profile_str = id_to_profile_str(user_meta)
    if profile_str:
        person_bits.append(f"User profile (demographics): {profile_str}")
    if user_profile_blob:
        person_bits.append(f"Additional user context: {user_profile_blob}")
    if conversation_goal:
        # v10 stringifies the WHOLE dict (its f-string emits the dict repr).
        # Don't pre-extract listener_goal — that's a different prompt.
        person_bits.append(f"Conversation goal: {conversation_goal}")
    if person_bits:
        sections.append("=== About this user ===\n" + "\n".join(person_bits))

    track_line = f"1. {format_track_for_prompt(gold_track_meta)}"
    sections.append(
        "=== Candidate tracks (ranked) ===\n"
        "These are the tracks you have selected. Recommend #1 as the primary choice. "
        "You may optionally reference #2 or #3 as alternatives.\n\n"
        + track_line
    )
    return "\n\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="./data/train_sft",
                        help="Destination directory for the HF dataset.")
    parser.add_argument("--max_rows", type=int, default=None,
                        help="Optional cap (for quick test runs).")
    args = parser.parse_args()

    prompts_dir = os.path.join(os.path.dirname(__file__), "..", "prompts")
    roleplay = open(os.path.join(prompts_dir, "roleplay.txt"), encoding="utf-8").read()
    response_gen = open(os.path.join(prompts_dir, "response_generation_v4.txt"), encoding="utf-8").read()

    print("loading train dataset...", file=sys.stderr)
    train = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="train")
    print(f"train sessions: {len(train)}", file=sys.stderr)

    print("loading track metadata...", file=sys.stderr)
    track_meta_ds = load_dataset(
        "talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks"
    )
    track_meta_dict = {r["track_id"]: r for r in track_meta_ds}
    print(f"tracks in catalog: {len(track_meta_dict)}", file=sys.stderr)

    print("loading user metadata...", file=sys.stderr)
    try:
        user_meta_ds = load_dataset(
            "talkpl-ai/TalkPlayData-Challenge-User-Metadata", split="all_users"
        )
        user_meta_dict = {r["user_id"]: r for r in user_meta_ds}
    except Exception as e:
        print(f"user metadata load failed ({e}); skipping", file=sys.stderr)
        user_meta_dict = {}

    kept: list[dict[str, str]] = []
    stats = {
        "total_sessions": 0,
        "missing_turn1_msg": 0,
        "missing_turn1_music": 0,
        "missing_turn1_asst": 0,
        "music_not_in_catalog": 0,
        "junk_response": 0,
        "too_short": 0,
        "contains_banned": 0,
        "contains_apology": 0,
        "contains_hallucination": 0,
        "no_citation": 0,
        "bad_close": 0,
        "kept": 0,
    }

    n_iter = len(train) if args.max_rows is None else min(args.max_rows, len(train))
    for idx in range(n_iter):
        stats["total_sessions"] += 1
        session = train[idx]
        conv = session["conversations"]
        # Turn-1 ONLY. Blind-A is 80 single-turn cold-start queries; turns 2-8
        # are conversational follow-ups whose body references the previous
        # track ("Sticking with X", "Since you liked Y"). Even after stripping
        # warm openers, multi-turn responses teach the wrong distribution.
        for tn in [1]:
            turn = [t for t in conv if t.get("turn_number") == tn]
            user_msg = next((t["content"] for t in turn if t["role"] == "user"), None)
            music_id = next((t["content"] for t in turn if t["role"] == "music"), None)
            asst_resp = next((t["content"] for t in turn if t["role"] == "assistant"), None)

            if not user_msg:
                stats["missing_turn1_msg"] += 1
                continue
            if not music_id:
                stats["missing_turn1_music"] += 1
                continue
            if not asst_resp:
                stats["missing_turn1_asst"] += 1
                continue
            if music_id not in track_meta_dict:
                stats["music_not_in_catalog"] += 1
                continue
            if asst_resp.strip().lower() in {"unknown message", "", "none"}:
                stats["junk_response"] += 1
                continue

            cleaned = strip_warm_opener(asst_resp)
            if len(cleaned.split()) < MIN_RESPONSE_WORDS:
                stats["too_short"] += 1
                continue
            if BANNED_WORDS.search(cleaned):
                stats["contains_banned"] += 1
                continue
            if APOLOGY_PATTERNS.search(cleaned):
                stats["contains_apology"] += 1
                continue
            if HALLUCINATION_PATTERNS.search(cleaned):
                stats["contains_hallucination"] += 1
                continue
            if not CITATION_PATTERN.search(cleaned):
                stats["no_citation"] += 1
                continue
            if not cleaned.rstrip().endswith((".", "!", "?", '"', "\u201d", ")")):
                stats["bad_close"] += 1
                continue

            sys_prompt = build_sys_prompt(
                roleplay=roleplay,
                response_gen=response_gen,
                user_profile_blob=session.get("user_profile") or "",
                user_meta=user_meta_dict.get(session.get("user_id")),
                conversation_goal=session.get("conversation_goal"),
                gold_track_meta=track_meta_dict[music_id],
            )

            kept.append({
                "session_id": f"{session['session_id']}_t{tn}",
                "user_query": user_msg,
                "music_id": music_id,
                "system_prompt": sys_prompt,
                "assistant_response": cleaned,
            })
            stats["kept"] += 1

    print("\n=== Filter stats ===", file=sys.stderr)
    for k, v in stats.items():
        print(f"  {k}: {v}", file=sys.stderr)

    ds = Dataset.from_list(kept)
    os.makedirs(args.output, exist_ok=True)
    ds.save_to_disk(args.output)
    print(f"\nsaved {len(ds)} rows to {args.output}", file=sys.stderr)

    # Print sample triple for sanity.
    if len(ds):
        sample = ds[0]
        print("\n=== Sample row ===", file=sys.stderr)
        print(f"user_query: {sample['user_query'][:200]}", file=sys.stderr)
        print(f"music_id: {sample['music_id']}", file=sys.stderr)
        print(f"response: {sample['assistant_response'][:300]}", file=sys.stderr)


if __name__ == "__main__":
    main()
