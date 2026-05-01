"""Build the S-DPO preference dataset from envelope-augmented reward parquet.

Per RecSys_Challenge_Plan §6.3 row B2 + §6.4: 1 positive + N negatives per turn.
For W5 (conditional, only runs if W4 KTO format compliance < 70%) we generate
4 cheap negatives per positive — total ~120k DPO pairs from ~30k POS turns.

Why we use vanilla TRL DPOTrainer instead of true Plackett-Luce S-DPO:
    Plan §6.3 cites S-DPO with "1 positive + 6 negatives" as a Plackett-Luce
    softmax loss. Vanilla TRL DPOTrainer is pairwise. We approximate by
    EXPANDING each (1-pos, N-negs) into N independent (chosen, rejected) pairs.
    This loses the cross-negative regularization but uses TRL directly.
    If the W5 gate misses, a custom S-DPO loss would be the upgrade path
    (subclass DPOTrainer.compute_loss; ~150 LOC).

Negative types (4 per positive turn):
    1. drop_track_name  — gold envelope, but track/artist name → "this song"/"the artist"
    2. inject_banned    — gold envelope + banned phrase appended inside <response>
    3. truncate_5       — gold envelope, but inner response truncated to 5 words
    4. cross_session_neg — same session's GPA-DOES_NOT_MOVE turn's response
                          (only present if the session has any NEG turn; falls
                          back to drop_why perturbation if not)

Plan §6.4 also lists "3 hard-track negs (BM25 top-2..4 not gold) paired with
on-policy Qwen responses" — DEFERRED. Producing those requires running
Qwen-7B inference on ~30k × 3 = 90k extra hard-track candidates (~2 A100-hr).
For W5 conditional path, the 4 cheap negatives suffice. The hard-track
extension is documented as `--include-hard-track` future flag.

CRITICAL: perturbations operate on the INSIDE of <response>...</response>
only. The <user_state> block + envelope tags are preserved verbatim so the
DPO chosen and rejected differ ONLY in response content, not envelope shape.
This keeps the model's r_format gradient consistent across both sides of
the pair (we don't want DPO to teach "envelope structure is bad").

Output:
    data/trl/sdpo.parquet
        columns: prompt, chosen, rejected, split
        ~120k rows from ~30k POS turns × 4 negatives.

Usage:
    # Pre-req: data/reward_train_envelope.parquet (from build_reward_dataset
    # → augment_envelope pipeline). The envelope wrapping must already be
    # present — without it the perturbations are meaningless for B2.
    python scripts/build_sdpo_dataset.py
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = REPO_ROOT / "data" / "reward_train_envelope.parquet"
DEFAULT_OUT = REPO_ROOT / "data" / "trl" / "sdpo.parquet"

# Regex matching the envelope produced by scripts/augment_envelope.py.
# Captures (state_block, response_text). Identical to scripts/reward_fns.ENVELOPE.
ENVELOPE_RE = re.compile(
    r"<user_state>(.*?)</user_state>\s*<response>(.*?)</response>",
    re.DOTALL,
)

# Banned phrase appended in the inject_banned perturbation. Same set the
# response prompt forbids (mcrs/system_prompts/response_generation_cot_user_state.txt:22).
BANNED_INJECTION = " absolutely fantastic, perfectly amazing!"


def _track_artist_from_text_a(text_a: str) -> tuple[str, str]:
    """Pull track_name + artist_name from the build_reward_dataset.py text_a.

    text_a contains "Recommended track: {name} by {artist} [{tags}]" or — when
    `tag_list` is empty — "Recommended track: {name} by {artist}" followed by
    a newline (Prior dialog: ...). See `scripts/build_reward_dataset.py:track_summary`.

    P0 #1 fix (W5 review): the prior regex only matched when followed by ` [`
    (the tag-list bracket) OR end-of-string with no MULTILINE flag. Real
    text_a has more lines after, so the regex returned None on every row
    without a tag_list, silently dropping the drop_track_name perturbation.
    Now anchors on `\\s+\\[ | \\n | $` with MULTILINE.
    """
    m = re.search(
        r"^Recommended track:\s*(.+?)(?:\s+by\s+(.+?))?(?:\s+\[|\n|$)",
        text_a,
        flags=re.MULTILINE,
    )
    if not m:
        return "", ""
    name = (m.group(1) or "").strip()
    artist = (m.group(2) or "").strip()
    return name, artist


def _split_envelope(envelope_wrapped: str) -> tuple[str, str, str]:
    """Return (prefix, inner_response, suffix) for the envelope. If parsing
    fails, return the whole text as inner_response with empty prefix/suffix
    — safe degradation, but `chosen != rejected` still holds for perturbations.
    """
    m = ENVELOPE_RE.search(envelope_wrapped)
    if not m:
        return "", envelope_wrapped, ""
    inner = m.group(2).strip()
    full_open_tag = "<response>"
    full_close_tag = "</response>"
    open_start = envelope_wrapped.find(full_open_tag)
    close_start = envelope_wrapped.rfind(full_close_tag)
    if open_start < 0 or close_start < 0:
        return "", inner, ""
    prefix = envelope_wrapped[: open_start + len(full_open_tag)]
    suffix = envelope_wrapped[close_start:]
    return prefix, inner, suffix


def _rewrap(prefix: str, new_inner: str, suffix: str) -> str:
    """Reassemble envelope with mutated inner response. Guarantees envelope
    tags are preserved verbatim — only inner response text changes."""
    if not prefix or not suffix:
        # Fallback when split failed: just return the new inner.
        return new_inner
    return f"{prefix}\n{new_inner.strip()}\n{suffix}"


# ---------------------------------------------------------------------------
# Perturbations — each operates on inner response text and returns the
# perturbed inner text. The envelope is reattached by `_rewrap`.
# ---------------------------------------------------------------------------

def _perturb_drop_track_name(inner: str, track_name: str, artist_name: str) -> str:
    """Replace track_name + artist_name with anonymous placeholders.

    P0 #3 fix (W5 review): use \\b word boundaries to avoid corrupting
    substrings — e.g. a track literally named "track" was matching inside
    "soundtrack" → "soundthis song". Also reject names < 3 chars (too
    likely to be common English words like "I"/"You"/"The").
    """
    out = inner
    if track_name and len(track_name) >= 3:
        out = re.sub(
            r"\b" + re.escape(track_name) + r"\b",
            "this song",
            out,
            flags=re.IGNORECASE,
        )
    if artist_name and len(artist_name) >= 3:
        out = re.sub(
            r"\b" + re.escape(artist_name) + r"\b",
            "the artist",
            out,
            flags=re.IGNORECASE,
        )
    return out


def _perturb_inject_banned(inner: str) -> str:
    return inner.rstrip() + BANNED_INJECTION


def _perturb_truncate_5(inner: str) -> str:
    return " ".join(inner.split()[:5]) or "(empty)"


def _perturb_drop_why(inner: str) -> str:
    """Replace musical-detail vocab with [X]. Used as fallback negative
    when no cross-session GPA NEG response is available."""
    why_re = re.compile(
        r"\b(because|since|features|leans|driven by|atmosphere|tempo|groove|"
        r"arrangement|released|from \d{4}|era|decade|vibe|texture|timbre|"
        r"harmony|melody|rhythm)\b",
        re.IGNORECASE,
    )
    return why_re.sub("[X]", inner)


# ---------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------

def _build_neg_inner(
    variant: str,
    inner: str,
    track_name: str,
    artist_name: str,
) -> str:
    if variant == "drop_track_name":
        return _perturb_drop_track_name(inner, track_name, artist_name)
    if variant == "inject_banned":
        return _perturb_inject_banned(inner)
    if variant == "truncate_5":
        return _perturb_truncate_5(inner)
    if variant == "drop_why":
        return _perturb_drop_why(inner)
    raise ValueError(f"unknown perturbation variant: {variant}")


PERTURBATION_VARIANTS = ["drop_track_name", "inject_banned", "truncate_5"]
# `drop_why` is the fallback when no cross-session GPA NEG exists.


def build_sdpo_pairs(df: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """Generate (prompt, chosen, rejected, split) DPO pairs.

    For each POS row (label==1):
      - Apply 3 perturbations to the gold envelope-wrapped response.
      - Try to find a cross-session GPA NEG response from the SAME session_id;
        if found, use it as a 4th negative; else fall back to drop_why.
    """
    required = {"text_a", "text_b", "label", "session_id", "turn_number"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"S-DPO input missing required columns: {missing}. "
            f"Got: {sorted(df.columns)}. Run scripts/build_reward_dataset.py "
            f"and scripts/augment_envelope.py first."
        )

    # Index NEG rows by session_id for quick cross-session lookup.
    neg_by_session: dict[str, list[dict]] = {}
    for _, row in df[df["label"].astype(int) == 0].iterrows():
        neg_by_session.setdefault(row["session_id"], []).append(row.to_dict())

    rng = np.random.default_rng(seed)
    pos_df = df[df["label"].astype(int) == 1]
    print(f"[sdpo] POS turns: {len(pos_df):,}  (will produce {4 * len(pos_df):,} pairs)")
    print(f"[sdpo] sessions with at least one NEG turn: {len(neg_by_session):,}")

    pairs: list[dict] = []
    stats = {
        "pos_turns_processed": 0,
        "perturbation_negs": 0,
        "cross_session_negs": 0,
        "drop_why_fallbacks": 0,
        "envelope_parse_failures": 0,
        "degenerate_pairs_skipped": 0,
        # P1 #7 fix (W5 review): silent-noop counters so we can detect when
        # a perturbation didn't actually change the response (e.g. track name
        # not present in the gold response). Without these, silent drops to
        # `degenerate_pairs_skipped` masked the regex bugs in P0 #1/#3.
        "drop_track_name_silent_noop": 0,
        "drop_why_silent_noop": 0,
        "track_name_extraction_failures": 0,
    }

    for _, row in tqdm(pos_df.iterrows(), total=len(pos_df), desc="sdpo pairs"):
        prompt = str(row["text_a"])
        chosen_full = str(row["text_b"])  # envelope-wrapped gold
        sid = row["session_id"]
        split_value = str(row.get("split", "train"))

        prefix, inner, suffix = _split_envelope(chosen_full)
        if not prefix or not suffix:
            stats["envelope_parse_failures"] += 1
        track_name, artist_name = _track_artist_from_text_a(prompt)
        if not track_name and not artist_name:
            stats["track_name_extraction_failures"] += 1

        stats["pos_turns_processed"] += 1

        # Three response-perturbation negatives (always produced).
        for variant in PERTURBATION_VARIANTS:
            new_inner = _build_neg_inner(variant, inner, track_name, artist_name)
            # P1 #7 fix: detect silent noops PER VARIANT before the
            # degenerate_pairs_skipped catch-all so we know which perturbation
            # is failing.
            if new_inner == inner:
                if variant == "drop_track_name":
                    stats["drop_track_name_silent_noop"] += 1
                elif variant == "drop_why":
                    stats["drop_why_silent_noop"] += 1
            rejected = _rewrap(prefix, new_inner, suffix) if prefix and suffix else new_inner
            if rejected == chosen_full:
                stats["degenerate_pairs_skipped"] += 1
                continue
            pairs.append({
                "prompt": prompt,
                "chosen": chosen_full,
                "rejected": rejected,
                "split": split_value,
                "neg_type": variant,
            })
            stats["perturbation_negs"] += 1

        # Fourth negative: cross-session GPA NEG, with drop_why fallback.
        candidates = neg_by_session.get(sid, [])
        if candidates:
            neg_row = candidates[int(rng.integers(0, len(candidates)))]
            rejected = str(neg_row["text_b"])
            if rejected != chosen_full:
                pairs.append({
                    "prompt": prompt,
                    "chosen": chosen_full,
                    "rejected": rejected,
                    "split": split_value,
                    "neg_type": "cross_session_neg",
                })
                stats["cross_session_negs"] += 1
            else:
                # Edge case: same response appears as both POS and NEG of the
                # session. Fall back to drop_why.
                stats["degenerate_pairs_skipped"] += 1
                fallback = _build_neg_inner("drop_why", inner, track_name, artist_name)
                fallback = _rewrap(prefix, fallback, suffix) if prefix and suffix else fallback
                if fallback != chosen_full:
                    pairs.append({
                        "prompt": prompt,
                        "chosen": chosen_full,
                        "rejected": fallback,
                        "split": split_value,
                        "neg_type": "drop_why",
                    })
                    stats["drop_why_fallbacks"] += 1
        else:
            fallback = _build_neg_inner("drop_why", inner, track_name, artist_name)
            fallback = _rewrap(prefix, fallback, suffix) if prefix and suffix else fallback
            if fallback != chosen_full:
                pairs.append({
                    "prompt": prompt,
                    "chosen": chosen_full,
                    "rejected": fallback,
                    "split": split_value,
                    "neg_type": "drop_why",
                })
                stats["drop_why_fallbacks"] += 1
            else:
                stats["degenerate_pairs_skipped"] += 1

    out_df = pd.DataFrame(pairs)
    # Final assertion: no degenerate pairs in output.
    n_degenerate = int((out_df["chosen"] == out_df["rejected"]).sum())
    assert n_degenerate == 0, f"{n_degenerate} degenerate pairs survived"
    return out_df, stats


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="in_path", default=str(DEFAULT_IN))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    in_path = Path(args.in_path)
    if not in_path.exists():
        print(
            f"ERROR: input parquet not found at {in_path}.\n"
            f"Run `python scripts/build_reward_dataset.py` then "
            f"`python scripts/augment_envelope.py` first.",
            file=sys.stderr,
        )
        return 1

    print(f"[sdpo] loading {in_path}")
    df = pd.read_parquet(in_path)
    print(f"[sdpo] input: {len(df):,} rows")

    out_df, stats = build_sdpo_pairs(df, seed=args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # TRL DPOTrainer expects exactly (prompt, chosen, rejected); we keep
    # `split` (train/val) and `neg_type` (diagnostic) as extra columns.
    out_df.to_parquet(out_path, index=False)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\n[sdpo] wrote {len(out_df):,} pairs → {out_path} ({size_mb:.1f} MB)")
    print("[sdpo] stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print("[sdpo] negative-type breakdown:")
    for nt, n in out_df["neg_type"].value_counts().items():
        print(f"  {nt}: {n:,}")
    # P1 #7 fix: warn loudly if perturbations are silently no-opping at scale.
    n_pos = stats.get("pos_turns_processed", 0)
    if n_pos > 0:
        dt_noop_pct = 100.0 * stats["drop_track_name_silent_noop"] / n_pos
        ext_fail_pct = 100.0 * stats["track_name_extraction_failures"] / n_pos
        if dt_noop_pct > 5:
            print(f"\n⚠️  drop_track_name silent-noop rate: {dt_noop_pct:.1f}% "
                  f"(threshold 5%) — track name not present in gold response.")
        if ext_fail_pct > 5:
            print(f"\n⚠️  track_name extraction failed on {ext_fail_pct:.1f}% "
                  f"of rows — check the regex against build_reward_dataset.py text_a format.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
