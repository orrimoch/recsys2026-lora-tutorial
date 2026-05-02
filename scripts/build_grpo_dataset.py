"""Build the W6 Rank-GRPO training dataset by joining envelope-augmented
POS rows with pre-computed retriever output.

Per RecSys_Challenge_Plan §6.3 row B3 + §6.4 + §6.5 + W6 design notes
(memory project_w6_design_notes.md):

- The retriever (CMQR + ProRank + catalog filter) is FROZEN at W6 start.
  Its outputs (`predicted_track_ids`, top-1 metadata, per-rank rationales)
  are pre-computed ONCE in the colab notebook for every unique
  (session_id, turn_number) in the POS subset and dumped as
  `data/trl/grpo_retrieval.parquet`. This script then joins on those keys
  and emits the final `data/trl/grpo.parquet` ready for TRL GRPOTrainer.

- The output prompt embeds the `<reranker_rationales>` block (plan §6.5).
  W4/W5 trained without rationales; W6 adds them so the policy learns to
  USE the reranker output. Dev-eval config 220 will likewise feed live
  rationales at inference, matching W6's training distribution.

- During GRPO, the responder generates TEXT only. R_retr is constant
  across rollouts (retriever is frozen) so its 0.70 weight contributes
  ZERO gradient — gradient comes from R_rule (0.15) + R_judge (0.10) +
  R_format (0.05). This is acceptable per W6 design notes; the constant
  R_retr term still provides a positional bias in the per-prompt advantage
  baseline.

Output schema (`data/trl/grpo.parquet`):
    prompt              str  full chat-template-ready prompt; text_a + rationales block
    gold_track_id       str  UUID of the assistant's recommended track at this turn
    predicted_track_ids list[str]  frozen retriever top-N (target len 20 post-catalog-filter)
    top1_meta_json      str  JSON: {"track_name", "artist_name"} for r_rule
    user_state_json     str  JSON of parsed <user_state> block; for r_rule echo
    history_text        str  prior dialog summary; for r_rule history grounding
    session_id          str
    turn_number         int
    split               str  "train" / "val"

Usage:
    # Pre-req: data/reward_train_envelope.parquet (W4 envelope augmentation)
    #          + data/trl/grpo_retrieval.parquet (notebook 32 cell N).
    python scripts/build_grpo_dataset.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENVELOPE_IN = REPO_ROOT / "data" / "reward_train_envelope.parquet"
DEFAULT_RETRIEVAL_IN = REPO_ROOT / "data" / "trl" / "grpo_retrieval.parquet"
DEFAULT_OUT = REPO_ROOT / "data" / "trl" / "grpo.parquet"

# Re-use the canonical envelope regex + state parser from reward_fns so the
# B1/B2/B3 stages share the same envelope contract. Importing here is safe:
# reward_fns has no torch/transformers imports at module top-level.
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
from reward_fns import parse_envelope, parse_user_state  # noqa: E402


# ---------------------------------------------------------------------------
# Reranker rationales block (plan §6.5)
# ---------------------------------------------------------------------------

def format_rationales_block(rationales: list[str], top_k: int = 20) -> str:
    """Render the `<reranker_rationales>...</reranker_rationales>` slot.

    Plan §6.5: B3's prompt envelope adds a per-item rationale block fed by
    the ProRank reranker. Format is one-line-per-rank, 1-indexed:

        <reranker_rationales>
        1. matches mood + low energy
        2. 1990s era pref
        </reranker_rationales>

    `top_k` is the max number of rationales emitted (default 20 = our
    retrieval target length post-catalog-filter). Empty input still emits
    a structurally-valid (empty-body) block so the prompt template is
    uniform across rows — the responder must learn to handle both.

    Stray internal newlines in a rationale are collapsed to spaces so each
    rank stays on a single line.
    """
    lines = ["<reranker_rationales>"]
    for i, rat in enumerate(rationales[:top_k], start=1):
        clean = " ".join(str(rat).split())
        lines.append(f"{i}. {clean}")
    lines.append("</reranker_rationales>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Envelope state extraction (re-uses reward_fns parsers)
# ---------------------------------------------------------------------------

def extract_user_state(envelope_wrapped: str) -> dict[str, str]:
    """Parse the `<user_state>` block from an envelope-wrapped text_b.

    Returns {} if the envelope is missing or the state block is empty —
    `compose_r_turn(... user_state={})` is well-defined (`r_rule` simply
    skips the echo sub-score). Wrong returns must NOT raise; W6 dataset
    rows with broken envelopes still need a parseable row in the parquet.
    """
    parsed = parse_envelope(envelope_wrapped)
    if not parsed:
        return {}
    state_block, _resp = parsed
    return parse_user_state(state_block)


# ---------------------------------------------------------------------------
# History extraction from text_a
# ---------------------------------------------------------------------------

# W6 review P0-4 fix: real `Prior dialog:` content from
# build_reward_dataset.py:88-102 spans multiple lines (newline-joined
# `role: content`). The previous lazy-non-greedy regex with MULTILINE only
# matched the FIRST line. Use DOTALL + `\Z` end-of-string anchor so we
# capture every line of history; collapse newlines to spaces for r_rule.
#
# Deep-review P1-5 fix: use `re.findall` and take the LAST match. Without
# this, if a user_query contains the literal substring "Prior dialog:"
# (rare but possible), the regex anchors there and over-captures. Anchoring
# on the last occurrence guarantees we capture the field as written by
# build_reward_dataset.py:170 (which always emits Prior dialog: as the
# final field).
_HISTORY_RE = re.compile(r"Prior dialog:\s*(.+?)(?=\nPrior dialog:|\Z)", re.DOTALL)


def extract_history_text(text_a: str) -> str:
    """Pull the full `Prior dialog: ...` block from build_reward_dataset.py text_a.

    Returns "" when no such marker exists. `r_rule` uses this as the
    history-grounding sub-score: any ≥5-char history token appearing in
    the response. Multi-line history (real shape — see W6 review P0-4) is
    collapsed to a single line so the token-match is per-token, not
    per-line. When multiple "Prior dialog:" tokens exist, the LAST match
    wins (deep-review P1-5).
    """
    matches = _HISTORY_RE.findall(text_a or "")
    if not matches:
        return ""
    # Take the LAST match — build_reward_dataset.py:170 emits Prior dialog
    # as the trailing field, so the last match is always the actual history.
    return " ".join(matches[-1].split())


# ---------------------------------------------------------------------------
# Schema validators
# ---------------------------------------------------------------------------

ENVELOPE_REQUIRED = {"text_a", "text_b", "label", "session_id", "turn_number"}
# `user_profile_json` is OPTIONAL in the envelope schema for back-compat with
# parquets built before gap-analysis Step 3. When present, build_grpo_dataset
# carries it through; when absent, downstream emits empty JSON ("{}") and
# r_user_prof contributes 0.
RETRIEVAL_REQUIRED = {
    "session_id", "turn_number",
    "gold_track_id", "predicted_track_ids",
    "top1_track_name", "top1_artist_name",
    "reranker_rationales",
}


def validate_envelope_schema(df: pd.DataFrame) -> None:
    """Raise ValueError with full diagnostic if envelope parquet is malformed."""
    missing = ENVELOPE_REQUIRED - set(df.columns)
    if missing:
        raise ValueError(
            f"envelope parquet missing required columns: {sorted(missing)}. "
            f"Got: {sorted(df.columns)}. Run scripts/build_reward_dataset.py "
            f"and scripts/augment_envelope.py first."
        )


def validate_retrieval_schema(df: pd.DataFrame) -> None:
    """Raise ValueError if retrieval parquet (notebook 32 output) is malformed."""
    missing = RETRIEVAL_REQUIRED - set(df.columns)
    if missing:
        raise ValueError(
            f"retrieval parquet missing required columns: {sorted(missing)}. "
            f"Got: {sorted(df.columns)}. Re-run the retrieval pre-compute "
            f"cell in colab/32_train_responder_grpo.ipynb."
        )


# ---------------------------------------------------------------------------
# Core join
# ---------------------------------------------------------------------------

def build_grpo_dataset(
    envelope_df: pd.DataFrame,
    retrieval_df: pd.DataFrame,
    seed: int = 42,
    system_prompt: "str | None" = None,
    include_rationales: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Join envelope-augmented POS rows with frozen-retriever output.

    Args:
        envelope_df: data/reward_train_envelope.parquet — text_a, text_b,
            label, session_id, turn_number, split.
        retrieval_df: data/trl/grpo_retrieval.parquet (notebook 32 output)
            keyed by (session_id, turn_number) with frozen retriever output.
        seed: unused today; accepted for symmetry with build_sdpo_dataset.
        system_prompt: optional. When provided, `prompt` column is emitted
            as a conversational list-of-message-dicts
            (`[{"role":"system",...}, {"role":"user", text_a + rationales}]`)
            so TRL `GRPOTrainer` auto-applies the tokenizer chat template
            (matching the chat-templated W4/W5 init AND the eval-time
            generation path — fixes W6 review P0-3 train/inference
            distribution mismatch). When None, emits the legacy flat-string
            prompt for back-compat.

    Returns:
        (out_df, stats). out_df has columns:
            prompt, gold_track_id, predicted_track_ids, top1_meta_json,
            user_state_json, history_text, session_id, turn_number, split.
        `prompt` is `str` when system_prompt is None, else `list[dict]`.
        stats has counters used by the colab notebook to surface bad joins
        early (low recall, missing retrieval rows, envelope parse fail).
    """
    validate_envelope_schema(envelope_df)
    validate_retrieval_schema(retrieval_df)

    pos_df = envelope_df[envelope_df["label"].astype(int) == 1].copy()
    n_pos = len(pos_df)

    joined = pos_df.merge(
        retrieval_df,
        on=["session_id", "turn_number"],
        how="inner",
        validate="one_to_one",
    )

    stats = {
        "pos_rows_in": n_pos,
        "joined_rows": len(joined),
        "unjoined_pos": n_pos - len(joined),
        "envelope_parse_failures": 0,
        "gold_in_predicted": 0,
        "recall_at_n": 0.0,
    }

    out_rows: list[dict] = []
    for _, row in tqdm(joined.iterrows(), total=len(joined), desc="grpo dataset"):
        text_a = str(row["text_a"])
        text_b = str(row["text_b"])
        rationales = list(row["reranker_rationales"]) if row["reranker_rationales"] is not None else []
        predicted = list(row["predicted_track_ids"]) if row["predicted_track_ids"] is not None else []

        user_state = extract_user_state(text_b)
        if not user_state and "<user_state>" not in text_b:
            stats["envelope_parse_failures"] += 1

        history_text = extract_history_text(text_a)

        # Deep-review P0-2 fix: rationales block is OFF by default. The
        # plan §6.5 design (rationales in prompt) was only ever wired into
        # W6 training, never into W4/W5 training, never into inference
        # (`run_inference_blindset.py` → `crs_baseline.batch_chat` does not
        # inject rationales). Including them at training time alone creates
        # a train/inference distribution mismatch. Re-enable via
        # `include_rationales=True` only when an inference-time adapter
        # exists in mcrs/crs_baseline.py.
        if include_rationales:
            rationales_block = format_rationales_block(rationales)
            prompt_body = f"{text_a}\n{rationales_block}"
        else:
            prompt_body = text_a
        # When a system_prompt is supplied, emit conversational form so TRL
        # auto-applies the chat template at training time (same shape the
        # W4/W5 init was trained on AND what the eval cell feeds at gen).
        # Otherwise fall back to the legacy flat string for back-compat.
        if system_prompt is not None:
            prompt = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_body},
            ]
        else:
            prompt = prompt_body

        top1_meta = {
            "track_name": str(row["top1_track_name"] or ""),
            "artist_name": str(row["top1_artist_name"] or ""),
        }
        gold = str(row["gold_track_id"])
        if gold and gold in predicted:
            stats["gold_in_predicted"] += 1

        # Carry user_profile_json from envelope when present (gap-analysis
        # Step 3). When absent (older parquets), emit empty JSON so the
        # reward closure's json.loads doesn't crash and r_user_prof gets 0.
        user_profile_json = row.get("user_profile_json")
        if user_profile_json is None or pd.isna(user_profile_json):
            user_profile_json = "{}"
        out_rows.append({
            "prompt": prompt,
            "gold_track_id": gold,
            "predicted_track_ids": predicted,
            "top1_meta_json": json.dumps(top1_meta, ensure_ascii=False),
            "user_state_json": json.dumps(user_state, ensure_ascii=False),
            "user_profile_json": str(user_profile_json),
            "history_text": history_text,
            "session_id": row["session_id"],
            "turn_number": int(row["turn_number"]),
            "split": str(row.get("split", "train")),
        })

    if stats["joined_rows"] > 0:
        stats["recall_at_n"] = stats["gold_in_predicted"] / stats["joined_rows"]

    out_df = pd.DataFrame(out_rows)
    return out_df, stats


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--envelope", default=str(DEFAULT_ENVELOPE_IN),
                   help="path to data/reward_train_envelope.parquet")
    p.add_argument("--retrieval", default=str(DEFAULT_RETRIEVAL_IN),
                   help="path to data/trl/grpo_retrieval.parquet (notebook 32 cell)")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--system-prompt-path", default=None,
        help=(
            "optional path to a system-prompt text file (e.g. roleplay.txt + "
            "response_generation_cot_user_state.txt concatenated). When set, "
            "the `prompt` column is emitted as conversational message-list "
            "form so TRL auto-applies the chat template at training time "
            "(W6 review P0-3 fix). Recommended for W6."
        ),
    )
    p.add_argument(
        "--include-rationales", action="store_true", default=False,
        help=(
            "Include the <reranker_rationales> block in the prompt body. "
            "OFF by default per deep-review P0-2 — the inference path does "
            "NOT inject rationales, so adding them at training creates a "
            "train/inference distribution mismatch. Enable only after "
            "wiring rationales into mcrs.crs_baseline.batch_chat."
        ),
    )
    args = p.parse_args(argv)

    env_path = Path(args.envelope)
    ret_path = Path(args.retrieval)
    if not env_path.exists():
        print(f"ERROR: envelope parquet not found at {env_path}.", file=sys.stderr)
        return 1
    if not ret_path.exists():
        print(f"ERROR: retrieval parquet not found at {ret_path}. "
              f"Run the retrieval pre-compute cell in colab/32_train_responder_grpo.ipynb.",
              file=sys.stderr)
        return 1

    system_prompt = None
    if args.system_prompt_path:
        sp_path = Path(args.system_prompt_path)
        if not sp_path.exists():
            print(f"ERROR: system-prompt file not found at {sp_path}.", file=sys.stderr)
            return 1
        system_prompt = sp_path.read_text(encoding="utf-8")
        print(f"[grpo] system prompt loaded ({len(system_prompt):,} chars)")

    print(f"[grpo] loading envelope:  {env_path}")
    env_df = pd.read_parquet(env_path)
    print(f"[grpo] loading retrieval: {ret_path}")
    ret_df = pd.read_parquet(ret_path)
    print(f"[grpo] envelope rows: {len(env_df):,}  retrieval rows: {len(ret_df):,}")

    out_df, stats = build_grpo_dataset(
        env_df, ret_df, seed=args.seed, system_prompt=system_prompt,
        include_rationales=args.include_rationales,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\n[grpo] wrote {len(out_df):,} rows → {out_path} ({size_mb:.1f} MB)")
    print("[grpo] stats:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")

    if stats["recall_at_n"] < 0.30 and stats["joined_rows"] > 0:
        print(f"\n⚠️  recall@N is low ({stats['recall_at_n']:.1%}). "
              f"R_retr will be near-zero for most rows → gradient signal "
              f"reduced to R_rule + R_judge + R_format only.")
    if stats["unjoined_pos"] / max(stats["pos_rows_in"], 1) > 0.10:
        print(f"\n⚠️  {stats['unjoined_pos']:,} POS rows "
              f"({100 * stats['unjoined_pos'] / max(stats['pos_rows_in'], 1):.0f}%) "
              f"missing from retrieval cache. Re-run the precompute cell over "
              f"all POS turns.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
