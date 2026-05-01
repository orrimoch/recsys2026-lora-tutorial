"""Wrap reward dataset responses in the <user_state>...<response> envelope.

Per RecSys_Challenge_Plan §6.3 ("KTO data envelope augmentation, P1 #17"):
the existing `build_reward_dataset.py` outputs `(text_a, text_b, label)`
without the `<user_state>...<response>` envelope. **This script wraps text_b**
so B1 (W4 KTO) and B3 (W6 Rank-GRPO) share format expectations.

Without envelope augmentation, B1 trains the responder to produce raw text;
B3 then expects envelope-wrapped output → distribution shift → all of B3's
R_format hard-zero penalties fire on what was a valid B1 output.

Input:
    data/reward_train.parquet   (from scripts/build_reward_dataset.py)
        columns: text_a, text_b, label, session_id, user_id, turn_number,
                 goal_category, split

State cache (optional but recommended):
    music-crs-baselines/experiments/cache/state/{session_id}__{turn_number}.json
    (produced by colab/22_extract_train_states.ipynb running StateTracker
    on all train sessions). If a state file is missing, the envelope uses
    `(unknown)` as the state block — degraded but valid.

Output:
    data/reward_train_envelope.parquet
        Same schema as input, but text_b is now:
            <user_state>
            mood: ...
            intent: ...
            ...
            </user_state>
            <response>
            {original_text_b}
            </response>

Stub mode (--stub):
    Skip the state cache entirely; wrap every text_b with `<user_state>(unknown)
    </user_state><response>...`. Useful for unblocking the W4 training pipeline
    BEFORE the train-state extraction Colab finishes (deferred).

Usage:
    # With state cache (production path):
    python scripts/augment_envelope.py

    # Stub mode (training pipeline unblock):
    python scripts/augment_envelope.py --stub
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = REPO_ROOT / "data" / "reward_train.parquet"
DEFAULT_STATE_CACHE = (
    REPO_ROOT / "music-crs-baselines" / "experiments" / "cache" / "state"
)
DEFAULT_OUT = REPO_ROOT / "data" / "reward_train_envelope.parquet"

# Must match mcrs/query_rewriters/state_tracker.py:ALLOWED_KEYS and
# scripts/reward_fns.py:ALLOWED_STATE_KEYS.
ALLOWED_STATE_KEYS = ("mood", "intent", "energy", "sonic_pref", "era_pref", "familiarity")


def load_state(state_cache_dir: Path, session_id: str, turn_number: int) -> Optional[dict]:
    """Read a cached state JSON for (session, turn). Returns None if missing/invalid.

    Cache filename matches mcrs/query_rewriters/state_tracker.py:_cache_path —
    `{session_id}__{turn}.json` with non-alphanumeric chars replaced by `_`.
    """
    import re
    safe = re.sub(r"[^A-Za-z0-9_\-]", "_", str(session_id))
    cp = state_cache_dir / f"{safe}__{int(turn_number):d}.json"
    if not cp.exists():
        return None
    try:
        with cp.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        # Strip the `_fallback` marker if present — caller doesn't need it.
        return {k: v for k, v in data.items() if k != "_fallback"}
    except (OSError, json.JSONDecodeError):
        return None


def format_state_block(state: Optional[dict]) -> str:
    """Render a state dict as `key: value` lines for the envelope.

    Filters disallowed keys + skips empty / 'unknown' values. If no usable
    fields remain, returns the literal `(unknown)` placeholder so the
    envelope is structurally valid (parses through ENVELOPE regex).
    """
    if not state:
        return "(unknown)"
    lines = []
    for k in ALLOWED_STATE_KEYS:
        v = state.get(k)
        if v is None:
            continue
        v_str = str(v).strip()
        if not v_str or v_str.lower() == "unknown":
            continue
        lines.append(f"{k}: {v_str}")
    return "\n".join(lines) if lines else "(unknown)"


def wrap_envelope(response_text: str, state_block: str) -> str:
    """Produce the canonical envelope: <user_state>...</user_state>\\n<response>...</response>.

    Format matches scripts/reward_fns.ENVELOPE regex exactly so the W6 R_format
    check passes on these training rows.
    """
    return (
        f"<user_state>\n{state_block}\n</user_state>\n"
        f"<response>\n{response_text}\n</response>"
    )


def augment_row(row: dict, state: Optional[dict]) -> dict:
    """Return a new row with text_b wrapped in the envelope."""
    state_block = format_state_block(state)
    new_text_b = wrap_envelope(str(row["text_b"]), state_block)
    out = dict(row)
    out["text_b"] = new_text_b
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def augment(
    df: pd.DataFrame,
    state_cache_dir: Optional[Path],
    stub: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Wrap each row's text_b. Returns (new_df, stats)."""
    required_cols = {"text_a", "text_b", "label", "session_id", "turn_number"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Input parquet missing required columns: {missing}. "
            f"Got: {sorted(df.columns)}"
        )

    stats = {
        "rows": len(df),
        "state_hit": 0,
        "state_miss": 0,
        "stub_mode": bool(stub),
    }

    new_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="envelope"):
        if stub or state_cache_dir is None:
            state = None
            stats["state_miss"] += 1
        else:
            state = load_state(
                state_cache_dir,
                str(row["session_id"]),
                int(row["turn_number"]),
            )
            if state is None:
                stats["state_miss"] += 1
            else:
                stats["state_hit"] += 1
        new_rows.append(augment_row(row.to_dict(), state))

    out = pd.DataFrame(new_rows)
    # Sanity: schema preserved
    assert set(out.columns) == set(df.columns), (
        f"column mismatch: in={sorted(df.columns)} out={sorted(out.columns)}"
    )
    return out, stats


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="in_path", default=str(DEFAULT_IN))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument(
        "--state-cache-dir", default=str(DEFAULT_STATE_CACHE),
        help="Directory containing {session}__{turn}.json files.",
    )
    p.add_argument(
        "--stub", action="store_true",
        help="Skip the state cache entirely; use (unknown) for every row's "
             "user_state. Use to unblock the W4 training pipeline before "
             "train-state extraction completes.",
    )
    args = p.parse_args(argv)

    in_path = Path(args.in_path)
    if not in_path.exists():
        print(
            f"ERROR: input parquet not found at {in_path}.\n"
            f"Run `python scripts/build_reward_dataset.py` first.",
            file=sys.stderr,
        )
        return 1

    state_dir = None if args.stub else Path(args.state_cache_dir)
    if not args.stub and not state_dir.exists():
        print(
            f"WARNING: state cache dir not found at {state_dir} — falling "
            f"back to stub mode (every row will use `(unknown)`).\n"
            f"To populate the state cache, run colab/22_extract_train_states.ipynb.",
            file=sys.stderr,
        )
        args.stub = True
        state_dir = None

    print(f"[envelope] loading {in_path}")
    df = pd.read_parquet(in_path)
    print(f"[envelope] input: {len(df):,} rows")

    out_df, stats = augment(df, state_dir, stub=args.stub)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\n[envelope] wrote {len(out_df):,} rows → {out_path} ({size_mb:.1f} MB)")
    print(f"[envelope] stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    if stats["state_miss"] > 0 and not args.stub:
        miss_pct = 100 * stats["state_miss"] / stats["rows"]
        print(f"  ⚠️  {miss_pct:.1f}% of rows had no cached state — they got "
              f"`(unknown)` envelopes. Run colab/22_extract_train_states.ipynb "
              f"to populate the cache before W4 KTO training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
