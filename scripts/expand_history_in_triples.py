"""One-shot post-pass that retroactively expands music-turn track_ids in
the `[HISTORY]:` block of an existing triples JSONL — without re-mining.

Why: the builder used to leave music-turn IDs in chat_history as raw 7-char
hashes (e.g. `tk_a14b7`) while production inference (`run_inference_devset.py`
+ `run_inference_blindset.py`) expands them via
`MusicCatalogDB.id_to_metadata` into
`track_id: X, track_name: y, artist_name: y, album_name: y`.

The builder is now fixed for future mines (see _format_history_music_turn
in build_bi_encoder_training_data.py), but JSONL files that were mined
BEFORE the fix still carry raw IDs. Re-mining costs ~85 min on Blackwell;
this script rewrites the `query` field of every row in ~1-2 min CPU. Only
the [HISTORY]: `A: <id>` slots are touched — all other fields are preserved
byte-for-byte.

Usage:
    python scripts/expand_history_in_triples.py \\
        --input  experiments/cache/retrieval_v2/triples_bge_m3.jsonl \\
        --output experiments/cache/retrieval_v2/triples_bge_m3_fixed.jsonl \\
        --track-meta-hf talkpl-ai/TalkPlayData-Challenge-Track-Metadata \\
        --corpus-types track_name,artist_name,album_name

Idempotent: running twice on the same file produces identical output (already
expanded slots don't match the ID-shape regex because they contain spaces /
colons / commas).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


# Matches `A: <token>` where <token> is anything up to the next ` | ` or
# end of line. We then test membership in metadata_dict to confirm <token>
# is actually a real track_id (rules out catalog-drift IDs and any
# coincidental "A: <text>" that happens to come from a non-music assistant
# turn — though there shouldn't be any in the current data).
_A_SLOT_RE = re.compile(r"A: ([^|\n]+?)(?=\s*\||\n|$)")


def _format_history_music_turn(track_id: str, metadata_dict: dict,
                                corpus_types: list) -> str:
    """Byte-identical to MusicCatalogDB.id_to_metadata. Duplicated here
    (rather than imported from the builder) so this script has no heavy
    deps — runs CPU-only in a few seconds."""
    if track_id not in metadata_dict:
        return track_id
    md = metadata_dict[track_id]
    parts = [f"track_id: {track_id}"]
    for ct in corpus_types:
        val = md.get(ct)
        if val is None:
            joined = ""
        elif isinstance(val, list):
            joined = ", ".join(str(v) for v in val)
        else:
            joined = str(val)
        parts.append(f"{ct}: {joined.lower()}")
    return ", ".join(parts)


def expand_query_history(query: str, metadata_dict: dict,
                          corpus_types: list) -> str:
    """Rewrite a single `query` string: find `A: <id>` slots in the
    [HISTORY]: block, replace each with the id_to_metadata expansion when
    <id> is in metadata_dict. Leaves unknown IDs untouched (catalog-drift
    fallback) and never touches U: segments (regex is anchored on A:).

    Idempotency: an already-expanded slot looks like
    `A: track_id: X, track_name: ...` — the regex's <token> capture would
    be `track_id: X` which is not in metadata_dict, so the slot is left
    alone. (Strictly speaking the regex also stops at ` | ` which won't
    appear inside the expanded metadata since corpus_types use `, ` joins.)
    """
    if "[HISTORY]:" not in query:
        return query

    def _repl(m):
        token = m.group(1).strip()
        if token in metadata_dict:
            return f"A: {_format_history_music_turn(token, metadata_dict, corpus_types)}"
        return m.group(0)

    # Only rewrite within the [HISTORY]: line(s). The structured format puts
    # [HISTORY]: on its own line ending at \n[QUERY]: — so we can slice and
    # apply the regex only to that segment.
    h_idx = query.find("[HISTORY]:")
    after = query[h_idx:]
    next_block = after.find("\n[QUERY]:")
    if next_block == -1:
        # Defensive: malformed query; bail.
        return query
    history_segment = after[:next_block]
    rest_after_history = after[next_block:]
    rewritten = _A_SLOT_RE.sub(_repl, history_segment)
    return query[:h_idx] + rewritten + rest_after_history


def rewrite_jsonl(in_path: str, out_path: str, metadata_dict: dict,
                   corpus_types: list) -> dict:
    """Stream in_path → out_path; rewrite the `query` field of each row;
    return stats dict with rows_total, rows_touched, ids_expanded."""
    stats = {"rows_total": 0, "rows_touched": 0, "ids_expanded": 0}
    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(in_path) as f_in, open(out_path, "w") as f_out:
        for line in f_in:
            stats["rows_total"] += 1
            row = json.loads(line)
            orig_query = row.get("query", "")
            new_query = expand_query_history(orig_query, metadata_dict, corpus_types)
            if new_query != orig_query:
                stats["rows_touched"] += 1
                # Count A: slots that were rewritten (each starts with the prefix).
                stats["ids_expanded"] += (
                    new_query.count("A: track_id:")
                    - orig_query.count("A: track_id:")
                )
                row["query"] = new_query
            f_out.write(json.dumps(row) + "\n")
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Post-pass: expand music-turn track_ids in [HISTORY]: blocks."
    )
    parser.add_argument("--input", required=True,
                        help="Input JSONL produced by build_bi_encoder_training_data.py.")
    parser.add_argument("--output", required=True,
                        help="Output JSONL path. Will be overwritten if exists.")
    parser.add_argument("--track-meta-hf",
                        default="talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
                        help="HF dataset holding track metadata. Used to build the "
                             "lookup that mirrors id_to_metadata's output.")
    parser.add_argument("--corpus-types",
                        default="track_name,artist_name,album_name",
                        help="Comma-separated corpus_types matching the production "
                             "config's id_to_metadata setup (default = production "
                             "config 021).")
    args = parser.parse_args()

    from datasets import load_dataset

    corpus_types = [ct.strip() for ct in args.corpus_types.split(",") if ct.strip()]
    print(f"[expand-history] loading {args.track_meta_hf}", file=sys.stderr)
    track_meta = load_dataset(args.track_meta_hf, split="all_tracks")
    metadata_dict = {row["track_id"]: dict(row) for row in track_meta}
    print(f"[expand-history] {len(metadata_dict):,} catalog entries loaded",
          file=sys.stderr)
    print(f"[expand-history] corpus_types={corpus_types}", file=sys.stderr)

    stats = rewrite_jsonl(args.input, args.output, metadata_dict, corpus_types)
    print(f"[expand-history] DONE → {args.output}", file=sys.stderr)
    print(f"  rows_total    = {stats['rows_total']:,}", file=sys.stderr)
    print(f"  rows_touched  = {stats['rows_touched']:,} "
          f"({100.0 * stats['rows_touched'] / max(1, stats['rows_total']):.1f}%)",
          file=sys.stderr)
    print(f"  ids_expanded  = {stats['ids_expanded']:,}", file=sys.stderr)


if __name__ == "__main__":
    main()
