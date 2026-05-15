"""Generate synthetic conversational queries per catalog track via a small LLM.

Doc2query: classic IR trick for sparse text catalogs against natural-language
queries. For each of 47K tracks, ask an LLM to produce ~5 conversational
queries a real user might use to be recommended that track. Index those
alongside the metadata so retrieval matches user intent, not just identifier text.

Why we expect this to help: Phase 0 diagnostic showed 78.5% of misses are
`not_in_either` (gold absent from both BM25 top-100 AND dense top-100). That's
a *recall* problem — the candidate set itself is too narrow. Doc2query directly
enlarges what the index can match against.

Usage:
    python scripts/doc2query_generate.py \\
        --model Qwen/Qwen2.5-1.5B-Instruct \\
        --n-queries 5 \\
        --batch-size 16 \\
        --max-tracks 20            # smoke run first; remove for full 47K

Output: parquet at experiments/cache/doc2query/<safe_model>/queries.parquet
  Columns: track_id (str), synthetic_queries (list[str])
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


_PROMPT_TEMPLATE = """You are a music recommendation expert. Given a track's metadata, generate {n_queries} diverse conversational queries that a real user might use to ask a recommender for this track. Each query should be a short natural-language request (5–15 words) — like what someone would actually type or say. Vary mood, occasion, era, and style across the {n_queries} queries.

Track metadata:
{metadata_lines}

Output exactly {n_queries} queries, one per line, no numbering or prefixes:"""


def build_doc2query_prompt(metadata: dict, n_queries: int = 5) -> str:
    """Build the LLM prompt asking for n_queries conversational descriptions of a track."""
    lines: list[str] = []
    label_map = {
        "track_name": "Title",
        "artist_name": "Artist",
        "album_name": "Album",
        "tag_list": "Tags",
        "release_date": "Release year",
    }
    for field, label in label_map.items():
        val = metadata.get(field)
        if val is None:
            continue
        if isinstance(val, list):
            if not val:
                continue
            rendered = ", ".join(str(v) for v in val)
        else:
            if val == "" or val is None:
                continue
            rendered = str(val)
        lines.append(f"- {label}: {rendered}")
    return _PROMPT_TEMPLATE.format(
        n_queries=n_queries,
        metadata_lines="\n".join(lines),
    )


_NUM_PREFIX_RE = re.compile(r"^\s*(?:\d+[\.\)]|\-|\*|•)\s+")


def parse_generated_queries(completion: str, n_expected: int = 5) -> list[str]:
    """Extract clean queries from a raw LLM completion (handles numbered lists, bullets, preamble)."""
    queries: list[str] = []
    for raw in completion.splitlines():
        line = raw.strip()
        if not line:
            continue
        # Drop common preamble lines ("Here are 5 queries:", "Queries:", etc.)
        if line.lower().endswith(":") and "quer" in line.lower():
            continue
        if line.lower() in ("here are the queries", "queries"):
            continue
        # Strip leading list markers (1. / 1) / - / * / •)
        cleaned = _NUM_PREFIX_RE.sub("", line).strip()
        if not cleaned:
            continue
        # Skip if still looks like a heading (ends with colon)
        if cleaned.endswith(":") and len(cleaned.split()) <= 3:
            continue
        queries.append(cleaned)
        if len(queries) >= n_expected:
            break
    return queries


# ---------------------------------------------------------------------------
# Orchestration: load catalog, batch-generate with sentence-transformers/LLM,
# write parquet
# ---------------------------------------------------------------------------

def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Generate synthetic doc2query strings per catalog track.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="HF instruct-tuned LLM for query generation.")
    parser.add_argument("--catalog-dataset", default="talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
    parser.add_argument("--catalog-split", default="all_tracks")
    parser.add_argument("--n-queries", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=200,
                        help="Generation cap per track (5 queries × ~30 tokens + buffer).")
    parser.add_argument("--max-tracks", type=int, default=None,
                        help="Truncate to first N tracks (default: full catalog). Use for smoke runs.")
    parser.add_argument("--cache-root", default=str(REPO_ROOT / "experiments" / "cache" / "doc2query"))
    parser.add_argument("--resume", action="store_true",
                        help="If output parquet exists, skip tracks already in it.")
    args = parser.parse_args()

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm
    import pandas as pd

    safe_model = args.model.replace("/", "_")
    out_dir = Path(args.cache_root) / safe_model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "queries.parquet"

    # Resume support — load already-generated track_ids to skip.
    done_ids: set[str] = set()
    existing_rows = []
    if args.resume and out_path.exists():
        df = pd.read_parquet(out_path)
        done_ids = set(df["track_id"].tolist())
        existing_rows = df.to_dict(orient="records")
        print(f"[doc2query] resume: {len(done_ids)} tracks already generated, skipping", file=sys.stderr)

    print(f"[doc2query] loading {args.catalog_dataset}[{args.catalog_split}]", file=sys.stderr)
    ds = load_dataset(args.catalog_dataset, split=args.catalog_split)
    if args.max_tracks is not None:
        ds = ds.select(range(min(args.max_tracks, len(ds))))

    rows_to_generate = []
    for row in ds:
        if row["track_id"] in done_ids:
            continue
        rows_to_generate.append(row)
    print(f"[doc2query] generating for {len(rows_to_generate)} tracks", file=sys.stderr)

    if not rows_to_generate:
        print(f"[doc2query] nothing to do; output at {out_path}", file=sys.stderr)
        return

    print(f"[doc2query] loading {args.model}", file=sys.stderr)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    print(f"[doc2query] device={device} dtype={dtype}", file=sys.stderr)

    new_rows: list[dict] = []
    flush_every = 500

    def _flush():
        all_rows = existing_rows + new_rows
        df = pd.DataFrame(all_rows)
        tmp = str(out_path) + ".tmp"
        df.to_parquet(tmp, index=False)
        os.replace(tmp, out_path)

    for i in tqdm(range(0, len(rows_to_generate), args.batch_size), desc="doc2query"):
        batch = rows_to_generate[i:i + args.batch_size]
        prompts = [build_doc2query_prompt(r, n_queries=args.n_queries) for r in batch]
        # Use the chat template (instruct-tuned models expect it).
        chat_msgs = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True,
            )
            for p in prompts
        ]
        enc = tokenizer(chat_msgs, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=True, temperature=0.8, top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = out[:, enc["input_ids"].shape[1]:]
        completions = tokenizer.batch_decode(generated, skip_special_tokens=True)

        for row, completion in zip(batch, completions):
            queries = parse_generated_queries(completion, n_expected=args.n_queries)
            new_rows.append({"track_id": row["track_id"], "synthetic_queries": queries})

        # Periodic flush so a crash doesn't lose hours of work.
        if len(new_rows) % flush_every == 0:
            _flush()

    _flush()
    print(f"[doc2query] wrote {len(existing_rows) + len(new_rows)} rows to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
