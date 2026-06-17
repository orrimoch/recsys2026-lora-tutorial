"""
Measure real enriched-doc token lengths from the A1 parquet corpus and
recommend a max_length for the K3b cross-encoder fine-tune.

Resolves the open decision in .claude/documents/features/53_K3b_ce_lora_finetune.md §4.2 (m11):
    Re-measure on the real A1 corpus — if real-doc-p99 + query-p99 ≈ 1366, drop to 1536 (cheaper).
"""
from __future__ import annotations

import argparse
import glob
import os
from typing import Tuple


# ---------------------------------------------------------------------------
# Pure helper — no I/O, fully testable offline
# ---------------------------------------------------------------------------

def recommend_max_length(
    query_p99: int,
    doc_p99: int,
    *,
    specials: int = 4,
    candidates: Tuple[int, ...] = (1024, 1536, 2048),
) -> int:
    """Return the smallest candidate that fits query_p99 + doc_p99 + specials.

    If no candidate is large enough (overflow), return the largest candidate
    (the doc will be truncated at serve time).

    Args:
        query_p99: 99th-percentile query token count (measured without special tokens).
        doc_p99:   99th-percentile doc token count (measured without special tokens).
        specials:  Number of special tokens added by the tokenizer (default 4: [CLS], [SEP] ×2, [SEP]).
        candidates: Tuple of candidate max_length values to consider (need not be pre-sorted; the function sorts internally).

    Returns:
        The recommended max_length.
    """
    needed = query_p99 + doc_p99 + specials
    for candidate in sorted(candidates):
        if needed <= candidate:
            return candidate
    # Overflow: return the largest candidate
    return max(candidates)


# ---------------------------------------------------------------------------
# Measurement function — loads parquet + tokenizer, returns percentile dict
# ---------------------------------------------------------------------------

def measure(
    parquet_glob: str,
    tokenizer_name: str = "BAAI/bge-reranker-v2-m3",
    sample: int = 20_000,
) -> dict:
    """Measure token-length statistics of enriched_doc texts in a parquet corpus.

    Loads the *newest* parquet file matching ``parquet_glob``, samples up to
    ``sample`` rows, tokenizes the ``enriched_doc`` column with the given
    tokenizer (add_special_tokens=False), and returns a stats dict.

    Args:
        parquet_glob:    Glob pattern for parquet files (e.g. 'outputs/catalog_enriched_*.parquet').
        tokenizer_name:  HuggingFace tokenizer/model name.
        sample:          Maximum rows to tokenize (random sample if file has more).

    Returns:
        dict with keys: p50, p90, p95, p99, max, mean, n  (all int/float).

    Raises:
        FileNotFoundError: If no parquet file matches the glob.
        KeyError:          If the parquet has no 'enriched_doc' column.
        OSError:           If the tokenizer cannot be downloaded (no network).
    """
    # --- Lazy imports so the module is importable without these deps installed ---
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required: pip install pandas") from exc

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("transformers is required: pip install transformers") from exc

    # Find newest matching parquet
    matches = sorted(glob.glob(parquet_glob))
    if not matches:
        raise FileNotFoundError(
            f"No parquet files found matching: {parquet_glob!r}. "
            "Run the enriched-catalog generator first (see scripts/generate_enriched_catalog.py)."
        )
    parquet_path = matches[-1]  # newest by lexicographic sort (filenames include timestamp)

    # Load the parquet — read all columns first so a missing 'enriched_doc'
    # column raises a clear KeyError rather than an opaque ArrowInvalid.
    df = pd.read_parquet(parquet_path)
    if "enriched_doc" not in df.columns:
        raise KeyError(
            f"Column 'enriched_doc' not found in {parquet_path!r}. "
            f"Available columns: {list(df.columns)}"
        )

    # Keep only the column we need to free memory before sampling
    df = df[["enriched_doc"]]

    # Sample
    if len(df) > sample:
        df = df.sample(n=sample, random_state=42)

    texts = df["enriched_doc"].dropna().tolist()
    if not texts:
        raise ValueError(
            f"No non-null enriched_doc values in {parquet_path!r}. "
            "All rows are null — cannot compute token-length statistics."
        )

    # Load tokenizer — raise a clear error if network is unavailable
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as exc:
        raise OSError(
            f"Failed to load tokenizer '{tokenizer_name}'. "
            "Check your network connection or pass a local path via --tokenizer-name. "
            f"Original error: {exc}"
        ) from exc

    # Tokenize in batches to keep memory reasonable
    import numpy as np

    lengths = []
    batch_size = 512
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        lengths.extend(len(ids) for ids in encoded["input_ids"])

    lengths = np.array(lengths, dtype=np.int32)
    return {
        "p50": int(np.percentile(lengths, 50)),
        "p90": int(np.percentile(lengths, 90)),
        "p95": int(np.percentile(lengths, 95)),
        "p99": int(np.percentile(lengths, 99)),
        "max": int(lengths.max()),
        "mean": float(lengths.mean()),
        "n": len(lengths),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure real enriched-doc token lengths and recommend max_length "
            "for the K3b cross-encoder fine-tune (resolves §4.2 m11 open decision)."
        )
    )
    parser.add_argument(
        "--parquet-glob",
        default="outputs/catalog_enriched_*.parquet",
        help="Glob pattern for enriched-catalog parquet files (default: outputs/catalog_enriched_*.parquet).",
    )
    parser.add_argument(
        "--tokenizer-name",
        default="BAAI/bge-reranker-v2-m3",
        help="HuggingFace tokenizer name or local path (default: BAAI/bge-reranker-v2-m3).",
    )
    parser.add_argument(
        "--query-p99",
        type=int,
        default=866,
        help=(
            "Query-side p99 token count from the spec measurement "
            "(default: 866, from 53_K3b_ce_lora_finetune.md §4.2)."
        ),
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=20_000,
        help="Maximum number of docs to sample for measurement (default: 20000).",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    print(f"Measuring enriched-doc token lengths from: {args.parquet_glob!r}")
    print(f"Tokenizer: {args.tokenizer_name}")
    print(f"Sample size: {args.sample:,}")
    print()

    stats = measure(
        parquet_glob=args.parquet_glob,
        tokenizer_name=args.tokenizer_name,
        sample=args.sample,
    )

    # Print percentile table
    print("Enriched-doc token length distribution (add_special_tokens=False):")
    print(f"  n    = {stats['n']:,}")
    print(f"  mean = {stats['mean']:.1f}")
    print(f"  p50  = {stats['p50']}")
    print(f"  p90  = {stats['p90']}")
    print(f"  p95  = {stats['p95']}")
    print(f"  p99  = {stats['p99']}")
    print(f"  max  = {stats['max']}")
    print()

    doc_p99 = stats["p99"]
    rec = recommend_max_length(args.query_p99, doc_p99)
    needed = args.query_p99 + doc_p99 + 4  # 4 specials

    print(f"Query p99  : {args.query_p99} tokens (from spec, 53_K3b §4.2)")
    print(f"Doc   p99  : {doc_p99} tokens (measured)")
    print(f"Total p99  : {needed} tokens (query + doc + 4 specials)")
    print()
    print(f"Recommended max_length: {rec}")
    if needed <= rec:
        print(f"  → p99 pair fits within {rec} with {rec - needed} tokens to spare.")
    else:
        print(f"  → p99 pair ({needed}) exceeds all candidates; doc will be truncated at serve time.")
        print(f"  → Using largest candidate: {rec}.")


if __name__ == "__main__":
    main()
