"""Pre-fetch all four HuggingFace datasets used by the pipeline.

The full pipeline pulls these lazily on first use, but pre-fetching is useful
for:
  - verifying HF connectivity before kicking off a long training run
  - warming the cache on a fresh machine
  - downloading on a fast network so the slow training run isn't gated by
    network speed

All datasets are PUBLIC under the talkpl-ai org. Set HF_TOKEN to avoid the
unauthenticated-rate-limit warning.

CLI:
    python download_data.py                # download all four
    python download_data.py --only blind   # download only the Blind-A test set
"""
from __future__ import annotations

import argparse
import os
import sys

from datasets import load_dataset

DATASETS = {
    "train": ("talkpl-ai/TalkPlayData-Challenge-Dataset", "train"),
    "tracks": ("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", None),
    "users": ("talkpl-ai/TalkPlayData-Challenge-User-Metadata", None),
    "blind": ("talkpl-ai/TalkPlayData-Challenge-Blind-A", "test"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=list(DATASETS.keys()), default=None,
                        help="Download a single dataset (default: all four).")
    args = parser.parse_args()

    keys = [args.only] if args.only else list(DATASETS.keys())

    if not os.environ.get("HF_TOKEN"):
        print("[warn] HF_TOKEN not set — downloads will be slower (rate-limited). "
              "Get a token at https://huggingface.co/settings/tokens",
              file=sys.stderr)

    for key in keys:
        repo, split = DATASETS[key]
        target = f"{repo}" + (f" [split={split}]" if split else " [all splits]")
        print(f"[download] {key:7s} {target}", file=sys.stderr)
        ds = load_dataset(repo, split=split) if split else load_dataset(repo)
        if hasattr(ds, "num_rows"):
            print(f"           rows: {ds.num_rows}", file=sys.stderr)
        else:
            for sp_name, sp in ds.items():
                print(f"           split={sp_name}: {sp.num_rows} rows", file=sys.stderr)

    cache_root = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    print(f"[download] done — datasets cached under {cache_root}", file=sys.stderr)


if __name__ == "__main__":
    main()
