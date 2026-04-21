#!/usr/bin/env bash
# Download all four HuggingFace datasets used by the pipeline into a
# repo-local cache directory.
#
# Why repo-local? Keeps data co-located with the code so the repo is fully
# reproducible: clone, run this script, run the pipeline. Avoids cluttering
# ~/.cache/huggingface and lets you delete everything by removing the repo
# directory.
#
# Usage:
#   ./download_data.sh                          # all four datasets, default path
#   HF_HOME=/some/other/path ./download_data.sh # use a different cache dir
#   ./download_data.sh --only blind             # one dataset only (smoke test)
#   HF_TOKEN=hf_... ./download_data.sh          # auth (avoids rate limits)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Configuration -----------------------------------------------------------

# Repo-local HF cache. All four datasets land under here.
# Override by exporting HF_HOME before invocation.
export HF_HOME="${HF_HOME:-$ROOT_DIR/data/hf_cache}"

# Venv to activate. Override if you've named your venv differently.
VENV_DIR="${VENV_DIR:-$ROOT_DIR/recsys26-lora}"

# --- Sanity check ------------------------------------------------------------

if [[ ! -d "$VENV_DIR" ]]; then
    echo "ERROR: venv not found at $VENV_DIR"
    echo "Run ./setup_venv.sh first, or export VENV_DIR to point at an existing venv."
    exit 1
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

# --- Show what's about to happen --------------------------------------------

echo "============================================================"
echo " Downloading TalkPlay HuggingFace datasets"
echo "============================================================"
printf "  %-15s %s\n" "HF_HOME"  "$HF_HOME"
printf "  %-15s %s\n" "venv"     "$VENV_DIR"
printf "  %-15s %s\n" "python"   "$(which python)"
if [[ -n "${HF_TOKEN:-}" ]]; then
    printf "  %-15s %s\n" "auth"   "HF_TOKEN set (faster, no rate limits)"
else
    printf "  %-15s %s\n" "auth"   "anonymous (slower; export HF_TOKEN to fix)"
fi
echo

mkdir -p "$HF_HOME"

# --- Download ---------------------------------------------------------------

cd "$ROOT_DIR"
python download_data.py "$@"

# --- Report ------------------------------------------------------------------

echo
echo "============================================================"
echo " Done. Cache contents:"
echo "============================================================"
if command -v du >/dev/null 2>&1; then
    du -sh "$HF_HOME" 2>/dev/null || true
fi
echo "  Datasets are at: $HF_HOME"
echo
echo "Tip: keep using this HF_HOME for the pipeline:"
echo "  HF_HOME=$HF_HOME ./run_pipeline.sh"
echo "(run_pipeline.sh defaults to the same path, so this is automatic.)"
