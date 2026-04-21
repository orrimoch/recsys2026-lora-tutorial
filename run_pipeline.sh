#!/usr/bin/env bash
# End-to-end LoRA fine-tune + Blind-A inference + prediction.zip packaging.
#
# Stages:
#   1. (optional) Bootstrap venv + install deps.
#   2. Build the LoRA SFT dataset from the HF train split (with v10-aligned filters).
#   3. Fine-tune Qwen 2.5-3B with PEFT-LoRA on the SFT dataset.
#   4. Run inference on Blind-A (BM25 retrieval + Qwen+LoRA generation).
#   5. Package + validate prediction.zip for CodaBench upload.
#
# Skip stages with the SKIP_* environment variables (e.g. SKIP_VENV=1 SKIP_DATASET=1).
# Override any hyperparam by exporting it before invoking this script.
#
# Usage:
#   ./run_pipeline.sh                  # full pipeline, defaults
#   SKIP_VENV=1 ./run_pipeline.sh      # skip venv setup
#   NUM_EPOCHS=2 LR=5e-5 ./run_pipeline.sh
#   SUBSET=3 ./run_pipeline.sh         # smoke: only 3 Blind-A rows in step 4

set -euo pipefail

# =============================================================================
#                              HYPERPARAMETERS
# =============================================================================

# --- Environment ---
VENV_DIR="${VENV_DIR:-recsys26-lora}"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"

# --- Paths ---
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Repo-local HF cache so datasets/models live with the code (not in $HOME).
# Both download_data.sh and run_pipeline.sh default to this path; override
# by exporting HF_HOME before running either script.
export HF_HOME="${HF_HOME:-$ROOT_DIR/data/hf_cache}"
SFT_DATASET_DIR="${SFT_DATASET_DIR:-$ROOT_DIR/data/train_sft}"
ADAPTER_DIR="${ADAPTER_DIR:-$ROOT_DIR/lora_adapters/qwen3b_blinda_v1}"
PROMPTS_DIR="${PROMPTS_DIR:-$ROOT_DIR/prompts}"
BM25_CACHE="${BM25_CACHE:-$ROOT_DIR/cache/bm25}"
PREDICTIONS_JSON="${PREDICTIONS_JSON:-$ROOT_DIR/output/predictions.json}"
PREDICTION_ZIP="${PREDICTION_ZIP:-$ROOT_DIR/output/prediction.zip}"

# --- Model ---
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"

# --- Training hyperparams ---
NUM_EPOCHS="${NUM_EPOCHS:-1}"
LR="${LR:-1e-4}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-1}"        # bump to 2-4 on A100
GRAD_ACCUM="${GRAD_ACCUM:-16}"                    # effective batch = PER_DEVICE * GRAD_ACCUM
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
TRAIN_MAX_LENGTH="${TRAIN_MAX_LENGTH:-2048}"      # M4: 2048 / A100: 3072
EVAL_SPLIT_FRAC="${EVAL_SPLIT_FRAC:-0.02}"
TARGET_MODULES="${TARGET_MODULES:-attn}"          # attn | attn_mlp
SEED="${SEED:-42}"

# --- Inference hyperparams ---
DEVICE="${DEVICE:-}"                              # auto-detect if empty
INFER_MAX_NEW_TOKENS="${INFER_MAX_NEW_TOKENS:-192}"
INFER_MAX_INPUT_LEN="${INFER_MAX_INPUT_LEN:-3072}"
LM_BATCH_SIZE="${LM_BATCH_SIZE:-1}"
SUBSET="${SUBSET:-}"                              # empty = all 80 Blind-A rows

# --- Skip switches (set to 1 to skip a stage) ---
SKIP_VENV="${SKIP_VENV:-0}"
SKIP_DATASET="${SKIP_DATASET:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_INFERENCE="${SKIP_INFERENCE:-0}"
SKIP_ZIP="${SKIP_ZIP:-0}"

# =============================================================================
#                              PIPELINE
# =============================================================================

cd "$ROOT_DIR"

print_section() {
    echo ""
    echo "============================================================"
    echo " $1"
    echo "============================================================"
}

print_kv() {
    printf "  %-22s %s\n" "$1" "$2"
}

# --- Stage 1: venv ---
if [[ "$SKIP_VENV" != "1" ]]; then
    print_section "STAGE 1/5 — venv setup"
    print_kv "venv dir"   "$VENV_DIR"
    print_kv "python"     "$PYTHON_BIN"
    if [[ ! -d "$VENV_DIR" ]]; then
        bash setup_venv.sh "$VENV_DIR"
    else
        echo "  venv exists at $VENV_DIR — skipping setup (delete to rebuild)"
    fi
fi

# Activate venv if it exists. On Colab/CI we typically use the system Python
# directly with SKIP_VENV=1 — no venv to source.
if [[ -f "$VENV_DIR/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "$VENV_DIR/bin/activate"
    echo "  active python: $(which python) ($(python --version)) [via $VENV_DIR]"
else
    echo "  no venv at $VENV_DIR; using system python: $(which python) ($(python --version))"
fi

# --- Stage 2: dataset ---
if [[ "$SKIP_DATASET" != "1" ]]; then
    print_section "STAGE 2/5 — build SFT dataset"
    print_kv "output dir" "$SFT_DATASET_DIR"
    print_kv "prompts"    "$PROMPTS_DIR"
    python lora/build_train_dataset.py --output "$SFT_DATASET_DIR"
fi

# --- Stage 3: training ---
if [[ "$SKIP_TRAIN" != "1" ]]; then
    print_section "STAGE 3/5 — LoRA fine-tune"
    print_kv "base model"        "$BASE_MODEL"
    print_kv "dataset"           "$SFT_DATASET_DIR"
    print_kv "adapter out"       "$ADAPTER_DIR"
    print_kv "epochs"            "$NUM_EPOCHS"
    print_kv "lr"                "$LR"
    print_kv "batch / grad_acc"  "$PER_DEVICE_BATCH x $GRAD_ACCUM (eff $((PER_DEVICE_BATCH * GRAD_ACCUM)))"
    print_kv "lora r/alpha"      "$LORA_R / $LORA_ALPHA  dropout=$LORA_DROPOUT"
    print_kv "target_modules"    "$TARGET_MODULES"
    print_kv "max_length"        "$TRAIN_MAX_LENGTH"
    print_kv "eval split"        "$EVAL_SPLIT_FRAC"
    python lora/train_lora.py \
        --dataset_path "$SFT_DATASET_DIR" \
        --output_dir "$ADAPTER_DIR" \
        --base_model "$BASE_MODEL" \
        --num_epochs "$NUM_EPOCHS" \
        --lr "$LR" \
        --per_device_batch "$PER_DEVICE_BATCH" \
        --grad_accum "$GRAD_ACCUM" \
        --lora_r "$LORA_R" \
        --lora_alpha "$LORA_ALPHA" \
        --lora_dropout "$LORA_DROPOUT" \
        --max_length "$TRAIN_MAX_LENGTH" \
        --eval_split_frac "$EVAL_SPLIT_FRAC" \
        --target_modules "$TARGET_MODULES" \
        --seed "$SEED"
fi

# --- Stage 4: inference ---
if [[ "$SKIP_INFERENCE" != "1" ]]; then
    print_section "STAGE 4/5 — Blind-A inference"
    ADAPTER_FINAL="$ADAPTER_DIR/final_adapter"
    if [[ ! -d "$ADAPTER_FINAL" ]]; then
        echo "  ERROR: adapter not found at $ADAPTER_FINAL"
        echo "  Run training first (or set SKIP_TRAIN=0)."
        exit 1
    fi
    print_kv "adapter"      "$ADAPTER_FINAL"
    print_kv "predictions"  "$PREDICTIONS_JSON"
    print_kv "device"       "${DEVICE:-auto}"
    [[ -n "$SUBSET" ]] && print_kv "subset (smoke)" "$SUBSET"
    INFER_EXTRA=()
    [[ -n "$DEVICE" ]] && INFER_EXTRA+=(--device "$DEVICE")
    [[ -n "$SUBSET" ]] && INFER_EXTRA+=(--subset "$SUBSET")
    python inference/run_inference.py \
        --output "$PREDICTIONS_JSON" \
        --base_model "$BASE_MODEL" \
        --lora_adapter_path "$ADAPTER_FINAL" \
        --prompts_dir "$PROMPTS_DIR" \
        --bm25_cache "$BM25_CACHE" \
        --max_new_tokens "$INFER_MAX_NEW_TOKENS" \
        --max_input_len "$INFER_MAX_INPUT_LEN" \
        --lm_batch_size "$LM_BATCH_SIZE" \
        ${INFER_EXTRA[@]+"${INFER_EXTRA[@]}"}
fi

# --- Stage 5: package ---
if [[ "$SKIP_ZIP" != "1" ]]; then
    print_section "STAGE 5/5 — package prediction.zip"
    print_kv "input json"   "$PREDICTIONS_JSON"
    print_kv "output zip"   "$PREDICTION_ZIP"
    if [[ -n "$SUBSET" ]]; then
        echo "  WARNING: SUBSET=$SUBSET means JSON has fewer than 80 rows."
        echo "  Validation will fail. Skipping zip step."
    else
        python inference/make_prediction_zip.py \
            --input "$PREDICTIONS_JSON" \
            --output "$PREDICTION_ZIP"
    fi
fi

print_section "Pipeline complete"
echo "  Predictions: $PREDICTIONS_JSON"
[[ -f "$PREDICTION_ZIP" ]] && echo "  Upload this:  $PREDICTION_ZIP"
