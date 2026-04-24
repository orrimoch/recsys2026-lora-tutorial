#!/bin/bash
# Run baseline models for RecSys 2026 Challenge
# Usage: ./run_baselines.sh [model] [batch_size]
#   model: random | popularity | bm25 | bert | bm25_blind | bert_blind | all_local | all_gpu
#   batch_size: (default 16, only used for bm25/bert)

set -e
cd "$(dirname "$0")"

MODEL="${1:-help}"
BATCH_SIZE="${2:-16}"

run_random() {
    echo "=== Running Random Baseline ==="
    python lowerbound/random_sample.py
    echo "Output: exp/inference/random.json"
}

run_popularity() {
    echo "=== Running Popularity Baseline ==="
    python lowerbound/popularity.py
    echo "Output: exp/inference/popularity.json"
}

run_bm25_devset() {
    echo "=== Running LLaMA-1B + BM25 (devset) ==="
    python run_inference_devset.py --tid llama1b_bm25_devset --batch_size "$BATCH_SIZE"
    echo "Output: exp/inference/devset/llama1b_bm25_devset.json"
}

run_bert_devset() {
    echo "=== Running LLaMA-1B + BERT (devset) ==="
    python run_inference_devset.py --tid llama1b_bert_devset --batch_size "$BATCH_SIZE"
    echo "Output: exp/inference/devset/llama1b_bert_devset.json"
}

run_bm25_blind() {
    echo "=== Running LLaMA-1B + BM25 (blind A) ==="
    python run_inference_blindset.py --tid llama1b_bm25_blindset_A --batch_size "$BATCH_SIZE" --eval_dataset blindset_A
    echo "Output: exp/inference/blindset_A/llama1b_bm25_blindset_A.json"
}

run_bert_blind() {
    echo "=== Running LLaMA-1B + BERT (blind A) ==="
    python run_inference_blindset.py --tid llama1b_bert_blindset_A --batch_size "$BATCH_SIZE" --eval_dataset blindset_A
    echo "Output: exp/inference/blindset_A/llama1b_bert_blindset_A.json"
}

case "$MODEL" in
    random)       run_random ;;
    popularity)   run_popularity ;;
    bm25)         run_bm25_devset ;;
    bert)         run_bert_devset ;;
    bm25_blind)   run_bm25_blind ;;
    bert_blind)   run_bert_blind ;;
    all_local)
        run_random
        run_popularity
        ;;
    all_gpu)
        run_bm25_devset
        run_bert_devset
        ;;
    *)
        echo "Usage: ./run_baselines.sh [model] [batch_size]"
        echo ""
        echo "Local (no GPU):"
        echo "  random       - Random track sampling"
        echo "  popularity   - Top-20 most popular tracks"
        echo "  all_local    - Run all local baselines"
        echo ""
        echo "GPU required:"
        echo "  bm25         - LLaMA-1B + BM25 (devset)"
        echo "  bert         - LLaMA-1B + BERT (devset)"
        echo "  bm25_blind   - LLaMA-1B + BM25 (blind A)"
        echo "  bert_blind   - LLaMA-1B + BERT (blind A)"
        echo "  all_gpu      - Run all GPU baselines on devset"
        echo ""
        echo "batch_size: default 16 (reduce if OOM)"
        ;;
esac
