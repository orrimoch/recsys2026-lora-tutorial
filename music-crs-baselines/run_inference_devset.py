"""
Batch inference script for Music CRS.
"""

import os
import json
import torch
import argparse
from mcrs import load_crs_baseline
from datasets import load_dataset
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
import pandas as pd
from omegaconf import OmegaConf

def chat_history_parser(conversations, music_crs, target_turn_number):
    """
    Parse conversation history up to a target turn.

    Args:
        conversations (List[Dict]): List of conversation turn dictionaries containing:
            - turn_number: Turn index (1-8)
            - role: Speaker role ('user', 'assistant', 'music')
            - content: Message content or track ID
        music_crs: CRS baseline instance (used to convert track IDs to metadata)
        target_turn_number (int): The turn to predict (history excludes this turn)

    Returns:
        Tuple[List[Dict], str]:
            - chat_history: List of previous messages formatted as [{"role": ..., "content": ...}]
            - user_query: The user query at the target turn
    """
    df_conversation = pd.DataFrame(conversations)
    df_history = df_conversation[df_conversation['turn_number'] < target_turn_number]
    chat_history = []
    for turn_data in df_history.to_dict(orient="records"):
        turn_number = turn_data['turn_number']
        current_role = turn_data['role']
        current_content = turn_data['content']
        if turn_data['role'] == "music":
            current_role = "assistant"
            current_content = music_crs.item_db.id_to_metadata(turn_data['content'])
        chat_history.append({
            "role": current_role,
            "content": current_content
        })
    df_current_turn = df_conversation[df_conversation['turn_number'] == target_turn_number]
    user_query = df_current_turn.iloc[0]['content']
    return chat_history, user_query

def main(args):
    """
    Run batch inference on TalkPlayData-2 test dataset.

    Args:
        args: Namespace object containing:
            - tid (str): Task/configuration identifier
            - batch_size (int): Batch size for inference
            - save_path (str): Output directory (currently unused)

    Returns:
        None. Results are saved to exp/inference/{tid}.json

    Processing:
        - Loads model configuration from config/{tid}.yaml
        - Processes all sessions × 8 turns in batches
        - Tracks progress with tqdm progress bar
        - Saves comprehensive results for evaluation
    """
    print("Removing cache directory for preventing memory issues...")
    os.system("rm -rf cache")
    # Defensive: reclaim any leaked CUDA memory from prior subprocesses
    # before allocating. Colab subprocess exits don't always free GPU
    # cleanly; this gives us back what PyTorch can reach.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            free_b, total_b = torch.cuda.mem_get_info()
            print(f"[startup] CUDA free: {free_b/1024**3:.2f}/{total_b/1024**3:.2f} GiB")
        except Exception:
            pass
    config = OmegaConf.load(f"config/{args.tid}.yaml")
    device = args.device or config.device
    attn_implementation = args.attn_implementation or config.attn_implementation
    response_prompt_name = config.get("response_prompt_name", "response_generation")
    reranker_type = config.get("reranker_type", None)
    reranker_model_path = config.get("reranker_model_path", None)
    retrieval_topk = int(config.get("retrieval_topk", 20))
    response_max_new_tokens = int(config.get("response_max_new_tokens", 64))
    top_n_for_prompt = int(config.get("top_n_for_prompt", 1))
    query_preprocessing_mode = str(config.get("query_preprocessing_mode", "raw"))
    response_reranker_type = config.get("response_reranker_type", None)
    response_reranker_model_path = config.get("response_reranker_model_path", None)
    response_n_candidates = int(config.get("response_n_candidates", 3))
    _rt = config.get("response_temperatures", None)
    response_temperatures = [float(x) for x in _rt] if _rt is not None else None
    use_vllm = bool(config.get("use_vllm", False))
    # W4 P0 #3: optional PEFT/LoRA adapter for the responder LM (KTO/DPO/GRPO outputs)
    lora_path = config.get("lora_path", None)
    lora_max_rank = int(config.get("lora_max_rank", 32))
    # W1 StateTracker (Gap 1) — opt-in via config; default OFF preserves the
    # exp 021 path bit-exact. When enabled, batch results carry an
    # `extracted_state` field that W2 CMQR / W4 responder envelope can read.
    use_state_tracker = bool(config.get("use_state_tracker", False))
    state_tracker_prompt_name = str(config.get("state_tracker_prompt_name", "state_extraction"))
    state_tracker_max_new_tokens = int(config.get("state_tracker_max_new_tokens", 96))
    # W2 CMQR (multi-query rewriter) — opt-in via config; default OFF preserves
    # the exp 021 path. When enabled, the inner retriever is wrapped so each
    # query expands to N rewrites, fused via RRF.
    use_cmqr = bool(config.get("use_cmqr", False))
    cmqr_prompt_name = str(config.get("cmqr_prompt_name", "cmqr_rewrites"))
    cmqr_n_rewrites = int(config.get("cmqr_n_rewrites", 4))
    cmqr_topk_per_rewrite = int(config.get("cmqr_topk_per_rewrite", 50))
    cmqr_rrf_k = int(config.get("cmqr_rrf_k", 60))
    cmqr_max_new_tokens = int(config.get("cmqr_max_new_tokens", 96))
    music_crs = load_crs_baseline(
        lm_type=config.lm_type,
        retrieval_type=config.retrieval_type,
        item_db_name=config.item_db_name,
        user_db_name=config.user_db_name,
        track_split_types=config.track_split_types,
        user_split_types=config.user_split_types,
        corpus_types=config.corpus_types,
        cache_dir=config.cache_dir,
        device=device,
        attn_implementation=attn_implementation,
        dtype=torch.bfloat16,
        response_prompt_name=response_prompt_name,
        reranker_type=reranker_type,
        reranker_model_path=reranker_model_path,
        retrieval_topk=retrieval_topk,
        response_max_new_tokens=response_max_new_tokens,
        top_n_for_prompt=top_n_for_prompt,
        query_preprocessing_mode=query_preprocessing_mode,
        response_reranker_type=response_reranker_type,
        response_reranker_model_path=response_reranker_model_path,
        response_n_candidates=response_n_candidates,
        response_temperatures=response_temperatures,
        use_vllm=use_vllm,
        lora_path=lora_path,
        lora_max_rank=lora_max_rank,
        use_state_tracker=use_state_tracker,
        state_tracker_prompt_name=state_tracker_prompt_name,
        state_tracker_max_new_tokens=state_tracker_max_new_tokens,
        use_cmqr=use_cmqr,
        cmqr_prompt_name=cmqr_prompt_name,
        cmqr_n_rewrites=cmqr_n_rewrites,
        cmqr_topk_per_rewrite=cmqr_topk_per_rewrite,
        cmqr_rrf_k=cmqr_rrf_k,
        cmqr_max_new_tokens=cmqr_max_new_tokens,
    )
    db = load_dataset(config.test_dataset_name, split="test")
    if getattr(args, "subset", None) is not None:
        n = min(args.subset, len(db))
        db = db.select(range(n))
        print(f"[run_inference_devset] --subset {args.subset} → using first {n} sessions ({n*8} turns)")
    # Prepare all batch data at once
    batch_data, metadata = [], []
    for item in db:
        user_id = item['user_id']
        session_id = item['session_id']
        for target_turn_number in range(1, 9):
            chat_history, user_query = chat_history_parser(item['conversations'], music_crs, target_turn_number)
            batch_data.append({
                'user_query': user_query,
                'user_id': user_id,
                'session_memory': chat_history,
                # session_id + turn_number are needed by the StateTracker (Gap 1).
                # Always populated; CRS_BASELINE only consumes them when
                # use_state_tracker=True so this is a no-op for exp 021.
                'session_id': session_id,
                'turn_number': target_turn_number,
            })
            metadata.append({
                'session_id': session_id,
                'user_id': user_id,
                'turn_number': target_turn_number
            })

    # Length-bucket: sort by total input character length so similar-length
    # sequences batch together. Eliminates left-padding waste — without this,
    # a single 8-turn conversation in a batch forces the other 15 short queries
    # to pay the long-prefill cost. Output order doesn't matter for scoring;
    # session_id+user_id+turn_number in metadata identify each row.
    def _est_len(item):
        history_chars = sum(len(t.get('content', '') or '') for t in item['session_memory'])
        return history_chars + len(item.get('user_query', '') or '')
    paired = sorted(zip(batch_data, metadata), key=lambda bm: _est_len(bm[0]))
    batch_data = [b for b, _ in paired]
    metadata = [m for _, m in paired]

    inference_results = []
    # try/finally guarantees the score cache (W3 P0 #1) is persisted even on
    # KeyboardInterrupt / Colab session timeout / OOM mid-loop. Periodic save
    # every SAVE_EVERY_N batches makes the run resumable from partial state.
    SAVE_EVERY_N = 50
    try:
        for i in tqdm(range(0, len(batch_data), args.batch_size), desc="Batch inference"):
            batch = batch_data[i:i+args.batch_size]
            batch_metadata = metadata[i:i+args.batch_size]
            results = music_crs.batch_chat(batch)
            for j, result in enumerate(results):
                inference_results.append({
                    "session_id": batch_metadata[j]['session_id'],
                    "user_id": batch_metadata[j]['user_id'],
                    "turn_number": batch_metadata[j]['turn_number'],
                    "predicted_track_ids": result['retrieval_items'],
                    "predicted_response": result["response"]
                })
            # Periodic cache flush — turns 800k-fwd-pass run into a
            # resumable artifact (W3 review P1 #9).
            batch_idx = i // args.batch_size
            if batch_idx > 0 and batch_idx % SAVE_EVERY_N == 0:
                music_crs.save_caches()
    finally:
        music_crs.save_caches()
    os.makedirs("exp/inference/devset", exist_ok=True)
    with open(f"exp/inference/devset/{args.tid}.json", "w", encoding="utf-8") as f:
        json.dump(inference_results, f, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run batch inference on TalkPlayData-2 test dataset for Music CRS evaluation."
    )
    parser.add_argument(
        "--tid",
        type=str,
        default="llama1b_bm25_testset",
        help="Task identifier matching a config file (e.g., 'llama1b_bm25' loads config/llama1b_bm25.yaml)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of queries to process in parallel. Reduce if encountering GPU memory issues."
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./exp/inference",
        help="Base directory for saving results (currently not used, results saved to exp/inference/)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override config.device (e.g. 'cuda' on Colab, 'mps' on M-series Mac, 'cpu'). Defaults to config value."
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        choices=[None, "eager", "sdpa", "flash_attention_2"],
        help="Override config.attn_implementation. On CUDA use 'sdpa' for ~40x less attention memory; "
             "on MPS stay on 'eager' (our memory has 'MPS + sdpa' dead-ends). Defaults to config value."
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Truncate the dev set to first N sessions (default: full 1000). "
             "Useful for T4 smoke runs before committing the ~3-6 hr full pipeline.",
    )
    args = parser.parse_args()
    main(args)
