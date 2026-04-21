"""End-to-end Blind-A inference: BM25 retrieval + Qwen 2.5-3B + optional LoRA.

Self-contained — no dependency on the larger music-crs-baselines repo. Reads
the test set from HuggingFace, builds a BM25 index over the catalog, generates
responses with the v10 champion prompt template, and writes the JSON the
challenge expects.

Pair with `make_prediction_zip.py` to get the final upload artifact.

CLI:
    python run_inference.py \
        --output ./output/predictions.json \
        --base_model Qwen/Qwen2.5-3B-Instruct \
        --lora_adapter_path ./lora_adapters/qwen3b_blinda_v1/final_adapter \
        --prompts_dir ./prompts \
        --bm25_cache ./cache/bm25 \
        --device mps
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import bm25s
import torch
from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

CORPUS_FIELDS = ["track_name", "artist_name", "album_name", "release_date"]
RETRIEVAL_TOPK = 40
FINAL_K = 20
TOP_N_FOR_LLM = 3


# ---------- BM25 retrieval ----------------------------------------------------

def stringify_metadata(meta: dict, fields: list[str]) -> str:
    out = []
    for f in fields:
        v = meta.get(f)
        if isinstance(v, list):
            v = ", ".join(v)
        out.append(f"{f}: {v}")
    return "\n".join(out)


def build_or_load_bm25(track_meta_dict: dict, cache_dir: str) -> tuple[bm25s.BM25, list[str]]:
    """Build BM25 index over the catalog, or reload from disk cache."""
    corpus_name = "_".join(CORPUS_FIELDS)
    index_dir = os.path.join(cache_dir, corpus_name)
    ids_path = os.path.join(index_dir, "track_ids.json")
    if os.path.exists(ids_path):
        print(f"[bm25] loading cached index from {index_dir}", file=sys.stderr)
        retriever = bm25s.BM25.load(index_dir, load_corpus=True)
        track_ids = json.load(open(ids_path))
        return retriever, track_ids

    print(f"[bm25] building index over {len(track_meta_dict)} tracks (this is a one-time cost)",
          file=sys.stderr)
    track_ids = list(track_meta_dict.keys())
    corpus = [stringify_metadata(track_meta_dict[tid], CORPUS_FIELDS) for tid in track_ids]
    corpus_tokens = bm25s.tokenize(corpus)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    os.makedirs(index_dir, exist_ok=True)
    retriever.save(index_dir, corpus=corpus)
    with open(ids_path, "w") as f:
        json.dump(track_ids, f)
    return retriever, track_ids


def bm25_retrieve(retriever, track_ids, queries: list[str], topk: int) -> list[list[str]]:
    """Batched query against the BM25 index. Handles both modes of bm25s
    output: dicts (when retriever was reloaded with corpus) or scalar
    indices (when retriever is the fresh build)."""
    query_tokens = bm25s.tokenize([q.lower() for q in queries])
    scores = retriever.retrieve(query_tokens, k=topk, return_as="tuple")
    out = []
    for i in range(len(queries)):
        hits = scores.documents[i]
        row = []
        for h in hits:
            if isinstance(h, dict):
                row.append(track_ids[h["id"]])
            else:
                row.append(track_ids[int(h)])
        out.append(row)
    return out


# ---------- Prompt building (matches v10 champion) ---------------------------

def first(v: Any) -> str:
    if isinstance(v, list):
        return v[0] if v else ""
    return v or ""


def format_track_for_prompt(meta: dict) -> str:
    name = first(meta.get("track_name"))
    artist = first(meta.get("artist_name"))
    album = first(meta.get("album_name"))
    release = first(meta.get("release_date"))
    year = release.split("-")[0] if release else ""
    tags = meta.get("tag_list") or []
    tag_str = ", ".join(tags[:5]) if isinstance(tags, list) else str(tags)
    parts = [f'"{name}"']
    if artist:
        parts.append(f"by {artist}")
    tail = [t for t in (album, year) if t]
    if tail:
        parts.append(f"({', '.join(tail)})")
    if tag_str:
        parts.append(f"[tags: {tag_str}]")
    return " ".join(parts)


def build_user_profile_str(user_profile_blob: str | None,
                           user_meta: dict | None) -> str:
    bits = []
    if user_meta:
        demo = [f"{k}={user_meta.get(k)}" for k in ("age", "country", "gender") if user_meta.get(k)]
        if demo:
            bits.append(f"User profile (demographics): {', '.join(demo)}")
    if user_profile_blob:
        bits.append(f"Additional user context: {user_profile_blob}")
    return "\n".join(bits)


def build_sys_prompt(roleplay: str, response_gen: str,
                     user_profile_blob: str | None, user_meta: dict | None,
                     conversation_goal: Any, top_tracks: list[dict]) -> str:
    sections = [roleplay.strip(), response_gen.strip()]

    person_bits = []
    prof = build_user_profile_str(user_profile_blob, user_meta)
    if prof:
        person_bits.append(prof)
    if conversation_goal:
        goal = conversation_goal.get("listener_goal", "") if isinstance(conversation_goal, dict) else str(conversation_goal)
        if goal:
            person_bits.append(f"Conversation goal: {goal}")
    if person_bits:
        sections.append("=== About this user ===\n" + "\n".join(person_bits))

    track_lines = [f"{i+1}. {format_track_for_prompt(m)}" for i, m in enumerate(top_tracks)]
    sections.append(
        "=== Candidate tracks (ranked) ===\n"
        "These are the tracks you have selected. Recommend #1 as the primary choice. "
        "You may optionally reference #2 or #3 as alternatives.\n\n"
        + "\n".join(track_lines)
    )
    return "\n\n".join(sections)


# ---------- LLM generation ---------------------------------------------------

def custom_batch_generate(model, tokenizer, sys_prompts: list[str],
                          user_queries: list[str], device: str,
                          max_new_tokens: int, max_input_len: int) -> list[str]:
    """[system, user] chat template only. Avoids the fake-assistant turn that
    causes 'Glad you enjoyed X' hallucinations on cold-start queries."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    formatted = []
    for sp, uq in zip(sys_prompts, user_queries):
        chat = [
            {"role": "system", "content": sp},
            {"role": "user", "content": uq},
        ]
        formatted.append(tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True))

    tok = tokenizer(formatted, return_tensors="pt", padding=True,
                    truncation=True, max_length=max_input_len)
    input_ids = tok.input_ids.to(device)
    attention_mask = tok.attention_mask.to(device)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
        )
    return tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)


# ---------- Main pipeline ----------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True,
                        help="Path to write predictions JSON.")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--lora_adapter_path", type=str, default=None,
                        help="Optional. Skip for v10-baseline-style (no fine-tune).")
    parser.add_argument("--prompts_dir", type=str, default="./prompts")
    parser.add_argument("--bm25_cache", type=str, default="./cache/bm25")
    parser.add_argument("--test_dataset", type=str,
                        default="talkpl-ai/TalkPlayData-Challenge-Blind-A")
    parser.add_argument("--track_meta_dataset", type=str,
                        default="talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
    parser.add_argument("--user_meta_dataset", type=str,
                        default="talkpl-ai/TalkPlayData-Challenge-User-Metadata")
    parser.add_argument("--device", type=str, default=None,
                        help="Auto-detect if omitted (cuda > mps > cpu).")
    parser.add_argument("--max_new_tokens", type=int, default=192)
    parser.add_argument("--max_input_len", type=int, default=3072)
    parser.add_argument("--lm_batch_size", type=int, default=1)
    parser.add_argument("--retrieval_batch_size", type=int, default=64)
    parser.add_argument("--subset", type=int, default=None,
                        help="Optional row cap for smoke testing.")
    args = parser.parse_args()

    # Device + dtype.
    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    print(f"[infer] device={args.device} dtype={dtype}", file=sys.stderr)

    # Load datasets.
    print("[infer] loading test set", file=sys.stderr)
    test = load_dataset(args.test_dataset, split="test")
    if args.subset:
        test = test.select(range(min(args.subset, len(test))))

    print("[infer] loading track metadata", file=sys.stderr)
    track_meta_ds = load_dataset(args.track_meta_dataset)
    track_meta_concat = concatenate_datasets([track_meta_ds[s] for s in track_meta_ds.keys()])
    track_meta_dict = {r["track_id"]: r for r in track_meta_concat}

    print("[infer] loading user metadata", file=sys.stderr)
    try:
        user_meta_ds = load_dataset(args.user_meta_dataset)
        user_meta_concat = concatenate_datasets([user_meta_ds[s] for s in user_meta_ds.keys()])
        user_meta_dict = {r["user_id"]: r for r in user_meta_concat}
    except Exception as e:
        print(f"[infer] user metadata unavailable ({e}); skipping demographics",
              file=sys.stderr)
        user_meta_dict = {}

    # BM25 index.
    retriever, track_ids = build_or_load_bm25(track_meta_dict, args.bm25_cache)

    # Prompts.
    roleplay = open(os.path.join(args.prompts_dir, "roleplay.txt"), encoding="utf-8").read()
    response_gen = open(os.path.join(args.prompts_dir, "response_generation_v4.txt"), encoding="utf-8").read()

    # Model.
    print(f"[infer] loading {args.base_model}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, low_cpu_mem_usage=True,
    )
    if args.lora_adapter_path:
        if not os.path.isdir(args.lora_adapter_path):
            raise FileNotFoundError(f"adapter not found: {args.lora_adapter_path}")
        print(f"[infer] applying LoRA adapter from {args.lora_adapter_path}",
              file=sys.stderr)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_adapter_path)
        model = model.merge_and_unload()
    model = model.to(args.device).to(dtype)
    model.eval()
    if hasattr(model, "generation_config"):
        model.generation_config.do_sample = False

    # Build per-row inputs.
    metadata_rows = []
    user_queries = []
    item_rows = []
    for item in test:
        user_queries.append(item["conversations"][-1]["content"])
        metadata_rows.append({
            "session_id": item["session_id"],
            "user_id": item["user_id"],
            "turn_number": item["conversations"][-1]["turn_number"],
        })
        item_rows.append(item)

    # Retrieval.
    all_picks = []
    for i in tqdm(range(0, len(user_queries), args.retrieval_batch_size), desc="bm25"):
        batch = user_queries[i:i + args.retrieval_batch_size]
        for hits in bm25_retrieve(retriever, track_ids, batch, RETRIEVAL_TOPK):
            picked = list(dict.fromkeys(hits))[:FINAL_K]
            all_picks.append(picked)

    # Build sys prompts.
    sys_prompts = []
    for idx, picks in enumerate(all_picks):
        top_metas = [track_meta_dict[t] for t in picks[:TOP_N_FOR_LLM] if t in track_meta_dict]
        sp = build_sys_prompt(
            roleplay=roleplay,
            response_gen=response_gen,
            user_profile_blob=item_rows[idx].get("user_profile") or "",
            user_meta=user_meta_dict.get(item_rows[idx].get("user_id")),
            conversation_goal=item_rows[idx].get("conversation_goal"),
            top_tracks=top_metas,
        )
        sys_prompts.append(sp)

    # Generate.
    all_responses = []
    for i in tqdm(range(0, len(sys_prompts), args.lm_batch_size), desc="generate"):
        bsp = sys_prompts[i:i + args.lm_batch_size]
        buq = user_queries[i:i + args.lm_batch_size]
        outs = custom_batch_generate(
            model, tokenizer, bsp, buq, device=args.device,
            max_new_tokens=args.max_new_tokens, max_input_len=args.max_input_len,
        )
        all_responses.extend(r.strip() for r in outs)

    # Assemble + write.
    results = []
    for i, meta in enumerate(metadata_rows):
        results.append({
            "session_id": meta["session_id"],
            "user_id": meta["user_id"],
            "turn_number": meta["turn_number"],
            "predicted_track_ids": all_picks[i],
            "predicted_response": all_responses[i] or "Here are some songs you might enjoy.",
        })

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    print(f"[infer] wrote {len(results)} predictions to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
