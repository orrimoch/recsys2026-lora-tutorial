"""End-to-end Blind-A inference: BM25 retrieval + Qwen 2.5-3B + optional LoRA.

This script reproduces the v10 champion pipeline EXACTLY, with an optional
LoRA adapter wrapped around the base model. Pair with `make_prediction_zip.py`
to package the upload artifact.

Behavior parity with v10 (`music-crs-baselines/run_inference_blindset_full_v6.py`)
verified as of 2026-04-21. Notable v10 invariants this preserves:
  - BM25 input is the FULL conversation history formatted as "role: content"
    (not just the last user turn). Blind-A has up to 13 turns per row.
  - Music-role turns in history are expanded to "track_id: ..., track_name: ..."
    via the same MusicCatalogDB.id_to_metadata stringification.
  - User profile uses HF columns: user_id, age_group, gender, country_name.
  - conversation_goal is rendered with str() — the raw dict gets repr'd into
    the prompt (matches v10's f-string behavior).
  - track_split_types is exactly ["all_tracks"] (the catalog), not test splits.

CLI:
    python run_inference.py --output ./output/predictions.json
    python run_inference.py --output ./output/predictions.json \\
        --lora_adapter_path ./lora_adapters/qwen3b_blinda_v1/final_adapter
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


# ---------- Catalog + user databases (match v10's mcrs.db_item / mcrs.db_user) ----

def load_track_catalog(dataset_name: str, split: str = "all_tracks") -> dict[str, dict]:
    """Catalog of all tracks. v10 uses ONLY the all_tracks split — not the
    catalog-test split. Test split is for evaluation purposes only."""
    ds = load_dataset(dataset_name, split=split)
    return {r["track_id"]: r for r in ds}


def load_user_profiles(dataset_name: str, split: str = "all_users") -> dict[str, dict]:
    """User-id → demographics map."""
    try:
        ds = load_dataset(dataset_name, split=split)
        return {r["user_id"]: r for r in ds}
    except Exception as e:
        print(f"[infer] user metadata unavailable ({e}); skipping demographics", file=sys.stderr)
        return {}


def id_to_profile_str(user_profiles: dict, user_id: str) -> str:
    """Match v10's UserProfileDB.id_to_profile_str: prints user_id + age_group
    + gender + country_name, one per line."""
    if user_id not in user_profiles:
        return ""
    user = user_profiles[user_id]
    cols = ["user_id", "age_group", "gender", "country_name"]
    return "\n".join(f"{k}: {user.get(k)}" for k in cols)


def id_to_metadata_str(track_meta: dict, track_id: str, corpus_types: list[str]) -> str:
    """Match v10's MusicCatalogDB.id_to_metadata exactly: lowercased,
    comma-separated 'field: value' string starting with track_id."""
    meta = track_meta[track_id]
    parts = [f"track_id: {track_id}"]
    for ct in corpus_types:
        v = meta[ct]
        if isinstance(v, list):
            v = ", ".join(v).lower()
        else:
            v = str(v).lower()
        parts.append(f"{ct}: {v}")
    return ", ".join(parts)


# ---------- BM25 ------------------------------------------------------------

def stringify_metadata_for_bm25(meta: dict, fields: list[str]) -> str:
    """Match v10's BM25_MODEL._stringify_metadata exactly."""
    out = ""
    for f in fields:
        v = meta[f]
        if isinstance(v, list):
            v = ", ".join(v)
        out += f"{f}: {v}\n"
    return out


def build_or_load_bm25(track_meta: dict, cache_dir: str) -> tuple[bm25s.BM25, list[str]]:
    corpus_name = "_".join(CORPUS_FIELDS)
    index_dir = os.path.join(cache_dir, corpus_name)
    ids_path = os.path.join(index_dir, "track_ids.json")
    if os.path.exists(ids_path):
        print(f"[bm25] loading cached index from {index_dir}", file=sys.stderr)
        retriever = bm25s.BM25.load(index_dir, load_corpus=True)
        track_ids = json.load(open(ids_path))
        return retriever, track_ids

    print(f"[bm25] building index over {len(track_meta)} tracks (one-time)", file=sys.stderr)
    track_ids = list(track_meta.keys())
    corpus = [stringify_metadata_for_bm25(track_meta[t], CORPUS_FIELDS) for t in track_ids]
    corpus_tokens = bm25s.tokenize(corpus)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    os.makedirs(index_dir, exist_ok=True)
    retriever.save(index_dir, corpus=corpus)
    with open(ids_path, "w") as f:
        json.dump(track_ids, f, indent=2)
    # Reload to get the load_corpus=True semantics (documents come back as dicts).
    retriever = bm25s.BM25.load(index_dir, load_corpus=True)
    return retriever, track_ids


def bm25_retrieve(retriever, track_ids, queries: list[str], topk: int) -> list[list[str]]:
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


# ---------- Per-row input construction (THE KEY FIX) ------------------------

def build_retrieval_input(conversations: list[dict], track_meta: dict,
                          corpus_types: list[str]) -> str:
    """v10 invariant: BM25 query is the WHOLE conversation, with 'music' turns
    replaced by the track's metadata string. Critical for multi-turn rows."""
    lines = []
    for turn in conversations:
        role = turn["role"]
        content = turn["content"]
        if role == "music":
            role = "assistant"
            content = id_to_metadata_str(track_meta, content, corpus_types)
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


# ---------- Prompt building (matches v10 exactly) ---------------------------

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


def build_sys_prompt(roleplay: str, response_gen: str,
                     user_profiles: dict, user_id: str | None,
                     conversation_goal: Any, user_profile_blob: str | None,
                     top_tracks: list[dict]) -> str:
    """Reproduces v10's build_sys_prompt_v2 exactly, including:
      - Demographics from id_to_profile_str (user_id/age_group/gender/country_name)
      - conversation_goal stringified via plain str() (dict repr in prompt)
      - Same section labels and ordering"""
    sections = [roleplay.strip(), response_gen.strip()]

    person_bits = []
    profile_str = id_to_profile_str(user_profiles, user_id) if user_id else ""
    if profile_str:
        person_bits.append(f"User profile (demographics): {profile_str}")
    if user_profile_blob:
        person_bits.append(f"Additional user context: {user_profile_blob}")
    if conversation_goal:
        person_bits.append(f"Conversation goal: {conversation_goal}")
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


# ---------- LLM generation --------------------------------------------------

def custom_batch_generate(model, tokenizer, sys_prompts: list[str],
                          user_queries: list[str], device: str,
                          max_new_tokens: int, max_input_len: int) -> list[str]:
    """Match v10's custom_batch_generate exactly: [system, user] template,
    truncation max_length=3072, greedy decode."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    formatted = []
    for sp, uq in zip(sys_prompts, user_queries):
        chat = [{"role": "system", "content": sp}, {"role": "user", "content": uq}]
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


# ---------- Main pipeline ---------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--lora_adapter_path", type=str, default=None)
    parser.add_argument("--prompts_dir", type=str, default="./prompts")
    parser.add_argument("--bm25_cache", type=str, default="./cache/bm25")
    parser.add_argument("--test_dataset", type=str,
                        default="talkpl-ai/TalkPlayData-Challenge-Blind-A")
    parser.add_argument("--track_meta_dataset", type=str,
                        default="talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
    parser.add_argument("--user_meta_dataset", type=str,
                        default="talkpl-ai/TalkPlayData-Challenge-User-Metadata")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=192)
    parser.add_argument("--max_input_len", type=int, default=3072)
    parser.add_argument("--lm_batch_size", type=int, default=1)
    parser.add_argument("--retrieval_batch_size", type=int, default=64)
    parser.add_argument("--subset", type=int, default=None)
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

    # Datasets.
    print("[infer] loading test set", file=sys.stderr)
    test = load_dataset(args.test_dataset, split="test")
    if args.subset:
        test = test.select(range(min(args.subset, len(test))))

    print("[infer] loading track catalog (all_tracks split only — match v10)", file=sys.stderr)
    track_meta = load_track_catalog(args.track_meta_dataset, split="all_tracks")

    print("[infer] loading user profiles", file=sys.stderr)
    user_profiles = load_user_profiles(args.user_meta_dataset, split="all_users")

    # BM25 over the catalog.
    retriever, track_ids = build_or_load_bm25(track_meta, args.bm25_cache)

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
        print(f"[infer] applying LoRA adapter from {args.lora_adapter_path}", file=sys.stderr)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_adapter_path)
        model = model.merge_and_unload()
    model = model.to(args.device).to(dtype)
    model.eval()
    if hasattr(model, "generation_config"):
        model.generation_config.do_sample = False

    # Per-row inputs. v10 invariant: BM25 input is the FULL conversation;
    # LLM user message is just the LAST user turn.
    retrieval_inputs = []
    user_queries = []
    metadata_rows = []
    item_rows = []
    for item in test:
        retrieval_inputs.append(build_retrieval_input(item["conversations"], track_meta, CORPUS_FIELDS))
        user_queries.append(item["conversations"][-1]["content"])
        metadata_rows.append({
            "session_id": item["session_id"],
            "user_id": item["user_id"],
            "turn_number": item["conversations"][-1]["turn_number"],
        })
        item_rows.append(item)

    # Retrieval.
    all_picks = []
    for i in tqdm(range(0, len(retrieval_inputs), args.retrieval_batch_size), desc="bm25"):
        batch = retrieval_inputs[i:i + args.retrieval_batch_size]
        for hits in bm25_retrieve(retriever, track_ids, batch, RETRIEVAL_TOPK):
            picked = list(dict.fromkeys(hits))[:FINAL_K]
            all_picks.append(picked)

    # Build sys prompts.
    sys_prompts = []
    for idx, picks in enumerate(all_picks):
        top_metas = [track_meta[t] for t in picks[:TOP_N_FOR_LLM] if t in track_meta]
        sp = build_sys_prompt(
            roleplay=roleplay,
            response_gen=response_gen,
            user_profiles=user_profiles,
            user_id=item_rows[idx].get("user_id"),
            conversation_goal=item_rows[idx].get("conversation_goal"),
            user_profile_blob=item_rows[idx].get("user_profile") or "",
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

    # Write JSON.
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
