# Design Spec — Ensemble-Additive Generative Retrieval (SID Sub-Stream) for RecSys 2026 Music CRS

**Date**: 2026-05-15 (v2.1, post-second-review)
**Status**: APPROVED — ready for `superpowers:writing-plans`. Two reviewer passes; v1 had NEEDS-MAJOR-REVISION → v2 had APPROVE-WITH-CHANGES (5 small fixes) → v2.1 incorporates those fixes.
**Author**: brainstorm session between Or Rimoch and Claude
**Adds**: a new `sid_generator` sub-retriever and a `wrrf_bm25_dense_sid_v1` fusion variant to the existing pipeline. **Does NOT retire** any current components.

**v2.1 small fixes** (per second reviewer pass):
- §1: explicit rationale for why CMQR doesn't route through SID (training/inference parity)
- §3.1: tied-embeddings caveat for Qwen2.5-1.5B (`tie_word_embeddings=True` confirmed via `AutoConfig`); explicit untie recipe before LoRA wrapping
- §3.3: pinned `user_profile` key paths via dataset peek (top-level keys: age, country_code, preferred_musical_culture)
- §4.2: factory interface kept clean — `SID_GENERATOR` derives all paths from `cache_dir` by convention (matches `dense_local.py` precedent)
- §5.1 W4: explicit gate `mean Δ ≥ +0.005 AND paired_bootstrap_ci lower bound > 0`

**Key revisions from v1** (per first reviewer pass):
- §1: pure-SID replacement → SID as ensemble-additive 4th stream in wRRF (honors `feedback_ensemble_first_mindset.md`)
- §2.1: own the L2-concat fusion as custom design (not claiming LETTER recipe — LETTER uses CF as loss term, not concat input)
- §2.2: Sinkhorn λ marked as hyperparameter-to-tune (not LC-Rec-specific value)
- §2.4: gate 4 reformulated (use generator's stratified dev nDCG, not nonsensical "BM25 over SIDs")
- §3.1: **add `modules_to_save=["embed_tokens", "lm_head"]` to LoRA — critical fix; without this the 768 new SID tokens never get trained**
- §3.5: gate tightened from nDCG@20 ≥ 0.10 to **≥ 0.12 with paired-bootstrap CI excluding 0**
- §4: drop τ-thresholded fallback (obsolete with ensemble fusion); name `PrefixConstrainedLogitsProcessor`; fix collision pathology with per-bucket cap
- §5.1: expand from 5 weeks to **6 weeks**; W4 gate from composite to **nDCG@20 ≥ 0.08** (avoids the responder-driven LLM-lift trap from config 132)
- §5.2: add risks for RQ-VAE seed non-determinism, tokenizer-extension gotchas, HF Hub storage, Gemini-judge sensitivity to top-1 swap
- Add Rank-GRPO to future work (per `recent_papers_ideas.md:150` — most on-point paper in the curated reading list)

---

## §0 Cross-cutting implementation principles (unchanged from v1)

Per `feedback_recsys_working_methodology.md`. Apply to every component.

- **§0.1 Notebook-per-component**: each major deliverable as `colab/NN_*.ipynb` executable on Colab L4 / Blackwell. Pure logic in `scripts/` and `mcrs/` modules — notebooks are thin wrappers.
- **§0.2 TDD for pure functions**: failing test → minimum code → green → next. Phase 0 + doc2query pattern (~40+ tests written this week).
- **§0.3 Cache discipline**: artifacts > 10 MB symlinked to Drive; symlink path matches `cache_dir` resolution in configs (per the path bug fixed in commit `90cbe6f`). Drive budget audit: BGE-M3 catalog (~200 MB) + SID quantizer + doc2query parquet (~50 MB) + 5× LoRA checkpoints (~500 MB each) = ~3 GB on Drive (well under 15 GB free tier).
- **§0.4 Commit cadence + code review**: after each completed component → commit specific files → push `fresh-model` → invoke `superpowers:code-reviewer` agent. **For W3+ generator training**: push LoRA-only checkpoints (~100 MB) iteratively; merge-and-push the full 3GB model only at W5 for the final Blind-A submission, to stay within HF Hub free storage.
- **§0.5 Eval discipline**: per `feedback_offline_eval_must_match_online.md` — full-pipeline + official evaluator on dev; Blind-A for Gemini-judged signal; `scripts/compare_diagnostic_runs.py` for paired-bootstrap.
- **§0.6 RecSys 2026 rules**: `track_split_types: ["all_tracks"]` always. Only `talkpl-ai/TalkPlayData-Challenge-*` HF datasets. Submission zip = `prediction.json` (singular) at root, 80 records for Blind-A. Optimize composite. CatDiv saturated at 0.03 — don't pursue.

---

## §1 Architecture — ensemble-additive (REVISED)

```
Blind-A session
  ↓ chat_history (windowed last 3 turn-pairs) + user_query + user_profile + conversation_goal
  ↓
┌────────────────────────────────────────────────────────────────────────┐
│  PARALLEL RETRIEVAL (4 sub-streams)                                    │
│                                                                        │
│   ① BM25 (5-field corpus)                                              │
│   ② dense_metadata_qwen3 (precomputed embeddings)                      │
│   ③ dense_lyrics_qwen3 (precomputed embeddings)                        │
│   ④ sid_generator (NEW — Qwen-1.5B + LoRA, trie-constrained beam)      │
│      ↓ → top-20 track_ids (decoded from 3-token SID sequences)         │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
  ↓
wRRF fusion (k=60, weights: BM25=1.0, dense_meta=0.4, dense_lyrics=0.4, sid=0.5*)
  *initial weight; tuned in W5
  ↓
top-100 candidates
  ↓
CMQR query rewriter (existing, applies to BM25/dense — NOT to SID generator)
  ↓
ProRank reranker (existing)
  ↓
top-20 track_ids → v5-kto responder → predicted_response
  ↓
prediction.json → CodaBench
```

**Key design choices**:
- SID generator is a **4th sub-stream**, not a replacement. No components retired.
- Fusion via existing wRRF infrastructure (`mcrs/retrieval_modules/rrf.py`) — only new code is the `sid_generator` retrieval_type registration + a new `wrrf_bm25_dense_sid_v1` factory entry.
- Initial SID weight = 0.5 (slightly higher than dense streams at 0.4, lower than BM25 at 1.0). Tunable in W5 once we have Blind-A performance signal.
- **CMQR routes only to BM25/dense streams, NOT to SID**: the SID generator was trained on raw queries (`format_query_for_sid_input` in §3.3); routing CMQR-rewrites through it at inference would create a training/inference distribution mismatch. The other streams benefit from CMQR's diversification, so we keep it for them.
- Cold-start naturally handled by ensemble: when SID is uncertain (turn-1 queries with no chat history), it contributes low-confidence ranks; the other 3 streams dominate the fusion automatically. **No explicit τ-fallback needed.**
- Pure-SID and pure-current-pipeline configs both retained as comparison points — we'll know empirically whether SID is dominant or just additive.

**Promotion path** (per reviewer's recommendation):
- v1 (W4): SID as 4th wRRF sub-stream
- v2 (W5): if dev paired-bootstrap CI shows SID-augmented wRRF Pareto-dominates current wRRF AND nDCG@20 ≥ +0.02, attempt pure-SID config as a comparison submission
- v3 (W6): freeze whichever (ensemble vs pure-SID) shows best Blind-A composite

**Paper citations** (revised — own custom design where appropriate):
- **TIGER** (Rajput et al., NeurIPS 2023, `TIGER_2305.05065.pdf`) — foundational generative-retrieval framework
- **Text2Tracks** (Spotify 2025, `Text2Tracks_2503.24193.pdf`) — music-domain validation; CF-RQ-VAE beat closest baseline +127% Hits@10
- **GRID** (Snap CIKM 2025, `GRID_2507.22224.pdf`, code at `github.com/snap-research/GRID`) — practitioner's handbook; catalog 10K–100K matches our 47K; **start from GRID reference repo for quantizer + generator scaffolding**
- **LIGER** (Meta FAIR 2024, `LIGER_2411.18814.pdf`) — motivates ensemble-additive (LIGER's recommendation: hybrid > pure; we adopt parallel-stream fusion via wRRF, simpler than LIGER's sequential dense-rerank but same ensemble spirit)
- **LETTER** (Wang CIKM 2024, `LETTER_2405.07314.pdf`) — cited only for *motivation* (CF signal matters for music); the L2-concat-fusion of (text + CF + audio) is **our custom design**, NOT LETTER's recipe (LETTER adds CF as loss term, not input concat)

**Future work cited but NOT implemented in this spec**:
- **Rank-GRPO** (Netflix ICLR 2026, `Rank-GRPO_2510.20150.pdf`) — RL post-training for the SID generator; if W6 Blind-A plateaus, candidate for v2 of the generator (replace SFT with rank-position-credit GRPO)
- **Joint Search+Rec SIDs** (Spotify RecSys 2025, `Joint-SIDs_2508.10478.pdf`) — bi-encoder trained on both search and rec tasks before RQ-VAE quantization; v2 quantizer upgrade if v1 shows asymmetric search vs rec performance

---

## §2 SID Quantizer (revised)

**Purpose**: turn each of 47K track embeddings into a unique 3-token SID. One-time offline step.

### §2.1 Input — multi-modal concatenation (~1664-dim) — OWN CUSTOM DESIGN

| Modality | Source column | Dim | Why included |
|---|---|---|---|
| **Text** | `metadata-qwen3_embedding_0.6b` | 1024 | TIGER recipe; covers entity queries |
| **Collaborative filtering** | `cf-bpr` | 128 | Text2Tracks's primary signal +127% Hits@10; LETTER paper motivates CF inclusion |
| **Audio** | `audio-laion_clap` | 512 | TalkPlay validates CLAP as music encoder; org Tip 2.1 explicitly recommends |
| (Excluded) Image SigLIP2 | — | 768 | Correlated with audio; defer to v2 |

L2-normalize each modality independently before concatenation. Equal fusion weight in v1 (RQ-VAE learns internal weighting). Cold CF rows (~57% of users are cold) imputed with zero vector — RQ-VAE learns to ignore zeros via Sinkhorn balance.

**Honest framing**: this concat fusion is our custom design inspired by LETTER's motivation, NOT LETTER's exact recipe. If v1 underperforms, the alternative is LETTER's actual approach (RQ-VAE + CF contrastive alignment loss + diversity loss).

### §2.2 Quantizer — RQ-VAE + Sinkhorn-uniform regularization

```
L2-norm(text) ⊕ L2-norm(CF) ⊕ L2-norm(audio)
  → MLP encoder (1664 → 512 → 256)
  → RQ layer × 3 with Sinkhorn-balanced codebooks (each 256 codes)
  → MLP decoder (256 → 512 → 1664)

Loss = MSE_reconstruction
     + β · commitment_loss        (β=0.25, van den Oord VQ-VAE 2017 default)
     + λ · sinkhorn_uniform_loss  (λ to tune; range [0.05, 0.5]; pick on §2.4 gate 2)
```

**Citations**:
- RQ-VAE: Lee et al. CVPR 2022 + TIGER application
- Sinkhorn-uniform regularization: LC-Rec 2024 *motivates* the technique (`LC-Rec_2311.09049.pdf` — "Sinkhorn-uniform VQ prevents codebook collapse"); the specific λ is a hyperparameter we tune
- Library: `vector-quantize-pytorch` (lucidrains) — `ResidualVQ` class — we add Sinkhorn loss on top (~50 LOC)

### §2.3 Reproducibility — seed protocol

RQ-VAE training is seed-sensitive (codebook init drift). Protocol:
- Fix random seeds: `torch.manual_seed(42)`, `numpy.random.seed(42)`, `random.seed(42)` at start
- Train **3 seeds** (42, 123, 7); evaluate each against §2.4 gates
- Pick the seed with the best §2.4 gate 3 score (cluster purity)
- Pin the chosen quantizer artifact by SHA256 hash, stored alongside `track_to_sid.parquet`
- Downstream artifacts (training data, generator weights) reference the hash; mismatched hash = invalidates downstream

### §2.4 Validation gates (revised — gate 1 made relative, gate 4 reformulated)

1. **Relative reconstruction**: RQ-VAE reconstruction MSE ≤ 1.5× a PCA-256 baseline (reduce 1664→256 via PCA, then reconstruct; the RQ-VAE should at least be in the same ballpark or better). Empirically defensible threshold.
2. **Codebook utilization ≥ 80%** per level. Direct test of Sinkhorn regularization.
3. **Cluster purity ≥ 60%** at level-1: sample 100 buckets; ≥60% of tracks in each share ≥1 `tag_list` element (lowercased, stripped). Plus a **manual spot-check of 5 random buckets** by the user — sanity check that automated metric isn't gamed.
4. **Joint search/rec balance** (reformulated — moved to §3.5 post-training): we can't test this on SIDs alone (the v1 reviewer's correct critique — BM25 over SIDs is nonsensical). Instead: after generator training in W3, evaluate dev nDCG@20 separately on search-like vs rec-like dev slices (artist/title regex split). Both slices must be ≥ Phase 0 baseline (0.099).

### §2.5 Collision handling — REVISED to prevent ranking pathology

Reviewer caught a real bug: with v1's collision handling, one high-confidence beam could flood the top-20 from one collision bucket. Fixed:

```
SIDs at quantization time: ~50–500 collisions expected on 47K tracks (birthday paradox + clustering)
Resolution at quantization time:
  - Group tracks sharing same 3-tuple SID
  - Order within bucket by descending popularity
At inference time (in §4.1 step):
  - Each beam's SID expands to AT MOST 1 track per collision bucket
  - "Extra" bucket members spill over to beams 21+ ranking (used to fill any deduplication gap)
  - This keeps the beam-20 diversity guarantee
```

### §2.6 Implementation footprint (revised count after reviewer feedback)

| File | Type | Lines |
|---|---|---|
| `scripts/build_sid_quantizer.py` | new | ~300 |
| `mcrs/sid/quantizer.py` (lookup + collision handler with per-bucket cap) | new | ~150 |
| `tests/test_sid_quantizer.py` (pure-function tests for encoder/lookup/collision) | new | ~180 |
| `colab/60_build_sid_quantizer.ipynb` | new | 9 cells |
| `experiments/cache/sid/quantizer_v1.pt`, `track_to_sid.parquet` (+ SHA256 hashes) | artifacts | — |

---

## §3 SID Generator Training — CRITICAL FIX (LoRA + tokenizer)

### §3.1 Model + LoRA config — REVISED to fix the silent build-time bug

**The fix**: extending the tokenizer with 768 SID tokens requires expanding the model's embedding matrix. The new embedding rows are randomly initialized. LoRA on `q/k/v/o_proj` does NOT update embeddings. Without `modules_to_save`, training would silently fail to learn SID emission.

```python
# Step 1: extend tokenizer
new_tokens = [f"<SID_L{level}_C{code}>"
              for level in range(3) for code in range(256)]
tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})

# Step 2: resize model embedding matrix
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model.resize_token_embeddings(len(tokenizer))

# Step 3: LoRA config WITH modules_to_save (CRITICAL)
lora_config = LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    modules_to_save=["embed_tokens", "lm_head"],   # ← THIS IS THE FIX
    task_type=TaskType.CAUSAL_LM,
)
```

**Mandatory merged push** (`embed_tokens` and `lm_head` are now part of the trainable adapter set; LoRA-only delivery does NOT carry the new embeddings). After training:
```python
merged = model.merge_and_unload()
merged.push_to_hub("OrRim123/recsys2026-sid-generator-qwen15b-v1-merged")
tokenizer.push_to_hub("OrRim123/recsys2026-sid-generator-qwen15b-v1-merged")
```
Storage impact: each merged push is ~3 GB. For W3 iteration we push LoRA-only checkpoints (~120 MB with embeddings, since `modules_to_save` adds the embedding matrix); only merge for final W5 submission. HF Hub free tier accommodates this.

**Tied-embeddings caveat (verified for Qwen2.5-1.5B-Instruct: `tie_word_embeddings=True`)**: Qwen2.5-1.5B-Instruct ties `embed_tokens` and `lm_head` to the same weight matrix. With LoRA's `modules_to_save=["embed_tokens", "lm_head"]`, PEFT may save them as separate copies, breaking the tying. **Recipe**: at training script init, explicitly untie before LoRA wrapping —
```python
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
if model.config.tie_word_embeddings:
    model.config.tie_word_embeddings = False
    model.lm_head = nn.Linear(
        model.config.hidden_size, len(tokenizer), bias=False,
    )
    model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
model.resize_token_embeddings(len(tokenizer))
# THEN apply LoRA with modules_to_save=["embed_tokens", "lm_head"]
```
This materializes separate trainable matrices; both get saved correctly. Smoke-test in W3: load the merged model in a fresh kernel, confirm SID tokens appear in `tokenizer.get_vocab()` and the embeddings are non-random (norm > 0.01 per row).

### §3.2 Training data — ~290K (query, SID-sequence) pairs (corrected count)

Reviewer caught the v1 row count error. Correct numbers:

| Source | Count | Calculation |
|---|---|---|
| Raw conversations (train split) | **~8K** | 1K train sessions × 8 turns = 8000 (one `(query_with_history, gold_track_SID)` per turn) |
| Metadata-as-query | 47K | one per track |
| Doc2query synthetic (5 queries × 47K tracks) | **~235K** | reuses `scripts/doc2query_generate.py` infrastructure from earlier this week |
| **Total** | **~290K** | every track has ≥7 training examples |

Stratified split for held-out validation:
- 5% of raw conversations → eval slice A (matches Blind-A distribution best)
- 5% of metadata-as-query → eval slice B (catalog coverage diagnostic)
- 5% of doc2query → eval slice C (synthetic query generalization diagnostic)

### §3.3 Query format — concrete transformation (revised; no longer hand-waved)

A concrete function (`scripts/build_sid_training_data.py:format_query_for_sid_input`) takes the dataset row and produces the prompt string. The same function is used at inference (`mcrs/sid/inference.py`) to guarantee training/inference match:

```python
def format_query_for_sid_input(
    chat_history: list[dict],   # post-windowing — at most 6 messages (3 turn-pairs)
    current_user_query: str,
    user_profile: dict | None,
    conversation_goal: dict | None,
) -> str:
    """Build the SID generator's input prompt. Used at both training and inference.
    Output is a single string suitable for tokenization."""
    parts = []
    # Verified key paths from `talkpl-ai/TalkPlayData-Challenge-Dataset` test split:
    # user_profile is a flat dict with: age (int), age_group, country_code,
    # country_name, gender, preferred_language, preferred_musical_culture, user_id, user_split.
    if user_profile is not None:
        parts.append(f"[USER]: age={user_profile.get('age', '?')}, "
                     f"country={user_profile.get('country_code', '?')}, "
                     f"prefers={user_profile.get('preferred_musical_culture', '?')}")
    if conversation_goal is not None:
        goal = conversation_goal.get('listener_goal', '')
        if goal:
            parts.append(f"[GOAL]: {goal}")
    if chat_history:
        parts.append("[HISTORY]:")
        for msg in chat_history:
            role = "user" if msg.get('role') == 'user' else "assistant"
            content = msg.get('content', '')
            parts.append(f"  {role}: {content[:200]}")  # truncate long messages
    parts.append(f"[QUERY]: {current_user_query}")
    return "\n".join(parts)
```

Field names confirmed against `run_inference_blindset.py:144` + `talkpl-ai/TalkPlayData-Challenge-Dataset` schema: `conversation_goal.listener_goal` is the correct path.

### §3.4 Training loop

- Loss: cross-entropy over expanded vocabulary (only the 3 SID tokens contribute — earlier prompt tokens masked via `labels=-100`)
- Optimizer: AdamW, lr=2e-4 (LoRA standard for `modules_to_save` workloads; if loss diverges, drop to 1e-4)
- Scheduler: cosine with warmup_ratio=0.05
- Batch size: 8 (gradient accumulation = 4 → effective 32); fits L4 24GB
- Max sequence length: 1024 prompt tokens; truncate from FRONT to preserve recent context
- Epochs: 3 (with early stop on held-out validation loss)
- Wallclock: ~3-4 hr on L4, ~1-2 hr on Blackwell (revised up from v1; vocab-expansion + `modules_to_save` slows training ~20%)

### §3.5 Validation gate — TIGHTENED with paired-bootstrap CI

Reviewer was correct that v1's gate (nDCG@20 ≥ 0.10) passed on Phase-0-baseline noise. Revised:

**Gate**: dev nDCG@20 ≥ **0.12** on the raw-conversational held-out slice (slice A), with **paired-bootstrap CI excluding 0** vs Phase 0 baseline (recall@20 = 0.099, by definition; nDCG@20 = 0.099 on same eval).

- Test procedure: run SID generator on dev test split, compute per-turn nDCG@20, paired with the Phase 0 baseline's per-turn nDCG@20 from the cached records JSONL. Use `scripts/compare_diagnostic_runs.py:paired_bootstrap_ci(n_resamples=1000, alpha=0.05)`.
- If CI includes 0 → SID isn't learning, debug (more data, longer training, smaller LR, revisit quantizer).
- Per the §2.4 gate 4 reformulation: ALSO check that BOTH slice splits (search-like + rec-like, regex-detected from current user query) achieve nDCG@20 ≥ 0.099 (Phase 0 floor).

### §3.6 Implementation footprint

| File | Type | Lines |
|---|---|---|
| `scripts/build_sid_training_data.py` (includes `format_query_for_sid_input` + augmentation) | new | ~250 |
| `scripts/train_sid_generator.py` (vocab expansion, LoRA with modules_to_save, training loop) | new | ~350 |
| `mcrs/sid/generator_dataloader.py` | new | ~150 |
| `tests/test_sid_training_data.py` (prompt format, augmentation, stratified split) | new | ~150 |
| `colab/61_train_sid_generator.ipynb` | new | 10 cells |
| HF Hub: `OrRim123/recsys2026-sid-generator-qwen15b-vN-merged` (final only) | output | — |

---

## §4 Inference Pipeline — REVISED (drop τ-fallback; name LogitsProcessor)

### §4.1 Beam search with trie-constrained decoding via `PrefixConstrainedLogitsProcessor`

Use Hugging Face's built-in mechanism (`transformers.PrefixConstrainedLogitsProcessor`), which takes a callable `prefix_allowed_tokens_fn(batch_id, input_ids) -> List[int]`. This is the canonical pattern for autoregressive entity retrieval (Cao et al. 2021 + every paper since).

```python
# One-time at SID generator init:
sid_trie = build_sid_trie(track_to_sid_lookup)  # ~5 sec

def prefix_allowed_tokens_fn(batch_id, input_ids):
    # Walk the trie based on input_ids' SID tokens emitted so far, return valid continuations
    sid_so_far = extract_sid_tokens(input_ids)
    return sid_trie.valid_next_tokens(sid_so_far)

# At inference:
outputs = sid_generator.generate(
    **inputs,
    max_new_tokens=3,  # exactly 3 SID tokens
    num_beams=20,
    num_return_sequences=20,
    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
    output_scores=True, return_dict_in_generate=True,
)
# Decode the 20 beams' SID sequences → lookup → candidate track_ids
candidates = [sid_to_tracks[decode_sid(seq)] for seq in outputs.sequences]
# Per-bucket cap (per §2.5): take at most 1 track per collision bucket per beam
candidates = dedup_with_per_bucket_cap(candidates, cap=1)
top_20 = candidates[:20]
```

**No more τ-thresholded BM25 fallback** — the ensemble fusion via wRRF (§1 architecture) handles cold-start naturally.

### §4.2 wRRF fusion integration

Register two new keys in `mcrs/retrieval_modules/__init__.py`. **Interface convention**: the existing factory signature is `load_retrieval_module(retrieval_type, dataset_name, track_split_types, corpus_types, cache_dir)` — no extra kwargs. To stay backward-compatible (per `dense_local.py` precedent set earlier this week), `SID_GENERATOR` derives all paths from `cache_dir` by convention and reads `model_hub_id` from a small text file at `cache_dir/sid/MODEL_ID` written by the W3 training script:

```python
elif retrieval_type == "sid_generator":
    # SID_GENERATOR.__init__ reads cache_dir/sid/{track_to_sid.parquet, MODEL_ID}
    # by convention — no extra factory args needed.
    return SID_GENERATOR(
        dataset_name, track_split_types, corpus_types, cache_dir,
    )

# wRRF variant with SID as 4th stream:
elif retrieval_type == "wrrf_bm25_dense_sid_v1":
    return RRF_MODEL(
        dataset_name, track_split_types, corpus_types, cache_dir,
        sub_specs=[
            {"type": "bm25", "corpus_types": [...], "topk_internal": 60, "weight": 1.0},
            {"type": "dense_metadata_qwen3_instruct", "topk_internal": 20, "weight": 0.4},
            {"type": "dense_lyrics_qwen3_instruct", "topk_internal": 20, "weight": 0.4},
            {"type": "sid_generator", "topk_internal": 20, "weight": 0.5},  # ← NEW
        ],
        k=60,
    )
```

The `SID_GENERATOR` class implements the same `batch_text_to_item_retrieval(queries, topk)` interface as BM25 + dense retrievers, so it slots into RRF_MODEL transparently.

### §4.3 Responder handoff (unchanged)

Identical to existing pipeline. Top-1 track_id (from the fused + reranked top-20) → v5-kto responder → 320-token CoT response. No changes to responder.

### §4.4 Implementation footprint

| File | Type | Lines |
|---|---|---|
| `mcrs/sid/inference.py` (trie + LogitsProcessor + dedup-with-cap) | new | ~250 |
| `mcrs/sid/__init__.py` + `SID_GENERATOR` class implementing retrieval interface | new | ~150 |
| `mcrs/retrieval_modules/__init__.py` (register `sid_generator` + new wRRF variant) | modified | +35 |
| `tests/test_sid_inference.py` (trie, prefix_allowed_tokens_fn, dedup-with-cap) | new | ~180 |
| `colab/62_run_blindset_sid.ipynb` | new | 9 cells |
| New configs: `170-wrrf-sid-v5kto-blindsetA.yaml` + `171-pure-sid-v5kto-blindsetA.yaml` (comparison) | new | — |

---

## §5 Validation gates + 6-week timeline + risks (REVISED — expanded scope)

### §5.1 Week-by-week timeline (expanded to 6 weeks per reviewer)

| Week | Component | Validation gate before moving on |
|---|---|---|
| **W1** (May 16–22) | Quantizer build (RQ-VAE + Sinkhorn, 3 seeds, pick best); start doc2query background generation in parallel | All 4 §2.4 gates pass on chosen seed; `track_to_sid.parquet` + SHA256 pinned |
| **W2** (May 23–29) | Training data prep (raw 8K + metadata 47K + doc2query 235K → 290K pairs); dataloader; stratified split | Training data parquet validated; doc2query quality spot-check (20 random tracks) passes; 5% held-out stratified by source |
| **W3** (May 30–Jun 5) | Generator fine-tune (Qwen-1.5B + LoRA with `modules_to_save` + tokenizer expansion). Iterate up to 2 training runs if §3.5 gate fails. | Dev nDCG@20 ≥ **0.12** on raw-conversational slice with paired-bootstrap CI excluding 0; both search/rec slices ≥ 0.099 |
| **W4** (Jun 6–12) | Inference pipeline (trie + LogitsProcessor + dedup); integrate as 4th wRRF sub-stream; first dev eval | Dev nDCG@20: ensemble shows mean Δ ≥ +0.005 vs current wRRF champion AND `paired_bootstrap_ci(n_resamples=1000, alpha=0.05)` lower bound > 0; THEN FIRST Blind-A submission |
| **W5** (Jun 13–19) | Tune wRRF SID weight (sweep {0.3, 0.5, 0.7, 1.0}); compare ensemble vs pure-SID config; second/third Blind-A submissions | Pareto-dominance decision: ensemble vs pure-SID. Best config locked. **W5 gate: Blind-A nDCG@20 ≥ 0.08** (≥+0.02 over v5-kto baseline 0.06) |
| **W6** (Jun 20–22) | Freeze; document; prep for Blind-B (which opens Jun 23) | Final Blind-A composite documented; memory updated; design doc amended with what shipped |

**Hard deadline**: Blind-B opens **2026-06-23**. W6 leaves 2-day buffer.

### §5.2 Risks + mitigations (REVISED — added what reviewer flagged)

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| RQ-VAE codebook collapse | Medium | High | Sinkhorn regularization; §2.4 gate 2 catches it; λ tuning if needed |
| **RQ-VAE seed non-determinism** (NEW per reviewer 5.1) | High | Medium | 3-seed protocol + SHA256 pinning of chosen artifact (§2.3); downstream artifacts reference hash |
| **Tokenizer expansion silently breaks LoRA** (NEW per reviewer 5.3) | High if missed | Critical | Fix in §3.1: `modules_to_save=["embed_tokens", "lm_head"]`; mandatory merged push; smoke-test in W3 that new SID tokens appear in generator outputs |
| Generator fails to learn query→SID mapping | Low–Medium | High | Tightened §3.5 gate catches this; max 2 retraining attempts in W3 budget |
| Training data quality (doc2query queries poor) | Medium | Medium | W2 spot-check 20 random tracks at start; ablate aggregate vs raw-only on dev if gate ambiguous |
| Cold-start sessions (turn-1, **25% of Blind-A** per reviewer 4.1 correction) | High | Low (with ensemble) | wRRF naturally handles via the other 3 streams when SID is uncertain; no explicit rule needed |
| Beam-search latency at Blind-A scale (revised per reviewer 5.6) | Low | Medium | 80 queries × 1024-prompt prefill + 3-step decode at batch=20 ≈ **5–10 min on L4** (not "<1 min" as v1 claimed); still well within Blind-A budget |
| Modality fusion noise | Medium | Medium | Equal L2-normalized concat; if §2.4 gate 3 fails, ablate to text-only as v1.b |
| **LIGER finding correct → ensemble wins over pure** | Medium | Low (with ensemble-additive) | This IS the design now; pure-SID kept as 171 config for comparison |
| **Gemini judge sensitivity to top-1 swap** (NEW per reviewer 5.7) | High | Medium | Compare LLM-score deltas on dev sample of 50 sessions before W4 Blind-A; if v5-kto responder regresses significantly on SID-retrieved top-1, prepare responder retraining for v2 (Rank-GRPO future work) |
| **HF Hub storage budget** (NEW per reviewer 5.4) | Low | Low | LoRA-only checkpoints (~120MB) during W3 iteration; merged push (3GB) only for final W5 model; checked free tier accommodates |
| 6-week timeline slips | Medium | Medium | Weekly gates allow early detection; doc2query parallelizable with W1; image modality + Rank-GRPO are explicit v2 features |

### §5.3 Deliverables checklist (end of W6)

- [ ] `experiments/cache/sid/quantizer_v1.pt` + `track_to_sid.parquet` + SHA256 hash on Drive
- [ ] `experiments/cache/doc2query/Qwen_Qwen2.5-1.5B-Instruct/queries.parquet` on Drive (~50 MB)
- [ ] `OrRim123/recsys2026-sid-generator-qwen15b-vN-merged` on HF Hub (final version only)
- [ ] `170-wrrf-sid-v5kto-blindsetA.yaml` + `171-pure-sid-v5kto-blindsetA.yaml` configs in repo
- [ ] At least 2 Blind-A submissions: ensemble (170) + pure-SID comparison (171)
- [ ] All 7 new scripts + 6 new test files in repo with green tests
- [ ] 3 new Colab notebooks (60 quantizer, 61 generator, 62 inference) reproducible from scratch
- [ ] Memory file `project_sid_retrieval_v1_result.md` documenting final config + ablations + lessons
- [ ] Updated `MEMORY.md` index entry
- [ ] Spec amendments (this doc) with what actually shipped vs what was planned

---

## Appendix — paper-to-decision crosswalk (REVISED)

| Decision | Primary paper(s) | What it tells us |
|---|---|---|
| Generative retrieval framework | TIGER 2023 | Feasibility on recommendation catalogs |
| Music-domain SID retrieval | Text2Tracks 2025 | CF-SIDs are music-strong; +127% Hits@10 |
| Ensemble-additive (not replacement) | LIGER 2024 | Hybrid > pure on cold-start |
| Reference implementation | GRID 2025 | Saves 1–2 weeks of pipeline scaffolding |
| RQ-VAE quantizer | TIGER + Lee CVPR 2022 | Canonical hierarchical quantization |
| Sinkhorn-uniform regularization | LC-Rec 2024 (motivation) | Prevents codebook collapse (λ to tune by us) |
| L2-concat (text + CF + audio) | **Custom** (motivated by LETTER 2024 + TalkPlay 2025) | Not LETTER's exact recipe; our design |
| Trie-constrained decoding | Cao 2021 + TIGER + GRID | Standard mechanism; use `PrefixConstrainedLogitsProcessor` |
| Doc2query augmentation | LIGER + IDGenRec | Synthetic query augmentation for sparse data |
| **Future v2: RL post-training of generator** | Rank-GRPO (Netflix ICLR 2026) | Most on-point paper per `recent_papers_ideas.md:150`; candidate if W6 plateaus |
| **Future v2: search/rec balanced encoder** | Joint-SIDs (Spotify 2025) | Bi-encoder on both tasks if asymmetric performance observed |
| Composite metric weights | `recsys_challenge_notes.md:159–187` | 0.50·nDCG + 0.10·CatDiv + 0.10·LexDiv + 0.30·LLM |

---

## Self-review pre-flight

Before re-invoking the reviewer agent, this spec must pass these checks (per brainstorming skill methodology):

1. **Placeholders / TODOs**: zero unresolved. (Verified: every section has concrete numbers / paths / line counts.)
2. **Internal consistency**: §1's "ensemble-additive" architecture matches §4's wRRF integration; §2's 3-token SID matches §3's training format; §3's `modules_to_save` matches §3.6's footprint; §5.1's timeline matches §2/§3/§4 implementation effort.
3. **Scope**: 6 weeks for a solo developer with L4/Blackwell; v2 features (image modality, Rank-GRPO, Joint-SIDs encoder) explicitly deferred.
4. **Ambiguity**: query format pinned via concrete `format_query_for_sid_input` function (§3.3); collision handling pinned via §2.5 per-bucket cap; validation gates have numeric thresholds and procedures.

**Spec is ready for re-review.**
