# RecSys 2026 — Music CRS Challenge

A modular, two-component conversational music recommender for the [RecSys 2026 Challenge](https://www.recsyschallenge.com/2026/) on the [TalkPlayData-2 benchmark](https://huggingface.co/datasets/talkpl-ai/TalkPlayData-Challenge-Dataset). The system takes a multi-turn user conversation about music (with user profile + interaction history) and produces, per turn, both **(a) a top-20 ranked list of track IDs** and **(b) a natural-language response** explaining the recommendation.

This README is the entry point for a technically-fluent reader who wants the *approach* and *why*. For phase-by-phase implementation status, see [`documents/RecSys_Challenge_Plan.md`](documents/RecSys_Challenge_Plan.md) and [`documents/experiments_log.md`](documents/experiments_log.md).

---

## 1. Challenge in one paragraph

Each session is up to 8 turns. At each turn we receive a `user_query`, prior `conversations` history (interleaved `user`/`assistant`/`music` rows), a `user_profile` (`country_name`, `age_group`, `gender`), and a `conversation_goal` (listener intent + category). We must return `predicted_track_ids` (≤20 unique tracks) and `predicted_response` (free text). The leaderboard scores via:

- **Retrieval quality**: `nDCG@{1, 10, 20}` against the held-out gold track for that turn.
- **Response quality**: a closed **Google Gemini** automatic judge on two text-only axes — **Personalization** (does the response reflect user context?) and **Explanation Quality** (is the justification coherent and grounded?).

Aggregation weights are not disclosed. Our internal proxy (per [`documents/RecSys_Challenge_Plan.md`](documents/RecSys_Challenge_Plan.md) §1.2) treats nDCG@20 (~0.50) and Gemini-judge (~0.30) as the two leverage axes; CatDiv/LexDiv are saturated.

**Submissions**: a single `prediction.json` zipped at root, validated via [`scripts/validate_prediction.py`](scripts/validate_prediction.py). Blind-A is live until 2026-06-23; Blind-B opens then.

---

## 2. Architecture

The system is **modular, signal → specialist → aggregator**: every information signal in the data has a dedicated specialist, and a final ranker (today: weighted RRF; future: LambdaRank) blends them. Two top-level components compose at inference:

```
┌─────────── per-turn pipeline ───────────────────────────────────────┐
│  INPUT: user_query, chat_history, user_profile, conversation_goal   │
│      │                                                              │
│  ┌── Component A: Agentic Retrieval ──────────────────────────────┐ │
│  │  A1 State Tracker  → JSON {mood, intent, energy, sonic, era}   │ │
│  │  A2 Multi-Query Rewriter (CMQR, N=4 rewrites)                  │ │
│  │  A3 wRRF retrieval (BM25 + dense_metadata + dense_lyrics)      │ │
│  │       fuse all 12 streams via RRF k=60 → top-100               │ │
│  │  A5 ProRank reranker (Qwen-0.5B + LoRA) → top-20               │ │
│  │  A6 catalog-membership filter + A7 dedupe → top-20 final       │ │
│  └────────────────────────────────────────────────────────────────┘ │
│      │  predicted_track_ids                                         │
│      ▼                                                              │
│  ┌── Component B: Fine-Tuned Responder ──────────────────────────┐  │
│  │  Qwen-2.5-7B-Instruct + LoRA r=32                             │  │
│  │  Trained: KTO → (S-DPO conditional) → Rank-GRPO               │  │
│  │  Reward: composite over R_retr + R_judge + R_rule + R_format  │  │
│  │          + R_user_prof, with intra-rollout diversity bonus    │  │
│  └───────────────────────────────────────────────────────────────┘  │
│      │  predicted_response                                          │
│      ▼                                                              │
│  OUTPUT: {predicted_track_ids: list[str], predicted_response: str}  │
└─────────────────────────────────────────────────────────────────────┘
```

### Why two components

The challenge has two near-orthogonal objectives: retrieval (nDCG, deterministic) and response quality (Gemini judge, semantic). Optimizing them jointly couples slow, expensive RL to a fast deterministic ranker — a poor compute trade-off. We freeze the retriever (Component A, ships at W3) and train only the responder (Component B, W4–W7). At inference both compose; at training only B is on-policy.

---

## 3. Component A — Agentic Retrieval

| Module | File | Role | Reference paper |
|---|---|---|---|
| **A1 State Tracker** | [`mcrs/query_rewriters/state_tracker.py`](music-crs-baselines/mcrs/query_rewriters/state_tracker.py) | One LM call per turn extracts a structured user_state JSON `{mood, intent, energy, sonic_pref, era_pref, familiarity}` from query + history. | RA-Rec ([arxiv 2406.00033](documents/research/RA-Rec_2406.00033.pdf)) |
| **A2 CMQR** | [`mcrs/query_rewriters/cmqr.py`](music-crs-baselines/mcrs/query_rewriters/cmqr.py) | Single LM call emits N=4 rewrites that each inject 1–2 user_state fields. Each rewrite hits the inner retriever at top-K=50; the N lists fuse via RRF (k=60). | CMQR (SIGIR 2024, [arxiv 2406.18960](documents/research/CMQR_2406.18960.pdf)) |
| **A3 wRRF retriever** | [`mcrs/retrieval_modules/rrf.py`](music-crs-baselines/mcrs/retrieval_modules/rrf.py) | Three precomputed-embedding sub-retrievers fused with weights `(BM25=1.0, dense_metadata=0.4, dense_lyrics=0.4)`. Encoder: Qwen3-Embedding-0.6B (asymmetric instruct prefix on queries). | Reciprocal Rank Fusion (Cormack et al. 2009) |
| **A5 ProRank** | [`mcrs/rerankers/pro_rank.py`](music-crs-baselines/mcrs/rerankers/pro_rank.py) | Last-token-logit-diff over Qwen-0.5B (+ LoRA fine-tuned in W3). Scores `P("yes")/P("no")` at the next position given a `(query, doc)` prompt. Optionally emits 3-5-word rationales per top-N. | ProRank ([arxiv 2506.03487](documents/research/ProRank_2506.03487.pdf)) |
| **A6 Catalog filter** | [`mcrs/crs_baseline.py:496-518`](music-crs-baselines/mcrs/crs_baseline.py) | Drops any track ID not in the official catalog set; backfills from the pre-rerank pool to keep `len == 20`. Hard guard against W4-W7 responder hallucination. | — |
| **A7 Dedupe** | same file | First-seen-wins dedupe. Cheap, mandatory at every fusion step. | — |

**Cold-start specialist (A4 MARec)** is deferred per plan §A4 — gated on whether the cold subset shows headroom vs the existing imputation chain (artist-mean → category-mean → global-mean) in [`dense_precomputed.py`](music-crs-baselines/mcrs/retrieval_modules/dense_precomputed.py).

---

## 4. Component B — Fine-Tuned Responder

Base: **Qwen-2.5-7B-Instruct** + **LoRA r=32, α=32, dropout=0.05, target=`all-linear`**. The training ladder lands a *single* checkpoint that conditions on the prompt-injected envelope and emits a structured response.

### 4.1 The envelope

Every assistant output is structured:

```
<user_state>
mood: reflective
energy: low
</user_state>
<response>
Holocene by Bon Iver leans into a layered arrangement and slow tempo,
matching the reflective mood you described.
</response>
```

The `<response>` block is what gets scored by the Gemini judge. The `<user_state>` block is the model's own re-extracted summary (CoT-style) — it nudges the policy to ground in user state before generating, mitigating the **structural-directive collapse** observed on Qwen-1.5B (every prompt-engineering variant of `exp 022–029` regressed Gemini score by 0.9–1.05 vs the unstructured `exp 021` champion).

### 4.2 Training ladder

| Stage | Algo | Data | Reference |
|---|---|---|---|
| **B1 (W4) KTO warmup** | KTO on (`text_a`, `text_b`, label) GPA-derived pairs (~60k turns); LoRA r=32 | [`scripts/build_reward_dataset.py`](scripts/build_reward_dataset.py) → [`scripts/augment_envelope.py`](scripts/augment_envelope.py) → [`scripts/build_trl_datasets.py`](scripts/build_trl_datasets.py) | KTO ([arxiv 2402.01306](documents/research/KTO_2402.01306.pdf)) |
| **B2 (W5) S-DPO refinement** *(conditional, fires only if KTO format <70%)* | TRL `DPOTrainer` over (1-pos, 4-neg) → 4-pair expansion. 4 negative types: `drop_track_name`, `inject_banned`, `truncate_5`, `cross_session_GPA_NEG`. | [`scripts/build_sdpo_dataset.py`](scripts/build_sdpo_dataset.py) | S-DPO ([arxiv 2406.09215](documents/research/S-DPO_2406.09215.pdf)) |
| **B3 (W6) Rank-GRPO main loop** | TRL `GRPOTrainer`, on-policy G=4 rollouts per prompt, `scale_rewards=False` (preserves reward magnitude under low G), KL β=0.04, lr=5e-6. | [`scripts/build_grpo_dataset.py`](scripts/build_grpo_dataset.py), [`colab/32_train_responder_grpo.ipynb`](colab/32_train_responder_grpo.ipynb) | Rank-GRPO ([arxiv 2510.20150](documents/research/Rank-GRPO_2510.20150.pdf)), Rec-R1 ([arxiv 2503.24289](documents/research/Rec-R1_2503.24289.pdf)) |

The chained-merge pattern (each LoRA stage merges into the prior stage's base before the next train) ensures every checkpoint deploys as a single fully-merged 7B with `lora_path: null`. Documented in [`memory/project_responder_merge_pattern.md`](.) (point-in-time observation; verify against current notebook code).

### 4.3 Reward design

`compose_r_turn` in [`scripts/reward_fns.py`](scripts/reward_fns.py) returns a dict; the trainer uses `r_turn`. Weights (v4, post deep-review):

| Term | Weight | Source | What it captures |
|---|---|---|---|
| `R_retr` | 0.40 | leaderboard `nDCG@{1,10,20}` weighted 0.5/0.3/0.2 vs gold track | Retrieval accuracy. **Constant per row at training** (frozen retriever) → contributes 0 GRPO advantage. Kept for offline-eval alignment. |
| `R_judge` | 0.30 | DistilledJudge (`cross-encoder/ms-marco-MiniLM-L-6-v2` regression-fine-tuned on 210k Gemini-aligned anchors in [`data/reward_calibration_anchors.parquet`](.)) | Approximates the closed Gemini judge for training-time gradient. |
| `R_rule` | 0.15 | Regex over response: track-name + artist mention, "why" vocabulary (`tempo`, `groove`, `because`, …), length 60–110 words, 1–4 sentences, no banned phrases, history-token grounding | Mechanical structure check. AUC 0.51 vs Gemini — kept as a regularizer, not a primary objective. |
| `R_format` | 0.10 | Binary: does `<user_state>...</user_state>...<response>...</response>` parse with at least one allowed state key? | Anchors envelope-fluency. |
| `R_user_prof` | 0.05 | Mention of user_profile fields (`country_name`, `age_group`, `gender`) in response | Personalization signal — one of the two Gemini judge axes. |

**Bonus**: `+0.05 × lex_div_pairwise(group_responses)` — across-rollout Jaccard distance over bigram sets. Rewards intra-prompt diversity; identical rollouts → 0 bonus, disjoint → +0.05. Final `r_turn` is clamped to `[0, 1]`.

**Hard guards** (multiplicative zero on r_turn): catalog-membership violation (any predicted_id ∉ `valid_catalog`); envelope parse failure (when `include_format=True`).

### 4.4 Distilled judge

Why R_judge needed a model: the open-weight `r_judge_stub` returned 0 always, leaving only `R_rule + R_format = 0.20` of weight as effective gradient. We train a tiny cross-encoder ([`scripts/train_distilled_judge.py`](scripts/train_distilled_judge.py)) on:

- **Source A** (105k): GPA POS/NEG anchors from training conversations.
- **Source D** (105k): synthetic perturbations of POS gold (drop_track_name, drop_why, etc.) labeled NEG.

Targets are normalized to `[0, 1]` via `(judge_anchor - 1) / 4`. Loss: MSE on raw logits (no sigmoid in loss). At inference [`DistilledJudge.score`](scripts/reward_fns.py) returns the raw logit clamped to `[0, 1]` (preserves the calibrated training range).

References: [Judging-Judges](documents/research/Judging-Judges_2406.07791.pdf) (position-bias diagnostics) — explicit caveat that local LLM-as-Judge is misaligned with Gemini.

---

## 5. Inference pipeline (input → output)

### Input (one turn from `talkpl-ai/TalkPlayData-Challenge-Dataset`):

```python
{
  "session_id": "3e0c2c15-...",
  "user_id": "272e35bc-...",
  "conversations": [
    {"turn_number": 1, "role": "user",      "content": "I want chill folk"},
    {"turn_number": 1, "role": "assistant", "content": "How about Bon Iver?"},
    {"turn_number": 1, "role": "music",     "content": "<track_uuid>"},
    {"turn_number": 2, "role": "user",      "content": "More upbeat"},
    # ... up to turn N (the prediction target)
  ],
  "conversation_goal": {"category": "discovery", "listener_goal": "..."},
  "user_profile": {"country_name": "Norway", "age_group": "30s", "gender": "F"}
}
```

### Pipeline (per `mcrs.crs_baseline.batch_chat`):

1. **A1**: extract `user_state = {"mood": "reflective", "energy": "low", ...}` from `(user_query, history)`.
2. **A2**: with `user_state` injected into a prompt, generate 4 rewrites.
3. **A3**: each rewrite → wRRF top-50; fuse all 4 lists → top-100.
4. **A5**: ProRank scores 100 candidates → top-20.
5. **A6+A7**: drop hallucinated UUIDs, dedupe, backfill from pre-rerank pool → final 20.
6. **Component B**: render `(user_query + listener_goal + recommended_track_meta + history + user_state)` into the envelope-aware prompt, invoke W6/W7-merged 7B, decode → response text.

### Output (one row of `prediction.json`):

```python
{
  "session_id": "3e0c2c15-...",
  "user_id": "272e35bc-...",
  "turn_number": 3,
  "predicted_track_ids": ["uuid1", "uuid2", ..., "uuid20"],   # exactly 20 unique
  "predicted_response": "Heart-Shaped Box by Nirvana features a grunge tempo..."
}
```

CodaBench packaging via [`scripts/validate_prediction.py`](scripts/validate_prediction.py) `package_zip` — one entry `prediction.json` at zip root, `ensure_ascii=False`.

---

## 6. Evaluation

Three evaluation surfaces, ordered by how much weight they carry in our gates:

1. **Per-stage `gate_result.json`** — every B-stage notebook (`colab/30/31/31p/32/33`) writes one. Format compliance (strict `r_format`) + reward delta vs B1 baseline. Gates: KTO ≥95%, GRPO ≥+0.03 R_turn over B1, no-regression dev nDCG@20 ≤ 0.005 drop.
2. **Composite eval** via [`scripts/responder_eval.py`](scripts/responder_eval.py) — reads `prediction.json` + HF gold + optional DistilledJudge; emits per-row + summary `r_turn`/`r_retr`/`r_rule`/`r_judge`/`r_format`/`r_user_prof` plus `recall@{1,10,20}`. The honest internal proxy.
3. **CodaBench leaderboard** — the real signal, but limited budget (≤3 Blind submissions per week per [`scripts/validate_prediction.py:172`](scripts/validate_prediction.py)). [`documents/submissions_log.md`](documents/submissions_log.md) is the audit log.

Phase gate decisions (YES/NO/DEFER) are logged in [`documents/experiments_log.md`](documents/experiments_log.md) per plan §11.

---

## 7. Repository layout

```
recsys2026/
├── README.md                           ← this file
├── documents/
│   ├── RecSys_Challenge_Plan.md        ← full 8-week plan (W1–W8)
│   ├── experiments_log.md              ← per-gate YES/NO/DEFER ledger
│   ├── submissions_log.md              ← Blind-A/B upload audit
│   └── research/                       ← 25 reference papers (PDFs)
├── music-crs-baselines/
│   ├── mcrs/                           ← Component A modules (state_tracker, cmqr, retrieval_modules, rerankers)
│   ├── config/                         ← per-experiment yamls (110 ProRank, 220 W6 dev-eval, 300 Blind-B, 301 Blind-A)
│   ├── run_inference_devset.py         ← dev-eval runner
│   └── run_inference_blindset.py       ← Blind-A/B runner
├── scripts/
│   ├── reward_fns.py                   ← compose_r_turn + DistilledJudge
│   ├── build_reward_dataset.py         ← train+dev → reward_train.parquet (with user_profile_json)
│   ├── augment_envelope.py             ← wraps text_b in <user_state>...<response> envelope
│   ├── build_sdpo_dataset.py, build_grpo_dataset.py, build_train_plus_dev.py
│   ├── train_distilled_judge.py        ← Gemini-anchor regression cross-encoder
│   ├── responder_eval.py               ← unified composite eval (plan §10)
│   └── validate_prediction.py          ← CodaBench schema + zip + budget
├── colab/                              ← Colab notebooks for each phase (30 KTO, 31 SDPO, 31p pilot, 32 W6, 33 W7, 40 Blind-B, 41 Blind-A)
└── tests/                              ← 342 unit tests, pure-CPU; run via `pytest`
```

---

## 8. Notebook execution flow

The Colab notebooks chain together via `gate_result.json` contracts on Drive. Each notebook reads its predecessors' outputs from `Drive/recsys2026-cache/{stage}_runs/{run_name}/gate_result.json` (looking for the `merged_hub_model` field — a fully-merged Hub repo to start from) and writes its own gate result.

### 8.1 Naming convention

Two-digit prefix encodes `{wave}{step}`:

| Decade | Wave | Active notebooks |
|---|---|---|
| `2X` | W4-prep | `22_extract_train_states` (one-time prereq for envelope augmentation) |
| `3X` | W4-W7 — responder cascade | `30_train_responder_kto`, `31_train_responder_sdpo`, `31p_pilot_grpo_with_judge`, `32_train_responder_grpo`, `33_train_responder_grpo_train_plus_dev` |
| `4X` | Blind-set inference | `40_run_blindset_B`, `41_run_blindset_A` |

The `p` suffix on `31p` denotes the W6 PILOT (relaxed gate, ~1.5 hr) — a cheap direction check before committing to the full W6 in `32` (~5 hr).

**Archived** under `colab/_archive/`: `10_train_cmqr_dev` (W2 query rewriter, already trained) and `20_train_prorank_dev` (W3 reranker, already trained). Component A is frozen at W3, so these aren't on the active leaderboard path. Kept for reproducibility audits.

### 8.2 Cascade graph

```
┌─────────────────────────────────────────────────────────────────────┐
│  30 — W4 KTO                                                        │
│  Train Qwen-3B + LoRA → merge in-place → push merged repo to Hub    │
│  Outputs:                                                           │
│    • Hub: recsys2026-b1-kto-qwen3b-{date}-merged                    │
│    • Drive: kto_runs/{run}/gate_result.json {merged_hub_model, ...} │
│  Gate: format compliance ≥ 95% (strict r_format)                    │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
       ┌──────────────────────┴──────────────────────┐
       │                                             │
       ▼                                             ▼
┌──────────────────────┐               ┌─────────────────────────────┐
│ 31 — W5 SDPO         │               │ 31p — W6 PILOT              │
│ CONDITIONAL —        │               │ Distilled judge + short     │
│ runs only if W4      │               │ GRPO (2k steps)             │
│ format < 70%         │               │ Reads kto_runs/             │
│                      │               │ Writes grpo_pilot_runs/     │
│ Reads kto_runs/      │               │ Gate (relaxed):             │
│ Writes sdpo_runs/    │               │   format ≥ 90%              │
│                      │               │   Δ R_turn ≥ +0.015         │
└──────────────────────┘               └─────────────────────────────┘
                                                     │
                                                     ▼
                                       ┌─────────────────────────────┐
                                       │ 32 — W6 GRPO (full)         │
                                       │ 12k steps × G=4 rollouts    │
                                       │ Reads sdpo_runs (or         │
                                       │ kto_runs as fallback)       │
                                       │ Writes grpo_runs/           │
                                       │ Gate (strict):              │
                                       │   format ≥ 95%              │
                                       │   Δ R_turn ≥ +0.030         │
                                       └─────────────────────────────┘
                                                     │
                                                     ▼
                                       ┌─────────────────────────────┐
                                       │ 33 — W7 train+dev retrain   │
                                       │ Same recipe as W6, but on   │
                                       │ train ∪ dev (no held-out)   │
                                       │ Reads grpo_runs/            │
                                       │ Writes grpo_final_runs/     │
                                       └─────────────────────────────┘
                                                     │
                          ┌──────────────────────────┴───────────────┐
                          ▼                                          ▼
            ┌─────────────────────────┐              ┌──────────────────────────┐
            │ 41 — Blind-A inference  │              │ 40 — Blind-B inference   │
            │ Cascading lookup:       │              │ Reads grpo_final_runs/   │
            │   W7 > W6 > 31p > W5 > W4│              │ Patches config 300      │
            │ Patches config 301      │              │ Inference + zip          │
            │ Inference (80 turns)    │              │ Drive-staged for upload  │
            │ → prediction.json       │              │ → prediction.json        │
            │ → blindset_A_*.zip      │              │ → blindset_B_*.zip       │
            └─────────────────────────┘              └──────────────────────────┘
```

### 8.3 Critical handoffs

Every B-stage notebook produces a `gate_result.json` that the next stage reads. The single load-bearing field is `merged_hub_model` — a fully-merged Qwen-3B Hub repo that the next stage `from_pretrained()`s as its starting base.

| Producer | Key field | Consumer(s) |
|---|---|---|
| `30` (W4 KTO) | `merged_hub_model`, `format_compliance_strict` | `31`, `31p`, `32`, `41` |
| `31` (W5 SDPO) | `merged_hub_model` | `31p`, `32`, `41` |
| `31p` (W6 pilot) | `merged_hub_model`, `judge` | `32` reads `judge`; `41` may use as source |
| `32` (W6 full) | `merged_hub_model`, `gate_passed` | `33` (gates on `gate_passed`), `41` |
| `33` (W7) | `merged_hub_model` | `40` |

### 8.4 Recommended path

For the full leaderboard pipeline:

1. `30` — W4 KTO (~80-100 min, ~15 units) — must pass format gate
2. `31p` — W6 pilot (~2 hr, ~30 units) — also trains the distilled judge that `32`/`33` reuse
3. `41` — optional Blind-A submission off the pilot for cheap Gemini-anchored signal (~30 min, ~6 units)
4. `32` — W6 full (~5 hr, ~65 units) — only if pilot gate passes
5. `33` — W7 train+dev retrain (~5 hr, ~65 units) — for the final Blind-B submission
6. `40` — Blind-B inference (~30 min, ~6 units)

Skip `31` unless `30` reports format compliance < 70% (rare).

---

## 9. Reproducibility — running the pilot

The W6 pilot ([`colab/31p_pilot_grpo_with_judge.ipynb`](colab/31p_pilot_grpo_with_judge.ipynb)) is the cheapest path to validate the Option B refactor end-to-end. Total ~2 A100-hr ≈ 30 Colab compute units (≈ 30% of one Pro month):

1. Train the distilled judge (~30 min A100, ~6 units) — cell 6.
2. Pre-compute retrieval over 1500 sessions (~5 min, ~1 unit) — cell 7.
3. Run GRPO 2000 steps × G=4 (~1.5 hr, ~20 units) — cells 9-10.
4. Format + Δ R_turn evaluation (~10 min, ~2 units) — cell 11.

**Pilot gate**: format ≥ 90% AND Δ R_turn vs W4 ≥ +0.015. If PASS, schedule the full W6 (`colab/32`, 12k steps, ~10 A100-hr ≈ 130 units). If FAIL, the cell-11 decision tree branches to specific recovery paths.

Local CPU work (no Colab budget): all 342 unit tests run via `pytest` in ~2.3s.

---

## 10. Status & deferred items

**Shipped (W1–W7 scaffolding complete)**: state-tracker, CMQR, wRRF, ProRank, KTO/SDPO/GRPO trainers, distilled judge, train+dev retrain pipeline, Blind-A and Blind-B submission notebooks, validators, 342 tests.

**Pending GPU runs** (the actual bottleneck): W4 KTO, distilled judge training, W6 pilot, full W6, W7 retrain, then submissions.

**Deferred per documented plan gates**:

- **A4 MARec cold-start specialist** (plan §A4) — gated on cold-subset showing nDCG@10 headroom.
- **W3 stretch Rank-R1** ([arxiv 2503.06034](documents/research/Rank-R1_2503.06034.pdf)) — only if ProRank lands W3 gate AND a side Colab session is free.
- **W8 stretch joint Search-R1 rollout** (plan §6.7) — hard cut-off Jun 22 per timeline.
- **`<reranker_rationales>` block at inference** — train-side dropped per deep-review P0-2; re-enable when [`mcrs/crs_baseline.batch_chat`](music-crs-baselines/mcrs/crs_baseline.py) wires `pro_rank.generate_rationales` into the prompt template.

The honest plan-vs-code alignment ledger lives in [`documents/experiments_log.md`](documents/experiments_log.md).

---

## 11. Key references

- **Retrieval**: [CMQR](documents/research/CMQR_2406.18960.pdf) (multi-query rewriting), [RA-Rec](documents/research/RA-Rec_2406.00033.pdf) (state tracker), [ProRank](documents/research/ProRank_2506.03487.pdf), [Rank-R1](documents/research/Rank-R1_2503.06034.pdf), [RankZephyr](documents/research/RankZephyr_2312.02724.pdf), [Mistral-SPLADE](documents/research/Mistral-SPLADE_2408.11119.pdf).
- **Preference + RL**: [KTO](documents/research/KTO_2402.01306.pdf), [S-DPO](documents/research/S-DPO_2406.09215.pdf), [Rank-GRPO](documents/research/Rank-GRPO_2510.20150.pdf), [Rec-R1](documents/research/Rec-R1_2503.24289.pdf), [DRPO](documents/research/DRPO_2410.18127.pdf), [OPO](documents/research/OPO_2410.04346.pdf).
- **Cold-start / generative IDs**: [MARec](documents/research/MARec_2404.13298.pdf), [LLM-Prior-ColdStart](documents/research/LLM-Prior-ColdStart_2411.09065.pdf), [LLM-ESR](documents/research/LLM-ESR_2405.20646.pdf), [IDGenRec](documents/research/IDGenRec_2403.19021.pdf), [LC-Rec](documents/research/LC-Rec_2311.09049.pdf), [LETTER](documents/research/LETTER_2405.07314.pdf), [LIGER](documents/research/LIGER_2411.18814.pdf), [GRID](documents/research/GRID_2507.22224.pdf), [Joint-SIDs](documents/research/Joint-SIDs_2508.10478.pdf).
- **Judge calibration**: [Judging-Judges](documents/research/Judging-Judges_2406.07791.pdf), [PEBOL](documents/research/PEBOL_2405.00981.pdf).
- **Joint search-rec (deferred stretch)**: [Bridging-Search-Rec](documents/research/Bridging-Search-Rec_2410.16823.pdf), [FIRST](documents/research/FIRST_2406.15657.pdf).

For the per-paper TL;DR + how each idea maps to a module, see `documents/research/recent_papers_ideas.md` (referenced in plan §8).
