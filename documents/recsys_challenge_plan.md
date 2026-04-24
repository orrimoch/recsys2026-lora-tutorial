# RecSys Challenge 2026 — Master Plan

**Status:** living document, `fresh-model` branch, 2026-04-24.
**Owner:** Or Rimoch (`orrimoch@gmail.com`).
**Scope:** Music-CRS task — per-turn top-20 track retrieval + natural-language response, evaluated on devset locally and Blind-A / Blind-B via CodaBench.

This is the **source-of-truth playbook**. A session (or teammate) should be able to pick up any task below and execute it using only this doc + `recsys_challenge_notes.md` + `data_exploration.md` + `research/recent_papers_ideas.md`.

### Working principles (read every session)

These are the non-negotiable rules that shape every task in §5 and every decision in §8. They are more important than any single experiment.

**0. No retrieval-only experiments.** Every shippable experiment is either (a) two-step — retrieval + LLM response generation — or (b) end-to-end — Semantic-ID generative. Retrieval-only configs (e.g. `lm_type: null`) are tolerated only as *informational baseline inventory* used to pick the retrieval branch of a two-step stack; they are NOT shippable because they always zero out LexDiv and earn no LLM-judge. Enforced in every new YAML config.

**0b. Prompt engineering is a first-class lever.** Response-generation prompts are designed artifacts, not stock templates. Every Track-C prompt deliberately applies a subset of: **persona**, **chain-of-thought (hidden scratchpad)**, **structured context engineering** (parsed user profile + labelled candidate list), **reflection / CoVe**, **nucleus sampling + anti-repetition**, **few-shot** (static curated OR dynamic NN from train), **negative examples / word bans**, **step-back intent classification**, **forced metadata grounding** (outlines/logits). **Stock prompts are forbidden** — the repo's `response_generation.txt` has a known-dead apology directive. Every prompt lives at `mcrs/system_prompts/response_generation_v{N}_{label}.txt`; never edit in place.


1. **Optimise the metrics that matter most: `nDCG@{1,10,20}` and LLM-as-Judge.** Catalog_diversity and lexical_diversity are distant thirds — we do not sacrifice nDCG or LLM-judge to chase them. The blind scoring weighs these two metric families most heavily (§2.4).
2. **The judge is Gemini, measured online.** We do not run any LLM-judge locally (no local hardware for Qwen/Llama-as-judge, no Gemini API spend in the current phase). The LLM-judge score is observed only via CodaBench blind submissions. Design for Gemini using qualitative rubric cues from challenge docs; confirm via online scoring. Local iteration is retrieval-first.
3. **No reinventing the wheel.** Before writing any new module, search for working open-source code (paper repos, HF Spaces, popular toolkits). Port / adapt first; write from scratch only when no reasonable OSS baseline exists. The rule and the search checklist are in §7.4.
4. **Small improvements first, heavy improvements second.** For any axis, ship the XS/S-effort candidate before the M/L/XL one. Cheap wins compound and de-risk expensive work (§8.4). Prior iteration proved this: a single-line persona prompt beat a 2-day LoRA fine-tune on LLM-judge.
5. **The iteration loop has a memory.** We do not re-discover the same lessons. Cross-experiment learnings are captured in the agentic-memory subsystem (§8.3) — dead ends, validated mechanisms, correlation between predicted and actual deltas. Every new candidate is scored against that memory.

---

## Table of Contents

1. [Context & Constraints](#1-context--constraints)
2. [Success Criteria & Benchmarks](#2-success-criteria--benchmarks)
3. [System Architecture](#3-system-architecture)
4. [Experiment Framework & Logging Protocol](#4-experiment-framework--logging-protocol)
5. [Task Catalog](#5-task-catalog)
6. [Colab / GPU Workflow](#6-colab--gpu-workflow)
7. [Testing & Verification Discipline](#7-testing--verification-discipline)
8. [Iteration Loop & Candidate-Generation Protocol](#8-iteration-loop--candidate-generation-protocol)
9. [Wave-Based Execution Order](#9-wave-based-execution-order)
10. [Appendix](#10-appendix)

---

## 1. Context & Constraints

### 1.1 The task (compressed)

Conversational music recommender. Each session has 8 turns; per turn the system produces:

1. **A ranked list of up to 20 track_ids** (ordered, no duplicates, all from the 47,071-track catalog)
2. **A natural-language `predicted_response`** justifying the recommendation

Inputs: conversation history, user profile (age, gender, country, preferred_musical_culture), optional `conversation_goal` metadata.

Full task spec in `documents/recsys_challenge_notes.md`. Data specifics in `documents/data_exploration.md`. This doc does not re-derive them; it assumes both are current.

### 1.2 Hard rules (violating any invalidates submission)

| Rule | Source | Enforcement point |
|---|---|---|
| `track_split_types` MUST be `["all_tracks"]` — no subsetting the catalog | `music-crs-baselines/readme.md:70–82` | CI check on any new YAML config |
| `prediction.json` (singular) at zip root, server extracts to `/app/input/res/prediction.json` | `music-crs-evaluator/readme.md` | Submission packager (E-3) |
| One entry per (session × turn) — missing turns fail | `music-crs-evaluator/readme.md:219` | Schema validator |
| `ensure_ascii=False` when writing JSON | track/artist names contain CJK/accents | Schema validator |
| No duplicate track_ids within a prediction | `metrics_recsys.py:127-130` raises | Schema validator |
| Never modify `music-crs-evaluator/exp/ground_truth/` | CLAUDE.md rule | pre-commit hook (future) |
| Train on train only; dev is eval-only; blind never peeked | project-level split discipline | Feature code must be identical across splits |

### 1.3 One-axis discipline

Every experiment declares exactly one axis under test (retrieval / reranker / query-rewriter / response-prompt / response-model / training-objective / fusion / aggregator). Stacked changes require a planned ablation — e.g. if v4 = tag_list-BM25 + query-expansion + few-shot, we must also ship sub-ablations that flip each axis independently. Unattributable regressions are an iteration tax we will not pay.

### 1.4 The 4 scored metrics

| Metric | Where | Range | Notes |
|---|---|---|---|
| `nDCG@1`, `nDCG@10`, `nDCG@20` | `music-crs-evaluator/metrics/metrics_recsys.py:100` | [0, 1] | Single GT per turn ⇒ IDCG=1, so nDCG is Hit@k with log-discount |
| `catalog_diversity` | `metrics_diversity.py:9` | [0, 1] | `|unique predicted| / |catalog|` |
| `lexical_diversity` | `metrics_diversity.py` | [0, 1] | Distinct-2 bigram ratio across `predicted_response` |
| **LLM-Judge (Google Gemini)** | **blind only, server-side** | 1–5 | Personalization + Explanation Quality. Prompt not disclosed. **Observed only via CodaBench blind submission** — no local judge in current phase (see §2.5 and Track J) |

Composite score as published per leaderboard uses weighted sum; our internal composite (for ranking experiments) is:

```
composite = 0.50·nDCG@20 + 0.10·catalog_diversity + 0.10·lexical_diversity + 0.30·(LLM_judge − 1) / 4
```

### 1.5 Known data quirks (from `data_exploration.md`)

- ~1% empty precomputed track embeddings (492–616 rows/column). **Impute**: artist-mean → global-mean → L2-normalize. Contract already in `music-crs-baselines/mcrs/retrieval_modules/dense_precomputed.py:106–171` — **reuse, don't re-implement**.
- `cf-bpr` user embeddings: 80.6% of test_cold users have empty cf-bpr. Cold users need content-only fallback.
- Track `duration == 0` and `release_date == "0000-01-01"` sentinels exist. Filter or impute before use.
- `tag_list` is user-generated noise; heavy cleaning required before BM25 inclusion (stopword filter, rare-tag threshold).
- Dim asymmetry: audio-CLAP=512 (L2-norm=1), image-SigLIP=768 (norm~10), cf-bpr=128 (norm~0.03), Qwen3 variants=1024 (norm~90+). **Must re-normalize per-space before cosine / fusion.**
- Blind-A mixes warm users (23/54) with net-new users (31/54). Cold-start matters more than train suggests.

---

## 2. Success Criteria & Benchmarks

### 2.1 Three benchmarks, always

Every experiment reports vs all three.

| Benchmark | Defined by | Use |
|---|---|---|
| **B-floor** | `LLaMA-1B + BM25` official devset baseline: `nDCG@1=0.0098, nDCG@10=0.0627, nDCG@20=0.0815, CatDiv=0.38, LexDiv=0.26` | Absolute sanity; functional regression indicator. A stack scoring below B-floor composite is broken |
| **B-champ** | Current top of `documents/benchmarks.md` — our best-shipped internal result | Shipping bar: `composite_dev(new) ≥ composite_dev(B-champ) + σ` (σ = noise floor, §2.3.2) |
| **B-target** | Public Blind-A leaderboard top-1 (per `.claude/memory/project_blind_a_state.md`: `nDCG@20≈0.21, LLM≈3.85, composite≈0.40`) | Stretch goal; drives which axes are worth pursuing |

### 2.2 Definition of done

- **Minimum viable submission** — composite_dev ≥ B-floor composite AND composite_blindA ≥ 0.25
- **Strong result** — composite ≥ 0.34 on Blind-A (matches prior rank-2)
- **Stretch** — composite ≥ 0.40 on Blind-A (matches or beats B-target); top-3 on Blind-B

### 2.3 What we actually optimise

**We optimise the composite score.** The leaderboard ranks by composite; the composite has a given formula (§1.4); we maximise it. Nothing else.

```
composite = 0.50 · nDCG@20  +  0.10 · catalog_diversity  +  0.10 · lexical_diversity  +  0.30 · (LLM_judge − 1) / 4
```

Per-metric deltas are informational — they decompose *why* composite moved and feed the agent memory (§8.3) — but **they are not independent constraints**. A stack that drops `lexical_diversity` by 0.05 and lifts `nDCG@20` by 0.02 is a strict win (+0.005 composite) and we ship it. A stack that lifts `lexical_diversity` by 0.10 but drops `nDCG@20` by 0.01 is a strict loss (−0.005) and we reject it, regardless of how the per-metric deltas read in isolation.

#### 2.3.1 The shipping decision rule

Ship if and only if `composite_dev > composite_dev(B-champ)` with statistical margin (see §2.3.2). Blind ships follow the same rule using the most recent blind score where available.

#### 2.3.2 Noise floor (avoid shipping noise as progress)

Dev nDCG@k has some per-session variance. Treat a composite delta smaller than the empirical noise floor as zero.

- Estimate the noise floor once (at W0) by running the same config twice with `temperature=0` and different random seeds on a shuffled split — Δcomposite observed = the noise band. Call it `σ`.
- Require `composite_dev(new) − composite_dev(B-champ) ≥ 1.0 · σ` to ship.
- This replaces the previous ">2% per-metric regression" guard: composite is what counts; the noise check is what keeps us from chasing tails.

#### 2.3.3 Pareto info (useful, not enforced)

Continue reporting per-metric deltas in every log entry — they drive the agent memory's Cross-correlations section (§8.3.1) and help spot fragile stacks. A stack that gains composite by sacrificing a P0 metric severely (e.g. nDCG@20 drops 0.05 but LLM jumps 0.5) is allowed to ship, BUT the agent memory must record the fragility (next blind or next judge prompt may swing the trade the other way). This shapes how aggressively we build on that stack.

#### 2.3.4 On the "outperform all 4 metrics" framing

That was the stated goal in the original problem framing, but it's a *soft* ambition subsumed by composite optimization. A composite-maximising trajectory naturally trends toward higher scores on all four metrics over many iterations, but no single experiment is gated by per-metric dominance. The formula decides.

### 2.4 Priority metrics (weighted focus)

Composite weighting already puts `nDCG@20` (×0.50) and `LLM-judge` (×0.30 after 1-based normalization) at ~80% of the total score. We explicitly encode this in how experiments are prioritised:

| Metric | Share of composite | Priority | Implication for candidate picking |
|---|---|---|---|
| `nDCG@1` | (part of leaderboard's primary recsys score) | **P0** | Any retrieval/rerank change is evaluated on nDCG@1 and nDCG@20 — not just nDCG@10. @1 rewards pushing the GT track to rank 1 |
| `nDCG@10` | implicit | **P0** | Standard retrieval quality proxy |
| `nDCG@20` | 0.50 × composite | **P0** | The leaderboard's headline retrieval metric |
| `LLM-judge (Gemini)` | 0.075 per raw point × 4 | **P0** | Biggest response-side lever; **Gemini-measured** locally before shipping (Track J) |
| `lexical_diversity` | 0.10 | P2 | Part of composite; never optimised directly — it moves as a side-effect of retrieval/response variety |
| `catalog_diversity` | 0.10 | P2 | Driven incidentally by retrieval variety; we don't optimise directly |

**Operational rule**: when choosing between two candidates of comparable effort, prefer the one that moves a P0 metric. When a Wave-4 queue is built, the top 50% of slots must be P0-targeting candidates.

### 2.5 Local evaluation harness — retrieval side only; LLM-judge comes from online scoring

We **do not run any LLM judge locally** (no local hardware for Qwen/Llama as judge; no Gemini API budget in the current phase). The LLM-judge term of the composite is observed only via CodaBench blind submissions. Local eval scores everything else — which is still most of the iteration signal, because the retrieval metrics (nDCG@{1,10,20}, CatDiv, LexDiv) together determine a fully-known 70% of the composite weight and are deterministic.

#### 2.5.1 What the harness computes locally

A new script `scripts/local_eval.py` produces the **retrieval-side composite** on dev.

| Metric | How computed locally | Local? |
|---|---|---|
| `nDCG@{1,10,20}` | `music-crs-evaluator/evaluate_devset.py` — same code path CodaBench's retrieval scorer uses | ✓ |
| `catalog_diversity` | `metrics_diversity.py:compute_catalog_diversity` | ✓ |
| `lexical_diversity` | `metrics_diversity.py:compute_lexical_diversity` | ✓ |
| `LLM-judge (Gemini)` | — | ✗ online only (CodaBench) |

Two derived quantities are reported:

- **`composite_retrieval`** — the §1.4 formula with the LLM-term zeroed out. Covers `0.50·nDCG@20 + 0.10·CatDiv + 0.10·LexDiv = 0.70` of the max composite.
- **`composite_projected`** — `composite_retrieval + 0.30 · (LLM_last_known − 1) / 4`, where `LLM_last_known` is the most recent blind LLM score of the stack's response-branch (defaulting to B-champ's blind LLM if none exists yet).

`composite_projected` is a planning estimate, not a truth. We never ship a stack on that alone (see §2.5.3).

#### 2.5.2 Harness interface

```
scripts/local_eval.py --predictions exp/inference/dev/<exp_id>.json \
                      --compare-against B-champ
# outputs:
#   - scores JSON at exp/scores/dev/<exp_id>.json (retrieval metrics + composite_retrieval + composite_projected)
#   - diff vs B-floor and B-champ (raw + formatted)
#   - decision string: "PROMOTE to blind" / "ITERATE more" / "REJECT"
#   - appends row to submissions_log.md tagged [dev-local]
```

On every new dev run, local_eval runs automatically (called by `scripts/run_experiment.py`).

#### 2.5.3 Two gate tiers — local and online

We collapse the earlier T1/T2/T3 scheme to just two tiers since there's no intermediate Gemini signal.

| Tier | Where | Cost | Used for |
|---|---|---|---|
| **L1 — Local retrieval-side eval** | Local dev run | $0 | Every candidate; fast-fail on retrieval regressions; promotion to blind requires `composite_retrieval ≥ B-champ.composite_retrieval + σ` |
| **L2 — Online blind eval** | CodaBench Blind-A/B submission | 1 submission slot | Final judge; produces the true composite including the Gemini LLM term. Budget-gated per §2.6 |

A candidate goes L1 → L2. We never "pre-grade" the LLM term locally.

#### 2.5.4 Handling the unseen LLM-judge term

Since the LLM-judge term is invisible locally, we rely on proxies and discipline, not direct measurement:

- **Response-branch change-freeze between blinds**: when we ship a blind submission, we lock the response-generation branch (prompt, model, few-shot recipe). Only retrieval-axis changes iterate locally after that. This way the next blind's ΔLLM can be attributed to at most one response-axis change at a time.
- **Response changes go blind eventually or not at all**: if a response-axis candidate doesn't eventually earn a blind slot, we reject it — we have no way to validate it.
- **Qualitative inspection**: every dev run's `predicted_response` sample (top-10 and bottom-10 by retrieval agreement with GT) is read by the human. Obvious failure modes (hallucinations, apologies, formulaic AI-speak, template-leak) are caught without a judge.
- **Blind-scored responses become the reference corpus**: every scored blind submission's (query, response, score) triples are archived in `documents/blind_responses_scored.md`. Patterns from worst-scoring rows are captured in `agent_memory.md` Judge-behavior-notes and drive the next prompt tweaks.

#### 2.5.5 Local-vs-blind calibration tracking

Every blind submission that returns a score creates a `(composite_retrieval_local, composite_blind)` data point. We track this in `agent_memory.md` under **Local→Blind calibration**:

- After 3 paired points, compute the systematic blind-vs-local offset and the `LLM_term_observed` distribution across submissions. This tells us the empirical LLM-term floor / ceiling of our response branch.
- If the retrieval-side metrics agree closely (local vs blind) but composite diverges, the LLM term is moving — diagnose via response inspection.
- If retrieval-side metrics disagree, audit: schema drift, split leakage, or a bug in our local pipeline.

### 2.6 Blind submission budget

Blind submissions are the **only source of the LLM-judge signal**. This makes the budget tighter, not looser — every slot is both our eval and our progress.

| Rule | Value |
|---|---|
| Max Blind-A submissions per week | 3 (leave headroom for CodaBench daily caps and for the final retrain) |
| Mandatory prerequisite to ship | L1 local retrieval-side composite ≥ B-champ.composite_retrieval + σ (§2.3.2) |
| Response-branch discipline | Each blind submission should change **at most one** response-branch axis from the previous one, so the ΔLLM-term is attributable |
| Reserve | 2 Blind-A slots held until last 2 weeks of Blind-A phase for final-stack retrain (train ∪ dev) per E-4 |
| Blind-B submissions | Separately budgeted — reset when Blind-B opens 2026-06-15; same 3/week cap |

Enforcement: `scripts/validate_prediction.py` hard-refuses packaging if the weekly cap would be exceeded, and emits a warning if the new submission changes both retrieval and response branches vs the previous blind (potential attribution violation).

#### 2.6.1 Submission-slot prioritisation

When multiple candidates pass L1, pick the blind slot owner by this ranking:

1. Candidate whose change is **response-branch** and where no recent blind data exists (we need to see the LLM term move)
2. Candidate with the largest L1 `Δcomposite_retrieval` vs current B-champ
3. Candidate whose change is hypothesized to move LLM-judge based on agent memory Judge-behavior-notes
4. Tie-break: cheaper / smaller change first (cheap-first rule §8.4)

---

## 3. System Architecture

### 3.1 Pipeline (modular)

```
Query + ChatHistory + UserProfile + (ConversationGoal)
         │
         ▼
 ┌───────────────────────┐
 │ QueryRewriter (opt.)  │  → list[str] (1-N rewrites)
 └──────────┬────────────┘
            ▼
 ┌───────────────────────┐
 │ Retriever Specialists │   BM25 / Dense(text/audio/image/CF) / SID-gen / Popularity
 │  N running in parallel│   each returns list[(track_id, score)]
 └──────────┬────────────┘
            ▼
 ┌───────────────────────┐
 │ Fusion                │   RRF / wRRF / LambdaMART / Learned stacker
 └──────────┬────────────┘
            ▼
 ┌───────────────────────┐
 │ Reranker (opt.)       │   Cross-encoder / Listwise-LLM / Rank-R1
 └──────────┬────────────┘
            ▼
        top-20 track_ids
            │
            ▼
 ┌───────────────────────┐
 │ ResponseGenerator     │   Qwen + persona + (dynamic few-shot) + grounding
 └──────────┬────────────┘
            ▼
   predicted_response
```

### 3.2 Interface contracts

All modules subclass a tiny interface. No ABC required — duck typing suffices and matches existing `mcrs/` style.

```python
# mcrs/retrieval_modules/<name>.py
class Retriever:
    def text_to_item_retrieval(self, query: str, topk: int) -> list[str]: ...
    def batch_text_to_item_retrieval(self, queries: list[str], topk: int) -> list[list[str]]: ...
    # Optional: score-aware version for fusion/reranking
    def text_to_item_retrieval_with_scores(self, query, topk) -> list[tuple[str, float]]: ...

# mcrs/query_rewriters/<name>.py
class QueryRewriter:
    def rewrite(self, query: str, context: dict) -> list[str]: ...  # 1+ rewrites

# mcrs/rerankers/<name>.py
class Reranker:
    def rerank(self, query: str, candidates: list[str], user_profile: dict, topk: int) -> list[str]: ...

# mcrs/lm_modules/<name>.py (extended interface)
class ResponseGenerator:
    # NOTE: takes top-N (not top-1) — fixes current crs_baseline.py:121 bottleneck
    def generate(self, system_prompt: str, chat_history: list[dict], candidates: list[dict]) -> str: ...

# mcrs/aggregators/<name>.py (new module)
class Aggregator:
    def combine(self, specialist_outputs: dict[str, list[tuple[str, float]]]) -> list[str]: ...
```

### 3.3 How to plug a new module in

1. Add the class to the appropriate `mcrs/<module>/<name>.py`
2. Register it in the module's `__init__.py` factory (`load_retrieval_module`, `load_lm_module`, etc.)
3. Write a YAML config in `music-crs-baselines/config/{NNN}-{track}-{change}-{split}.yaml`
4. Write a unit test in `tests/test_<module>.py`
5. Run smoke test via `scripts/run_experiment.py --config <yaml> --smoke`
6. Run full experiment via `scripts/run_experiment.py --config <yaml>`

### 3.4 Extension points already in the repo

- `/Users/orrimoch/PythonProjs/recsys2026/music-crs-baselines/mcrs/retrieval_modules/__init__.py:17` — retrieval factory; add one line per new retriever
- `/Users/orrimoch/PythonProjs/recsys2026/music-crs-baselines/mcrs/lm_modules/__init__.py:3` — LM factory
- `/Users/orrimoch/PythonProjs/recsys2026/music-crs-baselines/mcrs/crs_baseline.py:121,170` — **top-1-to-LLM bottleneck**; change to pass top-N candidates to the response generator; propagate `top_n_for_prompt` through config
- `/Users/orrimoch/PythonProjs/recsys2026/music-crs-baselines/mcrs/rerankers/__init__.py` — empty stub; first task: add a minimal factory
- `/Users/orrimoch/PythonProjs/recsys2026/music-crs-baselines/mcrs/query_rewriters/__init__.py` — empty stub
- `/Users/orrimoch/PythonProjs/recsys2026/music-crs-baselines/mcrs/embedders/__init__.py` — empty stub
- `/Users/orrimoch/PythonProjs/recsys2026/music-crs-baselines/mcrs/retrieval_modules/dense_precomputed.py:106-171` — embedding imputation contract; reuse

---

## 4. Experiment Framework & Logging Protocol

### 4.1 Experiment config naming

```
{NNN}-{track}-{change}-{split}.yaml
```

- `NNN` — zero-padded monotonic ID (`020`, `021`, …) starting at `020` (below 020 reserved for existing stock baselines)
- `track` — `R` (retrieval), `Q` (query rewrite), `X` (rerank), `C` (response), `G` (generative SID), `E` (ensemble), `J` (judge calibration)
- `change` — short kebab-case tag (`bge-m3`, `cmqr-3-rewrites`, `qwen7b-listwise`)
- `split` — `dev` or `blindA` or `blindB`

Examples: `020-R-bge-m3-dev.yaml`, `027-X-qwen7b-listwise-dev.yaml`, `045-G-rqvae-cf-bpr-dev.yaml`

### 4.2 Experiment runner

New script `scripts/run_experiment.py` at repo root. Pseudo-code:

```
parse --config <yaml> --smoke/--full --split dev/blindA
1. validate config (track_split_types, required fields)
2. if --smoke: run first 5 sessions; assert valid prediction.json
3. run full inference via music-crs-baselines/run_inference_{dev,blindset}.py
4. if --split dev:
   4a. call scripts/local_eval.py (L1 retrieval-side composite per §2.5.3)
   4b. harness writes composite_retrieval + composite_projected + per-metric deltas vs B-floor / B-champ
5. capture all metrics + config hash + git-sha; append to submissions_log.md
6. if new champion on dev retrieval (composite_retrieval > B-champ.composite_retrieval + σ): update benchmarks.md, move B-champ
7. if --split blindA: enforce §2.6 submission-budget check + response-axis attribution warning BEFORE packaging
8. write narrative skeleton to experiments_log.md for human fill-in
```

It is a thin wrapper — the heavy lifting is done by the existing `run_inference_*`, `evaluate_devset.py`, and `scripts/local_eval.py`. The LLM-judge term is observed only via blind submissions (§2.5); local_eval does not call any judge.

### 4.3 Logging files — schemas

**`documents/experiments_log.md`** — narrative log, one entry per experiment:

```markdown
### Exp NNN-{track}-{change} — {one-line headline} — YYYY-MM-DD

- **Hypothesis**: what we expect to move and why (mechanism)
- **Axis**: retrieval | reranker | query-rewrite | response-prompt | response-model | train-obj | fusion | aggregator
- **Config**: path to YAML
- **Code**: path(s) to any new modules, git sha
- **Smoke result** (5 rows): pass/fail + output sample
- **Full result** (dev / blindA):
  - nDCG@{1,10,20}, CatDiv, LexDiv, LLM-judge, Composite
  - Δ vs B-floor / B-champ / B-target (per metric)
- **Lessons**: what we learned (positive OR negative)
- **Verdict**: new champion / shipped but not champion / rejected / dead end
- **Suggests next**: 1-3 follow-up experiment IDs
```

**`documents/submissions_log.md`** — tabular:

```markdown
| exp_id | split | nDCG@1 | nDCG@10 | nDCG@20 | CatDiv | LexDiv | LLM | Composite | date | status |
|---|---|---|---|---|---|---|---|---|---|---|
```

**`documents/benchmarks.md`** — top-3 shippable champions (updated on every new champion):

```markdown
## Current champions (updated after every scored submission)

| Rank | exp_id | Composite | nDCG@20 | LexDiv | LLM | Defining mechanism | Config |
|---|---|---|---|---|---|---|---|
| 🥇 | ... | ... | ... | ... | ... | ... | ... |
| 🥈 | ... | ... | ... | ... | ... | ... | ... |
| 🥉 | ... | ... | ... | ... | ... | ... | ... |
```

**`documents/next_candidates.md`** — ranked queue. Template in §10c (ML techniques catalog serves as the initial bank).

### 4.4 Runtime provenance

Every experiment run stores:

- Config YAML hash (content-addressable)
- git sha of `recsys2026` repo at run time
- Python version, torch version, transformers version (via `pip freeze > exp/runs/{exp_id}/freeze.txt`)
- Wall-clock duration, peak RAM/VRAM
- Raw `prediction.json` at `music-crs-baselines/exp/inference/{split}/{exp_id}.json`
- Raw score JSON at `music-crs-evaluator/exp/scores/{split}/{exp_id}.json`

---

## 5. Task Catalog

Tasks are identified with stable IDs. Each has `depends-on`, `effort`, `expected Δcomposite`, and an `acceptance criterion`. **No task is considered done until its acceptance criterion is demonstrably satisfied.**

Effort scale: `XS` ≤ 1h, `S` 1–4h, `M` 4–12h, `L` 12h–1d, `XL` >1d.

### Wave 0 — Infrastructure (gate for everything)

| ID | Task | Depends | Effort | Acceptance |
|---|---|---|---|---|
| W0-1 | Create empty log templates (`experiments_log.md`, `submissions_log.md`, `benchmarks.md`, `next_candidates.md`, `agent_memory.md`) with schema headers per §4.3 + §8.3.1 | — | XS | Files exist and conform to the schemas |
| W0-2 | Write `scripts/run_experiment.py` wrapping inference + eval + log write | — | S | `scripts/run_experiment.py --config <any existing yaml> --smoke` runs and writes a new row to `submissions_log.md` |
| W0-3 | Smoke test harness — 5-session subset inference path on dev | W0-2 | XS | Subset prediction file conforms to schema and scores in `evaluate_devset.py` without crash |
| W0-4 | `tests/` dir + `pytest.ini`; one stub test per module (retrieval/rerank/query-rewrite/response/judge) | — | S | `pytest tests/` runs and passes (with stubs initially) |
| W0-5 | `prediction.json` schema validator (`scripts/validate_prediction.py`) — 8 turns, 20 unique IDs, non-ASCII safe | — | S | Validator rejects known-bad fixtures and accepts the official baseline file |
| W0-6 | Reproduce `llama1b_bm25_devset` as anchor baseline via `run_experiment.py` | W0-2, W0-3 | XS | Dev scores match challenge repo (`nDCG@10 ≈ 0.0627`); first row in `submissions_log.md` |
| W0-7 | **Local eval harness** — `scripts/local_eval.py` per §2.5.2 producing `composite_retrieval` + `composite_projected` on dev; `scripts/run_experiment.py` calls it automatically after each dev inference | W0-2 | S | Running harness on the anchor baseline produces a composite row + a `[dev-local]` entry in `submissions_log.md` |
| W0-8 | **Noise floor measurement** (§2.3.2) — run anchor baseline twice with different seeds on the shuffled split; record `σ` for `composite_retrieval` | W0-7 | XS | Value of `σ` written into `agent_memory.md` under Prediction-calibration section |
| W0-9 | **Submission budget enforcement** — hard check in `scripts/validate_prediction.py` that refuses Blind packaging when weekly cap (§2.6) would be exceeded; also warns on response+retrieval dual-axis change | W0-5 | XS | Unit test: a mock "4th submission this week" is refused; a mock dual-axis change emits a warning |
| W0-10 | **Blind-response archive** — create `documents/blind_responses_scored.md` as an append-only archive of (query, predicted_response, blind_score) per scored submission, used for Judge-behavior pattern mining (§2.5.4) | W0-1 | XS | File exists with schema header |

Wave 0 is complete when: `pytest` green, `submissions_log.md` has at least `llama1b_bm25_devset` row, `benchmarks.md` B-champ is set, `agent_memory.md` is seeded with §10e dead-ends + noise floor σ, local eval harness produces `composite_retrieval` on dev, blind-response archive is ready.

---

### Track R — Retrieval (primary for Wave 1–3)

#### R-0 Baseline reproductions (already have configs — just run and log)

| ID | Config | Axis | Acceptance |
|---|---|---|---|
| R-0.1 | `002-bm25-field-expansion.yaml` | retrieval | dev row in log; smoke passes |
| R-0.2 | `007-rrf-bm25-dense-v1.yaml` | retrieval | dev row in log |
| R-0.3 | `009-wrrf-bm25-dense-v1.yaml` | retrieval | dev row in log |
| R-0.4 | `010-wrrf-bm25-dense-lyrics-v1.yaml` | retrieval | dev row in log |
| R-0.5 | `llama1b_bert_devset.yaml` | retrieval | dev row in log |
| R-0.6 | `005-dense-qwen3-metadata.yaml` | retrieval | dev row in log |

After R-0 we have a calibrated map of which existing retrieval knobs actually work on this data. Expected to take <1 day total (each config runs in ~30 min on M4).

#### R-1 New retrieval specialists

**Focus discipline (per user directive 2026-04-24):** do not sweep every dense embedder. Pick the ones below; defer the rest.

| ID | Name | Depends | Effort | Status | Notes |
|---|---|---|---|---|---|
| R-1.1 | **BGE-M3** self-built multi-granularity retriever (dense + sparse + ColBERT in one) | R-0 | M | **ACTIVE** | The single "new dense" retriever we run. Multi-granularity makes it stand in for separate BGE/E5/Stella attempts. Reference: `recent_papers_ideas.md` CL18 / BGE-M3 paper |
| R-1.4 | **cf-bpr user×item affinity** retriever for warm users | R-0 | S | **ACTIVE** | Free — precomputed user_emb · track_emb. Fallback to BM25 for cold. New `mcrs/retrieval_modules/cf_affinity.py` |
| R-1.5 | **CLAP audio-text** retrieval using `audio-laion_clap` precomputed embeddings | R-0 | S | **ACTIVE** | Free precomputed audio side; orthogonal modality. Query encoded by CLAP text tower; cosine against precomputed audio emb. ~1% empty → impute per existing contract |
| R-1.6 | **Popularity prior** retriever (per `conversation_goal`-conditioned) | R-0 | XS | **ACTIVE** | Cheap fallback specialist; useful as aggregator input |
| R-1.2 | E5-mistral-7b-instruct dense | R-0, Colab | M | DEFERRED | Marginal over BGE-M3; revisit only if R-1.1 shows signal AND retrieval is the composite bottleneck |
| R-1.3 | Stella-1.5B-v5 dense | R-0 | S | DEFERRED | Same reasoning as R-1.2 |
| R-1.7 | NV-Embed-v2 dense (stretch) | R-1.2 | M | DEFERRED | Only reached if multiple earlier embedders all show signal |

Each acceptance criterion: unit test passes (mock query → expected shape) AND integration run on dev produces valid `prediction.json` AND nDCG@10 ≥ 0.5 × B-floor (sanity: fully broken retriever would score ≈0).

#### R-2 Query rewriting (fills `mcrs/query_rewriters/`)

| ID | Name | Depends | Effort | Notes |
|---|---|---|---|---|
| R-2.1 | **CMQR** multi-query rewrite (single LLM call, N=3 rewrites, RRF fuse) | R-0, W0-2 | S | Highest EV per hour per `recent_papers_ideas.md §1`. Local Qwen 3B ok. New `mcrs/query_rewriters/cmqr.py` |
| R-2.2 | **HyDE-light** — LLM writes a hypothetical track TITLE (not passage), dense-retrieve against title embeddings | R-1.1 or R-1.3 | S | Avoids the v4-era `repetition_penalty` degeneracy by constraining to a short single output |
| R-2.3 | **Doc2Query** offline — Qwen-0.5B generates 3 hypothetical queries per track, append to BM25 text; reindex | R-0 | M | ~8h one-time on M4 or ~2h on Colab. Reusable across Blind-A + Blind-B |
| R-2.4 | **Rule-based preprocessor** — lowercase, stem, stopword-strip, accent normalize, WordNet synonym expand | — | XS | Cheap, complements BM25 directly. Zero hallucination risk |
| R-2.5 | **RA-Rec JSON state tracker** — LLM maintains preference JSON across turns; state used as BM25 keywords + dense input | R-0, R-2.1 | M | Tackles multi-turn conversational context loss. Conditional on observing turn-number regression in R-0 results |

#### R-3 Reranking (fills `mcrs/rerankers/`)

| ID | Name | Depends | Effort | Notes |
|---|---|---|---|---|
| R-3.1 | **BGE-reranker-v2-m3** cross-encoder top-40 → top-20 | R-0 | S | Classic "free" reranker lift. Slow on M4 (~30 min/80 rows) — acceptable for blind, marginal on dev |
| R-3.2 | **Qwen-7B listwise** RankZephyr-style prompt | R-0, Colab or long local run | M | Single LLM call/query; re-order 20 IDs in prompt |
| R-3.3 | **FIRST** first-token listwise (50% latency cut vs RankZephyr) | R-3.2 | S | Only if R-3.2 works but is too slow |
| R-3.4 | **ProRank** SLM reranker (0.5B–1B) — GRPO warmup | R-3.1, Colab | M | High EV-per-GPU-hour per paper; acts as an always-on reranker |
| R-3.5 | **Rank-R1** GRPO-trained Qwen-7B reranker | R-3.2, Colab A100 | L | Higher-ceiling version of R-3.2; requires training loop |

#### R-4 Learning-to-Rank (train-time)

| ID | Name | Depends | Effort | Notes |
|---|---|---|---|---|
| R-4.1 | **LightGBM LambdaMART** with `goal_progress_assessments` as graded relevance (93% coverage on train) | R-0, R-1.*, R-2.* | L | Features per candidate: per-field BM25, dense cosines, metadata match flags, popularity, user-demo match, tag overlap. High retrieval ceiling |
| R-4.2 | **Cross-encoder fine-tune** BGE-reranker on train `(query, gold)` pairs | R-3.1, Colab | L | In-domain tuning expected +0.03–0.08 nDCG per `recent_papers_ideas.md §2` |
| R-4.3 | **Query-rewriter training** — train Qwen-0.5B to output the BM25-optimal query, supervised by rank-of-gold | R-2.*, Colab | L | Rec-R1-style (`recent_papers_ideas.md §4`). Conditional on R-2 results plateauing |

#### R-5 Fusion

| ID | Name | Depends | Effort | Notes |
|---|---|---|---|---|
| R-5.1 | **RRF** standard (k=60) over all R-1 specialists | R-1.1, R-1.4, R-1.5 (≥3 retrievers) | XS | Baseline ensemble |
| R-5.2 | **Weighted RRF** grid search on dev | R-5.1 | S | Sweep 3–5 weight combos; pick devset-best |
| R-5.3 | **Learned stacker** — logistic regression on per-retriever ranks + metadata features | R-5.2 | M | Train on dev (leak-safe with k-fold if needed for hyperparams) |

---

### Track C — Response Generation (primary for Wave 2+)

We never ship a retrieval-only submission — a thoughtful response is worth ~30% of composite per §1.4.

**Technique catalog** (per working principle §0b and `feedback_prompt_engineering.md`). Every Track-C task combines a subset of these; the task name should name the active techniques.

| Shorthand | Technique | Where it applies |
|---|---|---|
| `chat-v2` | Custom `[system, user]` chat template — no fake-assistant-turn | MANDATORY for all post-C-0.3 tasks; fixes "I'm glad you enjoyed X" dead-end |
| `topN` | Multi-track context (top-3/5 into prompt, not top-1) | C-0.2 |
| `persona` | Persona framing ("well-read critic") | C-1.1; proven +0.40 LLM prior branch |
| `word-ban` | Negative examples / banned AI-speak words | C-1.1 |
| `ctx-struct` | Structured user profile slots + labelled candidate list | all C-* |
| `cot` | Chain-of-thought (hidden scratchpad) | C-2.5 |
| `cove` | Chain-of-Verification | C-2.2 |
| `grounding` | Forced metadata grounding (outlines/logits) | C-2.3 |
| `json-prose` | Structured JSON → prose rendering | C-2.4 |
| `step-back` | Intent classification → conditioned response | C-2.5 |
| `fs-static` | Curated static few-shot | C-1.x |
| `fs-nn` | Dynamic NN few-shot (MiniLM-CPU + Qwen-MPS) | C-1.3 |
| `fs-cache` | 50 hand-crafted semantic cache pairs | C-1.4 |
| `sampling` | Nucleus sampling T=0.4–0.7, top_p=0.9, no_repeat_ngram=3 | default decoding |
| `multi+judge` | Multi-candidate + pairwise self-judge | DEFERRED (needs local judge) |

**Wave 2 (current, informational)**: `chat-v1 (stock, bad) + top-1 (stock) + no prompt features` — known to hallucinate "I'm glad you enjoyed X". Running to establish a Qwen-1.5B floor. Do not ship to blind.

**Wave 3 pilot (first shippable)**: `chat-v2 + topN (top-3) + persona + word-ban + ctx-struct + sampling(T=0.6, top_p=0.9, no_repeat=3)`. No few-shot in v1 for attribution. CoVe / CoT / grounding layered in only after this clean baseline is benchmarked.

| ID | Name | Depends | Effort | Notes |
|---|---|---|---|---|
| C-0.1 | **Baseline Qwen3B + stock prompt** devset run | W0 | XS | Anchor; produces the first response-generating row in the log |
| C-0.2 | **Multi-track prompt** — change `crs_baseline.py:121` to pass top-3 (not top-1) tracks to the response generator | W0 | S | Fixes the single-biggest structural issue. Generalises the LM interface to take a list |
| C-0.3 | **Custom `[system, user]` chat template** — avoid stock template's fake-assistant-turn injection | C-0.2 | XS | Dead-end-fix lifted from prior iteration; cheap and orthogonal |
| C-1.1 | **Persona prompt** + generic-AI-speak word-ban (`"absolutely", "fantastic", "truly", "amazing"`) | C-0 | XS | Proven high-EV prompt change (prior iteration +0.40 LLM). A fresh-branch rebuild, not a copy |
| C-1.2 | **Concrete citation directive** — require 1+ specific metadata reference per track in the response | C-1.1 | XS | Lifts Explanation Quality; low risk |
| C-1.3 | **Dynamic NN few-shot** from train — MiniLM encode train queries → retrieve top-3 at inference → inject as exemplars. **Force MiniLM to `device="cpu"` when Qwen is on MPS** (dead-end from prior iteration) | C-1.1 | M | +0.2–0.4 LLM per `recent_papers_ideas.md` |
| C-1.4 | **Semantic cache** of 50 hand-crafted (query, ideal_response) pairs, encoded with MiniLM | C-1.3 or independent | M | Alternative to C-1.3 if train distribution is mismatched to blind |
| C-2.1 | **Multi-candidate + pairwise self-judge** (Qwen-7B picks best of 3 greedy-vs-sampled) | J-D2 (local judge) reactivation | M | **DEFERRED — requires a local judge, which we don't run in this phase. Revisit on J-D reactivation.** |
| C-2.2 | **Chain-of-Verification (CoVe)** anti-hallucination | C-0 | S | Draft → extract claims → verify vs metadata → revise |
| C-2.3 | **Forced metadata grounding** via `outlines` logits constraint | C-0 | S | Constrains response to contain exact `track_name` + `artist_name`; zero hallucination |
| C-2.4 | **JSON-structured → prose** via template rotation | C-0 | S | Lifts Lex Div while preserving explanation structure |
| C-2.5 | **Step-back prompting** — classify intent, then condition generation | C-0 | S | Handles ambiguous mood/name-lookup/era queries |
| C-3.1 | **Qwen2.5-7B response** (fp32 on MPS or Colab bf16) — stacks on any prompt recipe | C-1.1 | S | **Dead-end warning from prior iteration**: 7B + rigid few-shot → overfits template → LLM regression. Only safe when prompt explicitly requires diverse structure / bans formulaic openers |
| C-3.2 | **Qwen2.5-14B** (Colab only; fp32 ~56GB > M4 budget) | Colab, C-3.1 | M | Stretch; conditional on C-3.1 not regressing |

---

### Track J — LLM-Judge (online via CodaBench; local judge DEFERRED)

The blind leaderboard judge is **Google Gemini**. We do not run any local LLM-judge in this phase — no local hardware, no Gemini API spend. The LLM-judge term is observed exclusively via blind submissions (§2.5).

What remains active in Track J:

| ID | Name | Depends | Effort | Notes |
|---|---|---|---|---|
| J-A1 | **Blind-response mining** — after every scored blind submission, extract (query, predicted_response, score) triples into `documents/blind_responses_scored.md`; pattern-mine worst-10 / best-10 responses into `agent_memory.md` Judge-behavior-notes | blind scoring | S (recurring) | Our only real signal on Gemini's preferences |
| J-A2 | **Rubric hypothesis registry** — maintain a list of "hypothesized Gemini rewards / penalties" in `agent_memory.md`; each hypothesis is validated or falsified by the next blind submission that tests it | J-A1 | XS (recurring) | Drives prompt engineering (Track C) candidate generation |
| J-A3 | **Response-branch change-freeze protocol** — between blinds, lock the response branch unless that blind is explicitly testing a response change (§2.6 rule). Attribution rule | — | — (discipline) | Prevents coupled changes that make the ΔLLM-term unattributable |

Deferred (parked, revisit if priorities change):

| ID | Name | Status | Revisit trigger |
|---|---|---|---|
| J-D1 | Gemini API integration (`mcrs/judges/gemini.py`) | DEFERRED | If API budget becomes available (~$50/month) — would enable a T2 dev-side pre-screen |
| J-D2 | Local Qwen-7B as judge | DEFERRED | If local GPU access becomes available (e.g. Colab A100 sessions specifically for judge runs) — would enable fast dev-side calibration |
| J-D3 | Calibration Gemini↔Qwen (ρ correlation) | DEFERRED | Depends on J-D1 + J-D2 |
| J-D4 | Position-bias diagnostics on local judge | DEFERRED | Depends on J-D2 |
| J-D5 | Pre-submission Gemini gate | DEFERRED | Depends on J-D1 |
| J-D6 | Gemini as training reward (for KTO/S-DPO/Rank-GRPO) | DEFERRED | Depends on J-D1 + GPU training infra |

Deferred tasks remain named so we can reactivate them without re-architecting. `mcrs/judges/` module directory is created at W0 but populated only on reactivation.

---

### Track E — Ensemble / Aggregator (final composition)

| ID | Name | Depends | Effort | Notes |
|---|---|---|---|---|
| E-1 | **Specialist registry** — YAML lists N specialists + weights + fusion method | R-1, R-5 | S | Central config that describes a full stack in one file |
| E-2 | **Learned aggregator** — logistic regression on per-specialist (rank, score) features; train on dev | R-5.3 | M | Generalises wRRF; admits arbitrary features (popularity, user-type, …) |
| E-3 | **`prediction.json` validator + zip packager** (CodaBench-compliant) | W0-5 | XS | Rejects bad files; writes `prediction.zip` with singular `prediction.json` at root |
| E-4 | **Train+Dev retrain helper** — for final Blind-B submission retrain on (train ∪ dev) — only after Blind-A scoring confirms dev-fit transfers | R-*, C-*, G-* | S | Matches prior-iteration protocol (`project_submission_prep.md`) |

---

### Track G — End-to-End Generative (Semantic IDs) — Wave 4+, Colab-async, does NOT block first Blind-A ship

| ID | Name | Depends | Effort | Notes |
|---|---|---|---|---|
| G-0 | **Paper sweep** — read `TIGER`, `Text2Tracks`, `GRID`, `LETTER`, `LIGER`. Write 1-page note in `documents/research/sid_plan_notes.md` | — | S | Foundational literacy |
| G-1.1 | **RQ-VAE over cf-bpr** (128d → 4 levels × 256 codes). Colab notebook `colab/Train_RQVAE_CFBPR.ipynb` | G-0, Colab | M | Text2Tracks-style; CF-based SIDs beat title decoding per paper |
| G-1.2 | **RQ-VAE over metadata-Qwen3** (1024d → 4 levels × 256) | G-0, Colab | M | Compare CF vs text SIDs on retrieval quality |
| G-1.3 | **LETTER** — RQ-VAE + contrastive CF alignment + diversity loss | G-1.1, Colab | L | Advanced; only if G-1.1 shows promise |
| G-2.1 | **SFT Qwen-3B** on `(query → SID sequence)` using train conversations; Colab A100 | G-1.*, Colab | L | Follow `recsys2026-lora-tutorial` 9-cell notebook pattern exactly |
| G-2.2 | **Constrained beam decoding** over catalog trie (local inference) | G-2.1 | S | Prevents hallucinated SIDs |
| G-2.3 | **LIGER hybrid** — SID-gen + dense fallback for cold tracks (tracks never emitted by SFT) | G-2.2, R-1.* | M | Expected to beat both pure gen and pure dense per paper |
| G-3.1 | **KTO** on `(GT / not-GT)` binary signal (cheap first RL pass) | G-2.1, Colab | M | Binary signal = cheapest alignment; no paired preferences |
| G-3.2 | **S-DPO** with hard BM25 negatives | G-3.1, Colab | L | 1-positive-N-negative Plackett-Luce loss |
| G-3.3 | **Rank-GRPO** (Netflix 2026) — rank-position credit assignment, geometric-mean importance ratio | G-3.2, Colab A100 | XL | Highest-ceiling training method. Only pursued if G-3.1/G-3.2 validate the SID approach |

Track G joins the ensemble at E-1 as another specialist; it does NOT replace the retrieval pipeline.

---

## 6. Colab / GPU Workflow

### 6.1 Pattern (copy from `recsys2026-lora-tutorial`)

For every GPU-heavy task:

1. One `.ipynb` under `colab/` — one notebook per task class (LoRA SFT, RQ-VAE train, cross-encoder fine-tune, Rank-GRPO)
2. 9-cell structure (ref: `recsys2026-lora-tutorial/colab/Train_LoRA_Colab.ipynb`):
   - Cell 0: markdown — runtime instructions ("Runtime → A100 GPU")
   - Cell 1: `!nvidia-smi` sanity
   - Cell 2: `!git clone` this repo (force-fresh) — print commit SHA
   - Cell 3: `!pip install -q -r requirements.txt` + import sanity
   - Cell 4: `os.environ['HF_TOKEN'] = 'hf_...'` (optional, for faster HF pulls)
   - Cell 5: main training invocation via `SKIP_*` env-var-gated `bash run_pipeline.sh` OR a direct python call to `lora/train_*.py`
   - Cell 6: `shutil.make_archive(...)` zip adapter / codebook
   - Cell 7a: `from google.colab import files; files.download(...)`
   - Cell 7b (commented): mount Google Drive + copy
   - Cell 8: markdown — how to pick up locally (`SKIP_DATASET=1 SKIP_TRAIN=1 ./run_pipeline.sh`)
3. Output artifact naming: `lora_adapters/{name}/final_adapter/` OR `codebooks/{name}/rqvae.pt`
4. Local inference reloads via `PeftModel.from_pretrained(...)` OR `torch.load(...)`

### 6.2 `run_pipeline.sh` orchestrator

Mirror `recsys2026-lora-tutorial/run_pipeline.sh` in `recsys2026/`. Stages:

| Stage | Purpose | SKIP_* var |
|---|---|---|
| 1 | venv setup | `SKIP_VENV` (Colab sets =1) |
| 2 | data download | `SKIP_DATASET` |
| 3 | GPU training (LoRA / RQ-VAE / rerank fine-tune / Rank-GRPO; dispatched by `STAGE`) | `SKIP_TRAIN` |
| 4 | local inference (CPU/MPS) | `SKIP_INFERENCE` |
| 5 | zip + validate | `SKIP_ZIP` |

Colab runs stages 1–3 (or just 3), local runs stages 4–5.

### 6.3 MPS gotchas (hardcoded as pre-flight assertions)

- Qwen batch generation on MPS: **fp32 only** (fp16 → NaN / `!!!!!`)
- Two HF models simultaneously on MPS deadlock. Fix: force secondary model (e.g. MiniLM for few-shot retrieval) to `device="cpu"`
- When loading Qwen-7B locally: float32 ≈ 30GB — fits 48GB M4 with headroom, but no room for a second model
- Qwen 14B float32 ≈ 56GB > 48GB M4 → Colab only or 4-bit quant

These rules encoded in `scripts/preflight.py` — called by `run_experiment.py` at start.

---

## 7. Testing & Verification Discipline

### 7.1 Gates

| Gate | Trigger | What it checks |
|---|---|---|
| Unit tests | Every module add or change | `pytest tests/` green. Mock input → shape-checked output |
| Smoke run | Every config before full run | 5-session subset completes without exceptions; prediction validates |
| Schema validator | Before every submission | 80 rows (blind-A) or 1000 sessions × 8 turns (dev); 20 distinct IDs per row; no dupes; non-ASCII safe |
| Pre-submission qualitative review | Every blind candidate | Human skim of 10 predicted_response samples for obvious failure modes (apologies, hallucinations, AI-speak, template-leak) per §2.5.4. No local judge available |
| Local eval L1 | Every dev inference completes | Retrieval-side composite (§2.5.1). Free. Fast-fail on retrieval regressions |
| Submission budget | Before every blind package | Hard check per §2.6 (3/week cap, 2 reserved for final retrain). Refuses packaging if exceeded |
| Attribution-warning | Before every blind package | Warn if both retrieval and response branches changed vs previous blind (§2.6 attribution rule) |
| Champion promotion (local) | After L1 | `composite_retrieval ≥ B-champ.composite_retrieval + σ` (§2.3.2). Per-metric deltas logged for agent memory but not gating |
| Champion promotion (blind) | After CodaBench return | `composite_blind ≥ B-champ.composite_blind + σ`. Pair the score with retrieval-side local for Local→Blind calibration (§2.5.5) |

### 7.2 Test structure (`tests/`)

```
tests/
  __init__.py
  conftest.py                    # pytest fixtures: minimal dataset, 5-row subset, track catalog stub
  test_data_loaders.py           # HF dataset load + impute contract
  test_retrieval_modules.py      # one test per retriever: mock query → list[str] len ≤ topk, no dupes
  test_rerankers.py
  test_query_rewriters.py
  test_response_generator.py     # prompt builder + output shape
  test_aggregator.py
  test_prediction_validator.py   # bad fixtures rejected, good accepted
  test_submission_packager.py    # zip format
  test_metrics.py                # reproduce known values from a fixture devset
```

Each new module's PR adds at least one row to the relevant test file.

### 7.2a Testing discipline — one integration test per wave (MANDATORY)

**User rule (2026-04-24):** every wave closes with an integration test added under `tests/test_wave{N}_integration.py`. This is a gate, not optional — no wave is "done" until the integration test exists and passes.

What "integration" means per wave:
- **Wave 0** (done): config → prediction → evaluator → composite → log round-trip (see `tests/test_wave0_integration.py`)
- **Wave 1**: retrieval baseline sweep — test that any new config can be scored end-to-end and appears in `submissions_log.md`
- **Wave 2**: first strong stack — test that the chosen retriever + response-gen pipeline produces a valid `prediction.json` that passes the schema
- **Wave 3**: first Blind-A ship — test that the E-3 packager produces a zip with exactly one `prediction.json` at root AND budget check runs
- **Wave 4+**: one integration test per track (R / C / E / G) as they activate

Unit tests are separate and land with every module (§7.2). Integration tests are per-wave and exercise the **seams** between modules.

### 7.3 Integration smoke — canonical command

```
# from recsys2026/
./scripts/run_experiment.py --config music-crs-baselines/config/<NNN>-<name>.yaml --smoke --split dev
```

Expected output: 5-session `prediction.smoke.json` + inline metric printout. Full run replaces `--smoke` with `--full`.

### 7.4 Open-source first — no reinventing the wheel

**Mandatory checklist before writing any new module** (retriever, reranker, query-rewriter, SID trainer, etc.):

1. **Paper repo** — every paper in `documents/research/recent_papers_ideas.md` has a code link. Check it first. Examples of known-good repos (already in §10b mapping):
   - `castorini/rank_llm` (RankZephyr, FIRST)
   - `ielab/llm-rankers` (Rank-R1)
   - `snap-research/GRID` (generative SID toolkit)
   - `facebookresearch/liger` (LIGER hybrid)
   - `RUCAIBox/LC-Rec` (LLaMA/Qwen → SID recipe)
   - `chenyuxin1999/S-DPO` (softmax DPO for rec)
   - BGE family: `FlagOpen/FlagEmbedding` (BGE-M3, BGE-reranker-v2-m3)
   - `castorini/pyserini` / `bm25s` (BM25 toolkits)
2. **Hugging Face** — search for `{task}` on Hub's *Models*, *Spaces*, and *Datasets*. Many SOTA components ship as HF checkpoints with `transformers`-compatible loaders.
3. **Popular toolkits** — `sentence-transformers`, `rerankers` (AnswerDotAI), `outlines` (constrained decoding), `trl` (GRPO/DPO/KTO/SFT), `peft` (LoRA), `LightGBM` (LambdaMART).
4. **Papers-with-code** — often surfaces smaller forks with better docs than the original paper's repo.
5. **Our own `recsys2026-lora-tutorial`** — reusable patterns for Colab + LoRA.

**Decision tree**:

```
Is there an OSS implementation of the exact method?
├─ Yes & license OK & maintained → adapt it (wrap our interface around it)
├─ Yes but deprecated / bad code → port minimal subset, test against paper numbers
└─ No → implement from scratch (document why in the task note)
```

**Record in the experiment log**: every entry in `experiments_log.md` includes a line `Code origin: {github.com/... | HF: ... | from scratch, reason=...}`. This keeps provenance auditable and prevents silent reinvention.

**Adaptation budget rule**: if porting an OSS component takes >1.5× the effort of writing from scratch, stop porting and note why. Usually signals a deep API mismatch that makes the port brittle.

---

## 8. Iteration Loop & Candidate-Generation Protocol

### 8.1 The loop (run after every LOCAL dev eval OR blind scoring)

**Retrieval-axis iteration runs off local `composite_retrieval` (§2.5); response-axis iteration runs off blind submissions.** Most loop turns fire locally. Blind turns are rarer but carry more information per event (they give us the only LLM-judge signal we have). The loop below applies to both — the difference is what data feeds into agent memory:

- After local dev eval: update retrieval-axis memory (Validated mechanisms, Dead ends, Prediction calibration on retrieval metrics)
- After blind scoring: additionally update Judge-behavior-notes (from the blind response archive §2.5.4), Local→Blind calibration, and response-axis memory

1. **Decompose the composite delta**. Which term moved? Was the direction as predicted? Magnitude? Identify the *one* axis that explains the largest share of the change. If multiple axes moved (signal that the experiment violated the one-axis rule), note it as an ablation debt.
2. **Update logs + agent memory** (§8.3.2):
   - Append tabular row to `submissions_log.md`
   - Append narrative entry to `experiments_log.md` per the §4.3 template
   - If new champion: update `benchmarks.md` (cascade down, evict prior #3)
   - **Update `agent_memory.md`**: add to Validated mechanisms / Dead ends / Prediction calibration / Judge behavior notes / Cross-correlations as applicable
3. **Regenerate `next_candidates.md`** (rebuild, don't patch) — now reading `agent_memory.md` per §8.3.3:
   - Filter out candidates matching Dead-end entries
   - For each candidate, recompute `EV = expected_Δcomposite × p(success) / effort_hours`, then apply §8.3.3 weighting (×1.5 if matches validated mechanism)
   - Apply Prediction-calibration bias (if axis X was consistently over-estimated by 40%, shrink predictions by 40%)
   - **Boost** candidates whose mechanism was validated by the latest result
   - **Demote** candidates sharing a disproven mechanism
   - **Add** new candidates from failure analysis (qualitative inspection of worst-5 rows, including Gemini's explanation on those rows when available)
   - **Refill** from §10c ML catalog when queue < 10 entries
   - Enforce **§8.4 cheap-first rule** — sort top-5 by effort ascending within EV tiers
   - Output: a fresh `next_candidates.md` ranked by EV, with a one-line rationale per entry
4. **Queue discipline**: always have ≥1 experiment running + ≥3 prepped (configs written, smoke passed). **Promotion to blind**: only candidates that pass L1 with `composite_retrieval > B-champ.composite_retrieval + σ` are eligible, AND the blind slot prioritisation rule §2.6.1 applies (response-branch candidates get priority when LLM-term data is stale).

### 8.2 Top-3 historical best, other resources, papers — the candidate generator

Candidates come from four streams, merged each iteration:

| Stream | Source |
|---|---|
| Historical best (top-3 champs) | `benchmarks.md` — stack additive changes onto the current champion's config |
| Unused data signals | §10d table — each unused column = a candidate experiment |
| Paper ideas | `documents/research/recent_papers_ideas.md` — already EV-ranked per paper-vs-task fit |
| Failure-mode analysis | Qualitative review of worst-N rows from the most recent scored run |

### 8.3 Agentic loop memory feedback — how the system learns across experiments

The log files (§4.3) capture point-in-time results. The **agentic memory** is the meta-layer that lets the iteration loop actually learn: which mechanisms worked, which assumptions were wrong, which predictors of success are reliable. It lives at `documents/agent_memory.md` (new, created in W0-1) and is updated every iteration.

#### 8.3.1 Structure of `agent_memory.md`

```markdown
# Agent memory — cross-experiment learnings

## Validated mechanisms (keep doing)
Each entry: rule + supporting exp_ids + confidence (low/med/high)
Example: "Persona prompt lifts Gemini LLM-judge by ≥0.2 — confirmed by C-1.1 dev (+0.3) and C-1.1-blindA (+0.4). Confidence: high."

## Dead ends (don't retry)
Mirrors §10e but adds any new dead ends from this branch's experiments.

## Open questions / hypotheses
Unresolved beliefs that need testing. Each has a "test by" exp ID.

## Prediction calibration
Table: predicted Δcomposite vs actual Δcomposite, per axis. Reveals systematic over/under-estimation.

## Judge behavior notes (Gemini-specific)
What Gemini appears to reward/penalise on THIS task. Grows from **blind-response mining** (J-A1) — analysing our own scored blind responses, since we don't run a local judge.

## Cross-correlations
"When axis X moves, axis Y tends to move in direction Z." E.g. retrieval changes that shuffle top-3 → LLM-judge moves with them (coupled experiments).
```

#### 8.3.2 Update triggers

After every scored submission (dev or blind):

1. **Add to Validated mechanisms** if result supports a previously untested hypothesis.
2. **Add to Dead ends** if result falsifies a previously entertained hypothesis.
3. **Update Prediction calibration** — record `(predicted_Δ, actual_Δ)` per axis.
4. **Update Judge behavior notes** if running Gemini on new output types surfaced new patterns (e.g. "Gemini prefers responses that cite release year when available").
5. **Update Cross-correlations** if a side-metric moved unexpectedly with the main axis.

#### 8.3.3 Read triggers

Before generating the next `next_candidates.md`:

1. **Filter**: any candidate whose mechanism matches a Dead-end entry is removed.
2. **Weight**: candidate EV = base_EV × `1 + 0.5 · matches_validated_mechanism − 0.3 · matches_open_question_adjacent`.
3. **Calibrate**: candidate's "expected Δ" is adjusted by Prediction-calibration bias for that axis.
4. **Route**: if a candidate targets LLM-judge, pull from Judge-behavior-notes to draft the prompt or eval criterion.

#### 8.3.4 What the agent (Claude / human) uses it for

- At session start: read `agent_memory.md` alongside `benchmarks.md` + `next_candidates.md`. Establishes prior.
- When proposing a new experiment: explicitly justify why it is not a Dead end.
- When interpreting a regression: first check Validated mechanisms — did we break one?
- When reviewing a run's `.expansions.json` or response samples: update Judge-behavior-notes with any new qualitative pattern.

**Minimum viable first version**: at W0-1, create `agent_memory.md` with six empty sections listed above + any lessons already hardcoded in §10e. Grow by one or two bullets per experiment.

### 8.4 Cheap-first rule (small-before-heavy)

For any given axis, order candidates strictly by effort ascending, breaking ties by expected Δ descending:

```
for axis in [retrieval, query-rewrite, rerank, response-prompt, response-model, fusion, SID]:
    candidates_on_axis.sort(key=(effort, −expected_Δ))
    ship XS/S candidates first; M/L/XL only after XS/S have been tried or explicitly ruled out
```

**Rationale** (measured in prior iteration):
- A single-line **persona prompt change** (C-1.1, XS effort) beat a **full LoRA fine-tune** (G-2.1-class, L/XL effort) on LLM-judge. Cheap wins both paid faster and set a stronger baseline for the expensive work.
- Every heavy task that lands on an uncalibrated baseline produces attribution noise. Cheap tasks de-risk the heavy ones.

**Operational check**: when `next_candidates.md` is re-ranked (§8.1 step 3), sort the top-5 by effort ascending within EV tiers. If the top row is M+ effort but an XS/S row exists within 0.6× its EV, promote the cheap one.

### 8.5 Initial queue (first ~20 candidates at plan write-time)

Already ranked via EV from `recent_papers_ideas.md §8 build order` + prior-iteration lessons, **reordered per §8.4 cheap-first rule**. Full list in `next_candidates.md` when created; top-5 here as the starter list:

1. **C-1.1** persona + word-ban prompt — XS effort, +0.02–0.04 LLM-judge (Gemini), P0 metric, cheap-first winner
2. **R-2.4** rule-based query preprocessor (lowercase/stem/stopword/synonym) — XS effort, +0.005–0.015 nDCG
3. **R-2.1** CMQR multi-query rewrite — S effort, +0.01–0.03 nDCG
4. **R-3.1** BGE-reranker-v2-m3 cross-encoder rerank — S effort, +0.005–0.015 nDCG
5. **R-1.1** BGE-M3 self-built index — M effort, +0.02–0.05 nDCG (promoted only after the XS/S entries above)

(Heavier candidates like R-4.1 LambdaMART and G-* SID track remain in the bank but gated by cheap-first.)

---

## 9. Wave-Based Execution Order

Waves are date-free because we're on a no-strict-deadline rhythm until Blind-B releases (2026-06-15). Shipping a first submission early and iterating beats optimizing a single "perfect" stack.

### 9.1 Wave 0 — Infrastructure (~1 day)

Tasks: W0-1 → W0-6 (sequenced as shown).
Exit criterion: `pytest` green, logs populated with baseline anchor, B-champ pointer set.

### 9.2 Wave 1 — Retrieval baseline sweep (~1 day)

Tasks: R-0.1–R-0.6 (parallelizable — each is an independent inference run).
Exit criterion: all 6 configs have rows in `submissions_log.md`; best one becomes B-champ-retrieval.

### 9.3 Wave 2 — Strong first stack (~2 days)

Tasks: R-1.1 (BGE-M3) + R-5.1 (RRF fusion BM25 + BGE-M3) + C-0 series + C-1.1 (persona).
Exit criterion: composite on dev ≥ 1.5 × B-floor; integration-tested end-to-end.

### 9.4 Wave 3 — First Blind-A ship (~0.5 day)

Tasks:
- L1 local eval on Wave 2's best stack — confirm `composite_retrieval` ≥ B-floor + reasonable margin
- Qualitative response review: skim 10 predicted_response samples for obvious failure modes (apologies, hallucinations, formulaic AI-speak) — §2.5.4 step
- E-3 (packager), E-4 (re-run champion on blind with train+dev), schema + submission-budget validate, upload via CodaBench
- On score return: archive all 80 (query, response, score) triples into `documents/blind_responses_scored.md`; record `(composite_retrieval_local, composite_blind)` pair in `agent_memory.md` Local→Blind calibration
- Mine worst-5 and best-5 scored rows; write initial Judge-behavior-notes bullets

Exit criterion: first Blind-A score in `submissions_log.md` AND first calibration data point logged AND initial Judge-behavior-notes populated.

### 9.5 Wave 4 — Parallel swell (~3–5 days, several tracks in flight)

Pick one from each axis, run async:

- **Retrieval**: R-3.1 (cross-enc rerank), or R-4.1 (LambdaMART, higher ceiling / more effort)
- **Query rewrite**: R-2.1 (CMQR)
- **Response**: C-1.3 (dynamic NN few-shot) or C-2.2 (CoVe)
- **Judge mining**: J-A1 + J-A2 (after Wave 3's blind score returns — extract Judge-behavior-notes from scored responses)
- **SID (async, Colab)**: G-0 → G-1.1 → G-2.1 — does not block shipping

Each completing task triggers the §8.1 loop.

### 9.6 Wave 5 — Second Blind-A ship (~1 day)

Tasks: E-1, E-2 (specialist registry + learned aggregator), re-ship with best stack from Wave 4.
Exit criterion: second Blind-A score strictly beats first (else revert).

### 9.7 Wave 6+ — Queue-driven iteration

Loop: pick top of `next_candidates.md` → run → log → regenerate queue → repeat. Continues until Blind-B closes.

### 9.8 Wave 7 — Blind-B preparation (triggered on 2026-06-15 Blind-B release)

Tasks: refresh Blind-B dataset, re-validate pipeline handles any schema tweaks, ship the top stack retrained on (train ∪ dev). Repeat iteration against Blind-B scores.

---

## 10. Appendix

### 10a. File / path reference

| Existing file | What to extend there |
|---|---|
| `music-crs-baselines/mcrs/crs_baseline.py:121,170` | Change top-1→LLM to top-N→LLM; propagate `top_n_for_prompt` from config |
| `music-crs-baselines/mcrs/retrieval_modules/__init__.py:17` | Register each new retriever in the factory |
| `music-crs-baselines/mcrs/lm_modules/__init__.py:3` | Register each new LM in the factory |
| `music-crs-baselines/mcrs/retrieval_modules/dense_precomputed.py:106-171` | Reuse the embedding imputation contract (artist→category→global mean) |
| `music-crs-baselines/mcrs/rerankers/__init__.py` (empty) | Add factory; register R-3 tasks here |
| `music-crs-baselines/mcrs/query_rewriters/__init__.py` (empty) | Add factory; register R-2 tasks here |
| `music-crs-baselines/mcrs/embedders/__init__.py` (empty) | Add factory if we produce our own embeddings (R-1.2 E5-mistral etc.) |
| `music-crs-baselines/mcrs/judges/` (NEW dir, DEFERRED) | Directory created at W0 but left empty. Populate only on J-D1 / J-D2 reactivation |
| `documents/blind_responses_scored.md` (NEW file) | Append-only archive of (query, predicted_response, blind_score) per scored blind submission. Source data for Judge-behavior-notes mining (§2.5.4, J-A1) |
| `music-crs-baselines/mcrs/aggregators/` (NEW dir) | Create for E-1/E-2 specialist combiner and learned stacker |
| `documents/agent_memory.md` (NEW file) | Cross-experiment learnings — validated mechanisms, dead ends, prediction calibration, Gemini rubric notes, Local→Blind calibration. See §8.3 + §2.5.4 |
| `scripts/local_eval.py` (NEW) | The retrieval-side composite eval harness (§2.5); produces `composite_retrieval` + `composite_projected`; called automatically by `scripts/run_experiment.py`. No judge calls |
| `scripts/run_experiment.py` (NEW) | Thin wrapper around inference + local_eval + log-write (§4.2) |
| `scripts/validate_prediction.py` (NEW) | Schema + submission-budget enforcement (§2.6, §7.1) |
| `music-crs-baselines/run_inference_devset.py`, `run_inference_blindset.py` | Invoked by `scripts/run_experiment.py`; no changes needed unless we add new top-level flags |
| `music-crs-evaluator/evaluate_devset.py` | Called from `run_experiment.py` after dev inference |
| `music-crs-evaluator/metrics/metrics_recsys.py:100` | `compute_recsys_metrics` — add new metrics (Personalization proxy etc.) via new file in `metrics/`, not here |
| `recsys2026-lora-tutorial/run_pipeline.sh` | Clone as `recsys2026/run_pipeline.sh` (adapted for RQ-VAE, Rank-GRPO stages) |
| `recsys2026-lora-tutorial/colab/Train_LoRA_Colab.ipynb` | Clone per GPU task (G-2.1 SFT, R-4.2 cross-enc fine-tune, etc.) |

### 10b. Paper → task mapping

| Paper (PDF in `documents/research/`) | Drives task(s) |
|---|---|
| CMQR | R-2.1 |
| Mistral-SPLADE | R-1.* (stretch — learned sparse variant) |
| TalkPlay | R-1.5 (audio-CLAP), R-2.* (lyrics as extra BM25 field) |
| ReFICR | R-1.* / C-0 (unified retrieval+gen fine-tune — not a first-wave target) |
| RA-Rec | R-2.5 |
| PEBOL | J-1 (NLI-as-signal complement to LLM judge) |
| ProRank | R-3.4 |
| FIRST | R-3.3 |
| RankZephyr | R-3.2 |
| Rank-R1 | R-3.5 |
| Text2Tracks | G-1.1, G-2.1 |
| GRID | G-0, G-1.*, G-2.* framework reuse |
| LIGER | G-2.3 |
| LETTER | G-1.3 |
| LC-Rec | G-2.* reference recipe |
| Joint-SIDs | G-1.* (bi-encoder fine-tune on search+rec before SID mint) |
| Bridging-Search-Rec | G-* design reference |
| IDGenRec | G-* alternative (textual IDs instead of codebook) |
| TIGER | G-0 foundational |
| Rank-GRPO | G-3.3 — the most on-point paper for our setting |
| Rec-R1 | R-4.3 (query-rewriter RL) |
| S-DPO | G-3.2 |
| KTO | G-3.1 |
| OPO / DRPO | G-3.* alternatives (differentiable nDCG) |
| Self-Rewarding LMs | C-2.1 — gated by J-D2/J-D3 reactivation (otherwise not usable without a local judge) |
| iLoRA | G-2.* MoE-LoRA if single adapter plateaus |
| MARec | R-1.* cold-start specialist (principled metadata alignment) |
| LLM-ESR | R-1.* long-tail enhancement |
| LLM-Prior-ColdStart | R-1.* Bayesian regularizer on any retriever |
| Judging-Judges | J-D4 (DEFERRED, position-bias diagnostics — only relevant if local judge reactivated) |

### 10c. ML techniques catalog (organized per module — feeds candidate generator)

**Retrieval**
- Sparse: BM25, BM25F multi-field, SPLADE, Mistral-SPLADE, Doc2Query-offline
- Dense: BGE-large / BGE-M3, E5-mistral-7B, NV-Embed-v2, Stella-1.5B, Qwen3-embedding
- Multi-modal: CLAP audio-text, SigLIP image-text
- Collaborative: cf-bpr, ALS, user×item affinity
- Graph: artist → collab → label-mate co-occurrence
- Hybrid: RRF, wRRF, learned stacker

**Query understanding**
- Rule-based: lowercase / stem / stopword / accent norm / WordNet expand
- LLM-based: CMQR multi-query, HyDE, step-back prompting, RA-Rec state tracking
- Trained: Rec-R1 query rewriter, goal-conditioned routing

**Reranking**
- Cross-encoder pointwise (BGE-reranker-v2-m3)
- Listwise LLM (RankZephyr, FIRST first-token)
- Training-based SLM: ProRank, Rank-R1 (GRPO)
- LambdaMART / LightGBM on engineered features

**Generation**
- Prompt: persona, word-ban, citation directive, CoT / scratchpad, step-back
- Few-shot: static generic, dynamic NN, semantic-cache
- Decoding: greedy, nucleus sampling (T-grid), contrastive decoding, speculative decoding
- Post-processing: word-subst, hallucination filter, CoVe, forced metadata grounding (`outlines`)
- Ensemble: multi-candidate + self-judge (gated by J-1), dual-model (Qwen + Llama)

**Alignment / training**
- SFT, LoRA-SFT, QLoRA, iLoRA MoE
- RL: KTO, DPO, S-DPO, Rank-GRPO, Rec-R1, OPO/DRPO
- Self-rewarding loops (gated by judge calibration)

**SID / end-to-end**
- RQ-VAE (content, CF, hybrid via LETTER), RQ-KMeans
- T5 or decoder-only autoregressive SID decoding
- Constrained beam search over catalog trie
- LIGER hybrid (gen + dense fallback)
- Textual IDs (IDGenRec)

**Ensemble**
- RRF / wRRF
- Learned stacker (logistic, GBM, small neural)
- Specialist router (goal-conditioned)

**Judging** (all local-judge options currently DEFERRED; online-only for now)
- Online CodaBench Gemini (active; the sole judge signal in this phase)
- Blind-response mining → qualitative pattern extraction into agent memory (J-A1, active)
- Gemini API (J-D1, deferred)
- Qwen-7B as local judge (J-D2, deferred; needs GPU)
- Pairwise with swap-average — position-bias mitigation (J-D4, deferred)
- NLI-as-signal — PEBOL (candidate if local CPU NLI model proves light enough)

### 10d. Data-signal coverage matrix

Each row = one unused/underused signal from `data_exploration.md`. Track which Track+task actually exploits it.

| Signal | Primary owner | Secondary | Status |
|---|---|---|---|
| `track_name`, `artist_name`, `album_name` | R-0.1 BM25 | all dense | used |
| `tag_list` (noisy) | R-0, R-1.1 (after clean) | — | partial; cleaning TBD |
| `release_date` | R-0 (filter) | — | partial |
| `popularity` | R-1.6 prior | R-4.1 LGBM | unused |
| `duration` | R-4.1 feature | — | unused |
| `lyrics` (via precomputed emb) | R-1.* (lyrics-qwen3) | — | unused |
| `audio-laion_clap` | R-1.5 | — | unused |
| `image-siglip2` | R-1.* (optional) | — | unused |
| `cf-bpr` (track) | R-1.4, G-1.1 | — | unused |
| `metadata-qwen3_embedding_0.6b` | R-0.6, R-5.1 | G-1.2 | partially used |
| `attributes-qwen3_embedding_0.6b` | R-1.* variant | — | unused |
| `cf-bpr` (user) | R-1.4 (warm users) | — | unused |
| User demographics (age, country, gender) | C-1.* prompt slot | R-4.1 feature | used (raw), not parsed |
| `preferred_musical_culture` | C-1.* | — | unused |
| `conversation_goal.category` (11 codes) | R-2.5, C-2.5 | R-4.1 feature | unused |
| `conversation_goal.specificity` (4 codes) | R-2.*, C-2.* | — | unused |
| `conversation_goal.listener_goal` (free text) | R-2.1 expansion seed | — | unused |
| Train conversations (gold music + response) | R-4.1, R-4.2, C-1.3, G-2.1 | — | unused |
| `goal_progress_assessments` (train, ~93% coverage) | R-4.1 graded labels | — | unused |

**Rule**: no experiment proposal is shipped unless it explicitly names the signal(s) it exploits from this table. Keeps us from re-running the same axis under new names.

### 10e. Dead ends — do NOT re-walk

Distilled from prior-iteration memory (`.claude/memory/project_blind_a_state.md`). These were paid-for; treat them as known-closed unless evidence changes.

| Dead end | Symptom | Prevention |
|---|---|---|
| MPS + float16 for Qwen batch gen | NaN in `torch.multinomial`, degenerate `!!!!!` output | Preflight assertion: fp32 on MPS; fp16 only on CUDA |
| Stock `response_generation.txt` prompt | "apologize on mismatch" → "I'm sorry" responses tanked LLM-judge | Never use stock; derive from C-1.1 |
| Stock `LLAMA_MODEL.batch_response_generation` chat template | Injects track as fake-assistant turn → "I'm glad you enjoyed X" hallucinations on turn 1 | Custom `[system, user]` only (C-0.3) |
| Qwen 7B + rigid few-shot | Overfits exemplar structure → homogeneous formulaic responses → LLM regression | Only use 7B with diverse-structure few-shot or zero-shot (C-3.1 gated) |
| Greedy query expansion without anti-repetition | `smooth jazz trap r&b × 4` wrecked BM25 retrieval | When using R-2.1 LLM rewrites: `no_repeat_ngram_size=3` + `repetition_penalty=1.2` |
| LoRA-on-train without v10 persona parity | Train-response style (chat-agent terse) opposite of judge-rewarded ("well-read critic") → −1.45 raw LLM | Any fine-tune must preserve the prompt-anchor; consider distillation from C-1.1 outputs instead of raw train |
| Any retrieval change that shifts submission ordering from BM25's natural ordering (without a decoupling fix) | Prior iteration found the Gemini judge penalizes non-BM25 ordering even when nDCG@20 improves | Decouple prompt-top-3 path from submission-20 path (BM25 top-3 → prompt; new mechanism → 20-list). Revalidate this on the fresh branch before trusting it |
| Word-replacement post-processing | Manual synonyms read worse to judge than Qwen's natural phrasing (v23: −0.25 LLM) | Don't regex-replace LLM output; fix at prompt level |
| Flat `submission.zip` with any filename other than `prediction.json` | CodaBench rejects | E-3 validator enforces singular filename at zip root |
| MPS deadlock with two HF models | v17 dead-locked for 85 min | `device="cpu"` for secondary model when Qwen is on MPS |

### 10f. Glossary

- **SID** — Semantic ID. Discrete code sequence assigned to each track (e.g. `(17, 203, 88, 42)` from a 4-level RQ-VAE with 256-way codebooks). LLMs predict SIDs instead of text.
- **RQ-VAE** — Residual-Quantized VAE. Compresses a continuous embedding to a hierarchy of discrete codes by quantizing successive residuals.
- **RRF** — Reciprocal Rank Fusion. Combines N ranked lists via `Σ 1 / (k + rank_i)`; `k=60` is standard.
- **wRRF** — Weighted RRF. Per-list weight applied before summation.
- **LambdaMART** — Gradient-boosted trees for ranking, optimising a listwise loss derived from NDCG gradients. LightGBM has it built-in.
- **S-DPO** — Softmax DPO. DPO generalised to 1-positive / N-negatives via Plackett-Luce.
- **KTO** — Kahneman-Tversky Optimisation. Prospect-theoretic alignment from binary desirable/undesirable signals (no paired preferences needed).
- **Rank-GRPO** — GRPO variant (Netflix 2026) where credit is assigned at rank-position granularity rather than token/sequence, with geometric-mean importance ratio for stability.
- **CoVe** — Chain-of-Verification. Draft → extract claims → verify each → rewrite only with verified claims.
- **HyDE** — Hypothetical Document Embeddings. LLM writes a hypothetical answer, encode with dense model, search.
- **CMQR** — Conversational Multi-Query Rewriting. One LLM call produces N rewrites; all retrieve; fuse.
- **ProRank** — 2025 paper; 2-stage GRPO warmup → last-token logit-diff scoring; 0.5B SLM beats 32B rerankers.
- **LIGER** — Hybrid of generative (SID) + dense retrieval; generation proposes, dense expands cold-start coverage.
- **LETTER** — RQ-VAE + contrastive CF alignment + diversity loss; SIDs reflect both text and co-listen signal.
- **MTEB** — Massive Text Embedding Benchmark. Public leaderboard; we track top performers (BGE-M3, E5-mistral, NV-Embed, Stella) as retrieval candidates.
- **Distinct-2** — Ratio of unique bigrams to total bigrams. Our `lexical_diversity` metric.

---

## End notes

This document is the system, not a summary of it. Every task below the top-level sections is meant to be picked up, executed, and logged. The framework is designed so that adding a new module never requires more than four mechanical edits (§3.3) and that no experiment ships without three hard gates (smoke, schema, judge).

Update protocol: this file is stable. When tasks complete, the _state_ moves to `submissions_log.md` / `benchmarks.md` / `next_candidates.md` — not here. Add a new task row to §5 when a new idea needs an ID; never reuse an ID.
