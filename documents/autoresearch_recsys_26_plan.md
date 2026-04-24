# Plan — Autonomous Research Agent System for RecSys 2026 Music CRS

**Version**: 2 · **Date**: 2026-04-17 · **Blueprint**: [`autoresearch_summary.md`](./autoresearch_summary.md)

This plan specifies an autonomous research agent system for the RecSys 2026 Music Conversational Recommendation challenge, adapted from karpathy/autoresearch. It is the canonical planning document; the execution order lives in §15 and the alignment audit vs. the blueprint lives in §16. Revision 2 folds in the F/L/P critical-review fixes and the A1–A8 autoresearch-alignment adjustments (see Changelog at the bottom).

---

## Context

The RecSys 2026 Music Conversational Recommendation challenge evaluates retrieve-then-generate systems on nDCG@{1,10,20} over a 1000-session dev set (8000 session×turn queries). The current public baseline (`LLaMA-1B + BM25`) scores **nDCG@10 = 0.0627** — there is enormous headroom. The user wants a **karpathy/autoresearch-style autonomous loop** adapted to this challenge: an agent system that researches → designs → implements → (trains →) evaluates on its own, driving nDCG@10 up iteratively.

**Hard constraint: Mac only, no GPU** (MPS + CPU). **Principle:** simple, robust, modular, branch-per-experiment, full logging, top-5-by-score weights kept. Training and evaluation strictly reuse the supplied repos; new work is additive, never destructive.

This plan does not implement the system yet — it specifies what will be built and how the agent loop operates once approved. Upon approval, step 1 is to copy this plan to `documents/autoresearch_recsys_26_plan.md`; step 2 is v0 scaffolding; step 3 is baseline reproduction.

---

## 1. Goal & headline metric

- **Primary**: `nDCG@10` on the dev set (`music-crs-evaluator/evaluate_devset.py` output).
- **Sanity anchors**: Random 0.0001 · Popularity 0.0024 · **LLaMA-1B + BM25 0.0627 (initial champion)**.
- **Win criterion — progress-based decaying threshold** *(A9)*: early iterations have large gains available (+0.010 to +0.030 expected in v2–v6), so accepting a +0.0015 trinket anchors the champion on shallow gains and blocks compound wins. Later, when the ceiling approaches, +0.0015 is legitimately the best you can get. The threshold therefore decays with closed headroom:
  ```
  progress  = max(0.0, (champion_ndcg10 - 0.0627) / (1.0 - 0.0627))
  threshold = max(0.002, 0.015 * (1.0 - progress))
  ```
  - Start (champion = 0.0627) → threshold = 0.0150. Demand meaningful wins.
  - Mid (champion = 0.30) → threshold = 0.0115.
  - Late (champion = 0.50) → threshold = 0.0065.
  - Near ceiling (champion ≥ 0.90) → threshold = 0.002 (noise floor).
  Self-correcting: stalling does not lower the bar; only an actual champion advance does.
- **Prerequisite pass** *(A9)*: if a backlog entry declares `unlocks: [vN, vM, ...]`, the threshold is relaxed to the 0.002 noise floor for that experiment only, *provided* its stated follow-ups are already in the backlog. This prevents the high-bar phase from rejecting enabler steps — e.g., v3 (dense retrieval, +0.005 standalone) unlocks v4 (RRF hybrid, +0.015), and without v3 being promoted, v4 cannot be built.
- **Guardrails** *(fixed, do not decay)*: `nDCG@1` and `nDCG@20` are logged every run. Regression beyond **max(10 % relative, 0.003 absolute)** on either one vetoes the win, even if the `ndcg@10` threshold is met. The guardrail floor is about measurement noise, not exploration strategy — so it stays constant. *(Fix L2.)*
- `predicted_response` affects only `lexical_diversity`, **not** nDCG → retrieval-only iteration is the default fast path.
- **Headline choice is a design assumption** (§14). If the challenge turns out to weight all three nDCGs equally, the win criterion is a one-line change.

## 2. Agent system architecture

Reuse the five agents already at `/Users/orrimoch/PythonProjs/recsys2026/.claude/agents/`. No agent redefinition; they are invoked with the appropriate task-specific prompt by the orchestrator.

| Agent | Role in loop | Writes to |
|---|---|---|
| `researcher` | WebSearch + WebFetch on papers / repos. Proposes backlog items with inline citations + optional `documents/research/<slug>.md`. | `documents/research/`, `experiments/backlog.md` |
| `data-explorer` | EDA on errors; bucketing by popularity/cold/genre; missing-data imputation strategy. | `documents/analysis/`, `experiments/runs/<tid>/eda_notes.md` |
| `experimenter` | Creates new YAML under `music-crs-baselines/config/`; adds new modules under `mcrs/retrieval_modules/`, `mcrs/rerankers/`, `mcrs/query_rewriters/`, `mcrs/embedders/`; wires factory; runs pipeline. | `config/`, `mcrs/`, `experiments/runs/<tid>/` |
| `code-reviewer` | Interface + submission-schema + Mac-compat check pre-run. | ledger `review` column |
| `prompt-engineer` | System prompt variants (HyDE, CoT query rewrite, response generation). | `mcrs/system_prompts/` |

**Two new binding documents** (not agents — playbook files read by the orchestrator = main Claude thread):

- **`experiments/program.md`** — static autonomous-loop playbook. Defines loop, allowed/forbidden actions, logging rules, `PAUSE` sentinel semantics, stopping rules. Human-authored.
- **`experiments/orchestrator.md`** — dynamic scratch file, rewritten each iteration with: current champion `tid`, last 5 ledger rows, top-3 backlog priorities, failure flags. Keeps context small.

### Orchestrator loop

```
while not stop_condition():
  # §17.5 daily cap + §17.6 watchdog checked between iterations
  if daily_wallclock_exceeded() or watchdog_kill_triggered(): break
  state = read ledger.tsv (last 20 rows), orchestrator.md
  idea  = pick top open backlog item sorted by (priority desc, expected_cost_min asc)
  # §17.8 anti-spin guard
  if idea.slug == last_three_claimed_slugs(): halt("anti_spin_same_slug_3x")
  if idea is None:
      dispatch researcher(topic = "next technique for <current_weakness>"); continue
  if idea.needs_eda:    dispatch data-explorer
  if idea.needs_theory: dispatch researcher
  tid   = next_monotonic_id() + "-" + slug        # see §3 for numbering
  git checkout main && git checkout -b exp/<NNN>-<slug>
  exp_start = now()                               # §17.3 per-experiment wallclock timer
  exp_tokens = 0                                  # §17.4 per-experiment token counter
  dispatch experimenter(spec=idea, tool_cap=80)   # §17.2 tool-call cap; writes config + modules
  dispatch code-reviewer(diff=git diff main..., tool_cap=30)   # must pass
  # after each dispatch: exp_tokens += result.usage; if exp_tokens > 200_000 or now()-exp_start > 3*expected_cost_min*60: abandon
  if idea.has_train_step:
      run_experiment.py --tid <tid> --phase train # §3 step "Train"
  if retrieval_only(config):
      run_experiment.py --tid <tid> --phase inference --subset 100   # smoke
      if smoke_ok:
          run_experiment.py --tid <tid> --phase inference --subset 1000
  else:
      run_experiment.py --tid <tid> --phase inference                  # full run, no subset
  run_experiment.py --tid <tid> --phase evaluate   # wraps cd + evaluator (see F1)
  if idea.deserves_bucket_analysis:
      run_experiment.py --tid <tid> --phase bucket_analysis
  git add music-crs-baselines/mcrs music-crs-baselines/config
  git commit -m "exp <NNN>: <hypothesis> | ndcg@10=<value>"
  log_to_trunk(tid, metrics)                     # see §9 for pattern
  evict_weights_top5()                           # deletes non-top-5 weights.pt
  threshold = compute_win_threshold(champion_ndcg10, idea.unlocks)   # §1, progress-based decay
  if delta_ndcg10 >= threshold and guardrails_ok and review_passed:
      git checkout main && git merge --no-ff exp/<NNN>-<slug>
      update champion in orchestrator.md
      dispatch researcher("variants on this win")
  # else: branch kept; weights.pt dropped by eviction
  emit one-line status for user; check PAUSE sentinel
```

**Stop conditions** (any one) — kept deliberately minimal to honor autoresearch's **NEVER STOP** discipline:
1. `PAUSE` sentinel file exists at repo root. *(Documented in `experiments/program.md`, P4.)*
2. Wallclock budget exceeded (configurable, default none).

**Never-stop hardening** *(A2)*: a run of rejected experiments does **not** trigger a pause. Instead, after 5 consecutive `kept=rejected`, the orchestrator dispatches `researcher` with `topic="think harder: 5 consecutive rejects on <current_weakness>. Work through the Escape-from-stuck tactics menu in experiments/program.md §12 in order."` and continues. Same when the backlog looks empty: re-seed via researcher, never halt. Only `PAUSE` or wallclock budget halts.

**No `sleep`.** The loop naturally yields between experiments via the one-line status + PAUSE check. *(Fix P4.)*

## 3. Experiment lifecycle

| Stage | Owner | Output | Path |
|---|---|---|---|
| Idea | researcher / human | backlog entry | `experiments/backlog.md` |
| Claim | orchestrator | branch + ledger row `status=running` | `git checkout -b exp/<NNN>-<slug>` from `main` |
| Config | experimenter | new YAML | `music-crs-baselines/config/<tid>.yaml` |
| Code | experimenter | new module(s) | `music-crs-baselines/mcrs/<new_file_or_dir>` + factory registration |
| Review | code-reviewer | pass/fail | ledger `review` column |
| **Train** *(if `has_train_step`)* | runner | weights + curves | `experiments/runs/<tid>/weights.pt`, `curves.png` |
| Run | runner | predictions | `music-crs-baselines/exp/inference/devset/<tid>.json` |
| **Link** | runner | visibility to evaluator | `cd music-crs-baselines && python ../music-crs-evaluator/evaluate_devset.py --tid <tid>` — evaluator's relative `exp/inference/devset/` resolves to baselines cwd. No copy, no symlink. *(Fix F1.)* |
| Eval | runner | scores | `music-crs-baselines/exp/scores/devset/<tid>.json` (created by evaluator relative to baselines cwd, because of the `cd` trick) |
| Bucket analysis *(optional)* | runner | per-bucket nDCG | `experiments/runs/<tid>/buckets.json` *(Fix L4.)* |
| Log | runner | ledger row + per-run dir | `experiments/runs/<tid>/`, `experiments/ledger.tsv` |
| Decide | orchestrator | promote/reject/abandon | updates `experiments/orchestrator.md`, merge if win |

### Branch and ID naming *(Fix F3, L5)*

- **`NNN`** = a monotonic, zero-padded 3-digit counter across *all* claimed experiments, independent of the `vN` roadmap label. `NNN` increments by 1 on every claim (win, reject, crash — all consume an ID). v1 → `001`; v2 → `002`; a researcher-inserted idea before v3 may take `003`, pushing v3 to `004`. The ledger column `tid` is `NNN-slug` (e.g. `001-baseline-repro`, `007-bm25-tags-corpus`).
- **Branch** = `exp/<NNN>-<slug>`; slug ≤ 40 chars, lowercase, hyphen-separated.
- **`parent_tid`** in the ledger = the tid of the **current champion at claim time** (the branch-point on trunk). After a promotion, subsequent experiments have a new parent_tid. *(Fix L6.)*

### Status vocabulary *(Fix L1)*

`status ∈ {running, done, crashed, no_score, abandoned}`. `kept ∈ {promoted, rejected, crashed, pending}` is a *derived* column:
- `status=done` + wins guardrails → `kept=promoted`
- `status=done` + fails guardrails → `kept=rejected`
- `status=crashed` → `kept=crashed`
- `status=running` (while in flight) → `kept=pending`

### Commit discipline

- One code commit per experiment on the experiment branch: `"exp <NNN>: <hypothesis> | ndcg@10=<value>"`.
- `main` receives only: scaffolding commits, promoted wins (`merge --no-ff`), ledger/backlog updates (see §9), per-run artifact commits (see §9).
- Branches are **never deleted**; force-pushes to `main` forbidden.

### Disk budget

At most 5 `weights.pt` files across all runs. `experiments/evict_weights.py` reads `ledger.tsv`, keeps only the top 5 by `ndcg@10` whose `weights_kept=yes`, deletes the rest from disk and toggles those ledger rows to `weights_kept=evicted`. *(P5 — this is not LRU; it is top-5-by-score.)* Other artifacts (~100 KB/run × 1000 runs = 100 MB) kept forever.

## 4. Scaffolding to create (v0 plumbing)

```
experiments/
  program.md                     # playbook (static)
  orchestrator.md                # dynamic scratch
  ledger.tsv                     # append-only (columns in §8)
  backlog.md                     # ordered ideas (seeded with v1–v5 + placeholders for v6+)
  run_experiment.py              # thin wrapper: dispatch train/inference/eval/bucket
  bucket_analysis.py             # per-bucket nDCG (pop × cold-user × cold-track)  *(L4)*
  evict_weights.py               # top-5-by-score eviction                          *(P5)*
  plot_curves.py                 # matplotlib utility (train loss + val ndcg)
  train_lgbm_reranker.py         # created when v6 is claimed
  train_twotower.py              # created when v10 is claimed
  runs/
    <tid>/
      config.yaml                # frozen copy of music-crs-baselines/config/<tid>.yaml
      code.diff                  # git diff main...HEAD at run time
      run.log.gz                 # stdout+stderr, gzipped
      metrics.json               # evaluator output + {wallclock_s, subset, seed, device, peak_rss_mb}
      curves.png                 # loss / val-ndcg curves (trained experiments only)
      eda_notes.md               # optional, if data-explorer ran
      buckets.json               # optional, from bucket_analysis.py
      weights.pt                 # only if currently in top-5 by ndcg@10
documents/
  research/                      # researcher agent output                          *(P8)*
  analysis/                      # data-explorer agent output
```

Also add (all new files — never edits to existing):

- `music-crs-baselines/run_inference_devset_retrieval_only.py` — retrieval-only entrypoint (§5).
- `music-crs-baselines/mcrs/rerankers/` — new package (for LightGBM, cross-encoder).
- `music-crs-baselines/mcrs/query_rewriters/` — new package (for HyDE / CoT).
- `music-crs-baselines/mcrs/embedders/` — loaders for precomputed track/user embedding matrices.
- `.gitignore` additions *(Fix F2)*:
  ```
  music-crs-baselines/exp/inference/
  music-crs-baselines/exp/scores/
  music-crs-baselines/cache/
  music-crs-evaluator/exp/inference/
  music-crs-evaluator/exp/scores/
  !music-crs-evaluator/exp/ground_truth/          # explicitly kept
  experiments/runs/*/weights.pt                    # blocked by default
  !experiments/runs/TOP5_WHITELIST/**/weights.pt  # top-5 tracked via whitelist file
  *.log
  !*.log.gz
  ```
- `experiments/cache/` — a **single shared** BM25 index cache dir pinned via config for retrieval-only iteration (v2–v5) to avoid 2–3 min re-index per run. *(Fix P3.)* Each config sets `cache_dir: ../../experiments/cache`.

### `experiments/program.md` contents (human-authored, summary)

The playbook file tells the orchestrator:

**0. The goal (one line)** *(A11, from autoresearch)*: **"Maximize nDCG@10 on the 1000-session dev set. Everything else is secondary."** Every iteration decision — what to try next, whether to keep a change, when to give up — answers to this one sentence.

1. The §2 loop pseudocode, verbatim.
2. The `PAUSE` sentinel semantics.
3. The forbidden list from §11.
4. The expected format of `run_experiment.py` CLI.
5. The NEVER-STOP rule (adapted): run until PAUSE or wallclock budget; rejects and empty-backlog trigger researcher re-seed, never pause. **Cold start of a new experiment never asks the human for permission** — the human may be asleep. *(A11, from autoresearch NEVER STOP.)*
6. **Simplicity criterion** *(A1, directly from autoresearch)*: "A tiny nDCG@10 gain that adds ugly complexity is not worth it. A simplification that holds nDCG@10 flat or better is a win — prefer it over any comparable-gain complexity. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 nDCG@10 improvement from deleting code is a great outcome. An improvement of ~0 but with much simpler code? Keep."
7. **In-scope file reads** *(A5)*: at the top of each experiment, before modifying anything, re-read: `CLAUDE.md`, `experiments/program.md`, `experiments/orchestrator.md`, the current `backlog.md` entry, the `music-crs-baselines/config/qwen1.5b_bm25_devset_mps.yaml` template, and — if the idea touches retrieval — `mcrs/retrieval_modules/{bm25,bert}.py`. Repo is small; re-reading keeps context fresh.
8. **RAM / disk growth policy** *(A7, from autoresearch "VRAM soft constraint")*: `peak_rss_mb` is a soft constraint. Some growth is acceptable for meaningful nDCG@10 gains, but it should not blow up dramatically. If peak_rss_mb exceeds 2× the current champion's memory AND the nDCG@10 gain is < 0.003, reject on complexity grounds (per simplicity criterion).
9. **Log discipline** *(A6, from autoresearch "never tee")*: the runner redirects everything to `run.log.gz` (via `>run.log 2>&1` then `gzip`). **Never `tee`, never stream subprocess output into Claude's context** — flooding the context window is the fastest way to lose track of the loop.
10. **One idea per experiment** *(A11)*: each experiment changes **exactly one** concept — e.g., only the corpus_types list, or only the retriever class, or only the rerank weight. Bundling multiple changes confounds attribution: you can't tell which change caused the delta, and can't compose future work on top. Exception: trivial fixes required to make the new idea run (missing import, typo) are folded into the same commit and noted.
11. **Reproducibility clause** *(A11)*: headline-comparison runs (the rows used to decide champion promotion) **must use the same subset size** (default: `--subset 1000`) and the same seed (default: 42). Subsampled smoke runs (`--subset 100`) are exploratory; they must never be used to set or depose a champion. When a new subset size or seed is introduced, a matching baseline re-run at that setting is required first.
12. **Escape-from-stuck tactics** *(A11, from autoresearch NEVER STOP)*: when the researcher is re-seeded after 5 consecutive rejects or an empty backlog, it works through this menu in order, picking the first that applies:
    1. **Re-read in-scope files** (program.md, CLAUDE.md, last 3 promoted wins' `code.diff`) for angles the ledger doesn't obviously show.
    2. **Read papers referenced in recent promoted wins** — their "related work" sections often list adjacent techniques.
    3. **Combine previous near-misses** — scan the ledger for rejected experiments with `delta_ndcg10 ∈ (0, threshold)`; two near-misses sometimes compose into a clear win.
    4. **Try a more radical architectural change** — switch category (retrieval → reranker → prompt → training), not just a hyperparameter.
    5. **Cross-bucket check** — look at `buckets.json` from the last data-explorer run; pick an idea targeting the worst bucket that hasn't been addressed.

## 5. Retrieval-only fast path (primary iteration mode)

Only `predicted_track_ids` moves nDCG. Add a sibling entrypoint `music-crs-baselines/run_inference_devset_retrieval_only.py` that:

1. Loads the retriever only (`load_retrieval_module(...)`), no LM, no `CRS_BASELINE`.
2. Reuses `chat_history_parser` logic from `run_inference_devset.py` to build retrieval queries (copied, not edited).
3. Calls `retriever.batch_text_to_item_retrieval(queries, topk=40)` — **requests 40, not 20**, to survive dedup *(Fix F6)*.
4. For each record: `unique = list(dict.fromkeys(candidates))[:20]`. If after dedup fewer than 20 remain, pad by requesting more from retriever or by popularity-based backfill. The goal: exactly 20 distinct track_ids per record, always.
5. Emits one record per (session, turn ∈ 1..8) with `"predicted_response": "ok"` (schema-valid stub).
6. Writes to `music-crs-baselines/exp/inference/devset/<tid>.json` (same path as the official runner).
7. Supports `--subset 100 | 1000` to truncate the test HF dataset after `load_dataset(...)`. No modification to the upstream `run_inference_devset.py`. *(Fix F5 — subset only applies here, not to LLM-path runs.)*

**Activation**: configs with `lm_type: null` or no `lm_type` field route via `run_experiment.py` to this entrypoint. Existing configs stay on the original entrypoint.

**Impact on Mac**: a full 1000-session BM25 run drops from ~45 min (LLM in loop) to ~3 min. Default iteration mode is full-set retrieval-only. Subsampling becomes a pure latency lever for trained components (v6, v10).

**Smoke vs full** *(Fix F5)*:
- Retrieval-only experiments: smoke `--subset 100` (~20 s) → full `--subset 1000` (~3 min).
- LLM-in-loop experiments (v1, v8, v9): **run once, full**. No smoke. No subset. Budget one slot per idea (~20–50 min each).

### Standard stdout summary block *(A3, grep-able — autoresearch parity)*

After every run, `run_experiment.py` prints a fixed, machine-readable block to stdout (also in `run.log.gz`) so the orchestrator can extract results with a single `grep`, mirroring autoresearch's `grep "^val_bpb:" run.log` pattern:

```
---
tid:              001-baseline-repro
status:           done
ndcg@1:           0.009800
ndcg@10:          0.062700
ndcg@20:          0.081500
cat_div:          0.198
lex_div:          0.680
subset:           1000
wallclock_s:      2784.0
peak_rss_mb:      1420.3
retrieval_type:   bm25
lm_type:          qwen2.5-1.5b
seed:             42
code_sha:         a1b2c3d
notes:            baseline reproduction
```

The orchestrator extracts the key metric with `grep "^ndcg@10:" run.log` (uncompressed before gzip). If the block is missing, the run is treated as crashed — same discipline as autoresearch.

### Hard timeout per experiment *(A4 — autoresearch parity on fixed budget)*

Every backlog item declares `expected_cost_min`. The runner enforces a **hard timeout = `2 × expected_cost_min`** (e.g., a 5-min expected run is killed at 10 min). On timeout: `SIGTERM` → 30 s grace → `SIGKILL`; ledger row written with `status=crashed, notes="timeout@<N>min"`; move on. This replaces autoresearch's fixed 5-min budget with a per-experiment budget, preserving the "bounded wall-clock" guarantee so the loop throughput stays predictable.

## 6. Bottom-up experiment roadmap

Each row: **vN — name — hypothesis — grounding — module touched — Mac wallclock.** (Expected gains are guesses; the ledger is the source of truth.)

**v0 — Plumbing (no science).** Build scaffolding listed in §4. No ledger row (counter starts at 001 for v1). ~1 h.

**v1 (`tid=001`) — Baseline reproduction.** Run `qwen1.5b_bm25_devset_mps.yaml` full pipeline, Mac MPS. Target: match `nDCG@10 = 0.0627 ± 0.001`. Locks champion. LLM-in-loop, no subset. ~50 min.

**v2a (`tid=002`) — BM25 field expansion.** Add `tag_list, release_date, popularity` to `corpus_types`. `Robertson & Zaragoza 2009 (BM25F)`. New YAML only. Retrieval-only. ~3 min. Expected +0.005–0.015.

**v2b (`tid=003`) — BM25 with imputation.** Same as v2a, plus imputing missing `tag_list`/`release_date` with artist-level median. data-explorer decides imputation rule. Retrieval-only. ~3 min. Expected +0.003 on top of v2a. *(Fix P1.)*

**v3 — Dense precomputed Qwen3-metadata retrieval.** New `DenseRetriever` loads `data/TalkPlayData-Challenge-Track-Embeddings` column `attributes-qwen3_embedding_0.6b`; query encoder = Qwen3 0.6B on CPU/MPS; cosine top-40. `Karpukhin 2020 (DPR)`, `Wang 2022 (E5)`. New: `mcrs/retrieval_modules/dense_precomputed.py`, registered. Retrieval-only. ~8 min. Expected ≈ baseline or +0.005.

**v4 — RRF hybrid BM25 + dense.** `HybridRRF` with k=60. `Cormack, Clarke, Büttcher 2009`. New: `mcrs/retrieval_modules/rrf.py`. Retrieval-only. ~5 min. Expected +0.005–0.015 vs best singleton.

**v5 — Multi-modal RRF.** Add `cf-bpr` (warm users only), `audio-laion_clap`, `lyrics-qwen3`. Convex-combo weights tuned on held-out 100-session slice. `Wang 2023 (multi-modal retrieval fusion)`. Extends `rrf.py`. Retrieval-only. ~10 min. Expected +0.005–0.015.

**v6 — LightGBM LambdaMART reranker.** Cross top-100 candidates with features: BM25 score, each dense score, popularity, release-date delta, CF-BPR dot, tag-overlap. Pairwise training on **train split** (15k sessions × ≤8 turns ≈ 120k (q, gold, neg-sample) tuples). `Burges 2010 (LambdaMART)`. New: `mcrs/rerankers/lgbm_rerank.py`, `experiments/train_lgbm_reranker.py`. `lightgbm` added to `pyproject.toml`. Training is the new lifecycle step (§3). Train ~5 min + eval ~3 min. Expected +0.015–0.030. *(Fix F4.)*

**v6.1** — focal loss on reranker if v6 underfits rare genres (data-explorer flags). *(from §7 table.)*

**v7 — Cross-encoder reranker.** `cross-encoder/ms-marco-MiniLM-L-6-v2` over top-50. Input: `(query, track_metadata_str)`. `Nogueira & Cho 2019 (MonoBERT)`. New: `mcrs/rerankers/crossencoder.py`. ~4 min on 100-session subset; ~40 min on full. Expected +0.015–0.025 over v4. **Note**: multilingual tracks → consider `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` if data-explorer confirms multilingual corpus. *(Fix P2.)*

**v8 — LLM pseudo-query expansion (HyDE / Query2Doc).** Qwen-0.6B generates 3 pseudo-queries from chat history; concatenate for BM25, or mean-embed for dense. `Gao 2022 (HyDE)`, `Wang 2023 (Query2Doc)`. Prompt: `mcrs/system_prompts/query_rewrite.txt` (new). Module: `mcrs/query_rewriters/hyde.py`. LLM-in-loop, no subset. ~20 min. Expected +0.010–0.020.

**v9 — Few-shot / CoT query rewrite.** Same as v8 with in-context examples mined from train. prompt-engineer iterates. Prompt file only. Same cost. Expected +0.003–0.010 on top of v8.

**v10 — Two-tower contrastive fine-tune (InfoNCE).** Student: `BAAI/bge-small-en-v1.5` (English) or `BAAI/bge-m3` / `intfloat/multilingual-e5-small` if multilingual *(Fix P2)*. Query tower: chat-history encoder; item tower: metadata encoder. InfoNCE + ANCE-style hard-negative mining (top-BM25 non-golds). Train on ≈120k triples, 2 epochs, batch 256 on MPS. `Oord 2018 (InfoNCE)`, `Xiong 2020 (ANCE)`. New: `experiments/train_twotower.py`, `mcrs/retrieval_modules/twotower.py`, `sentence-transformers` added to `pyproject.toml`. Training is a lifecycle step (§3). Fixed `seed` (§8). ~2–4 h train + ~3 min eval. Expected +0.020–0.050.

**v11 — Cold-start content-only fallback.** For cold users (26 %) / cold tracks (15.7 %): reweight RRF away from CF-BPR toward audio+metadata. `Schein 2002`. Extends `rrf.py` with `cold_aware_weighting`. Retrieval-only. ~3 min. Expected +0.005 on cold slice, mixed overall.

**v12 — Popularity-smoothing prior.** Add `α · log(1+popularity)` to final score; α tuned on 100-session held-out slice. `Koren 2009`. `rrf.py` param. Retrieval-only. ~3 min. Expected +0.003.

**v13+ — Exotic (deferred).** `DRAGON` zero-shot; generative retrieval (DSI, `Tay 2022`); Mix-of-Experts over retrievers gated by query classifier. High risk/cost — only after retrieval ceiling reached.

**Response-quality (lexical_diversity)** — deferred unless challenge weights diversity.

## 7. Imbalance / cold-start decision tree

`data-explorer` runs bucket analysis (§4 `bucket_analysis.py`) every ~5 experiments or on cold-start-suspicious regressions. Buckets = (popularity quintile × cold/warm user × cold/warm track). The dominant weak bucket decides next action:

| Failure mode | Deploy | Experiment |
|---|---|---|
| 26 % cold users, CF-BPR useless | drop CF weight for cold users | v11 |
| 15.7 % cold-only tracks never seen in train | exclude from negative pool; always content-indexable | v10, v11 |
| Long-tail popularity | popularity prior at rerank | v12 |
| Missing tags / release_date | artist-level median imputation | v2b |
| Underspecified queries ("play something") | HyDE pseudo-query expansion | v8 |
| Genre imbalance in train | large-batch InfoNCE (256–1024) implicit hard-neg coverage | v10 |
| Hard negatives drowned by easy ones | ANCE-style mining each epoch | v10 |
| Rare-class under-weighting at rerank | focal loss variant | v6.1 |

## 8. Logging spec (exact)

**Always kept, every run** (~100 KB):
- `experiments/runs/<tid>/config.yaml` — frozen copy of `music-crs-baselines/config/<tid>.yaml`
- `experiments/runs/<tid>/code.diff` — `git diff main...HEAD` at run time
- `experiments/runs/<tid>/run.log.gz` — gzipped stdout+stderr
- `experiments/runs/<tid>/metrics.json` — evaluator JSON + `{wallclock_s, subset, seed, device, peak_rss_mb}`
- `experiments/runs/<tid>/curves.png` — train-loss + val-ndcg curves (trained experiments only)
- `experiments/runs/<tid>/buckets.json` — if bucket analysis ran
- Row in `experiments/ledger.tsv`

**`metrics.json` unit note**: `peak_rss_mb` captured via `resource.getrusage(RUSAGE_SELF).ru_maxrss`. On macOS this returns **bytes** (divide by 1024²); on Linux it returns KB (divide by 1024). Document the conversion in code or the ledger number is wrong. *(Fix P9.)*

**Top-5 only** (by `ndcg@10`):
- `experiments/runs/<tid>/weights.pt`

**`ledger.tsv` columns** (TSV, one header row) *(adds `seed`, removes unused)* *(Fix P7)*:

```
tid  date  branch  parent_tid  status  kept  review  seed  ndcg@1  ndcg@10  ndcg@20  cat_div  lex_div  subset  wallclock_s  peak_rss_mb  retrieval_type  lm_type  corpus_types  code_sha  weights_kept  tokens_used_k  attempts  hit_cap  notes
```

Budget-related columns *(§17.11)*: `tokens_used_k` (cumulative k-tokens across agent dispatches for this experiment), `attempts` (claim count on this backlog slug so far), `hit_cap ∈ {none, run_timeout, cum_wallclock, tokens, oom, retries, watchdog}` (which circuit breaker, if any, closed this experiment).

Example:
```
001-baseline-repro  2026-04-17  exp/001-baseline-repro  -            done  promoted  pass  42  0.0098  0.0627  0.0815  0.198  0.68  1000  2784  1420  bm25  qwen2.5-1.5b  text  a1b2c3d  yes  "champion locked"
024-dense-qwen3m-rrf  2026-04-18  exp/024-dense-qwen3m-rrf  001-baseline-repro  done  promoted  pass  42  0.0124  0.0781  0.0993  0.231  0.52  1000  183  2105  dense_qwen3_rrf  none  metadata  ab3cd12  yes  "RRF k=60; +0.015"
```

**`backlog.md` row schema**:

```markdown
### [P1] v4 · RRF hybrid BM25 + dense
- hypothesis: RRF fuses ranked lists robustly without weight tuning.
- grounding: Cormack et al. 2009 (RRF).
- parent_champion: whichever is champion at claim time (runner reads at claim)
- touches: mcrs/retrieval_modules/rrf.py (new), mcrs/retrieval_modules/__init__.py
- has_train_step: false
- retrieval_only: true
- expected_cost_min: 10
- expected_gain_ndcg10: +0.010
- risk: low
- unlocks: [v5, v11]              # optional; if set, triggers the §1 prerequisite-pass relaxation
- research_notes: documents/research/rrf.md   # optional
- status: open
```

Source of truth for ranking = `ledger.tsv`. No sidecar state.

## 9. Git discipline (cleaner ledger pattern)

- `main` = trunk. Always green (baseline reproducible).
- `exp/<NNN>-<slug>` = experiment branch, from `main` at claim.
- **Promotion**: `git checkout main && git merge --no-ff exp/<NNN>-<slug>` when `Δ nDCG@10 ≥ threshold` (progress-based, §1) AND guardrails pass AND code-reviewer passed.
- **Per-run artifacts + ledger row live on `main`**, not on the experiment branch *(Fix F7)*. Flow:
  1. On experiment branch: commit code changes only (`git commit -m "exp <NNN>: ..."`). No `experiments/runs/` or ledger edits.
  2. Runner finishes inference/eval. All artifacts written to `experiments/runs/<tid>/` in the working tree.
  3. Runner `git checkout main` (the untracked `experiments/runs/<tid>/` files are preserved across checkout — they're not tracked yet).
  4. On `main`: `git add experiments/runs/<tid>/ experiments/ledger.tsv experiments/backlog.md` and commit: `"log <NNN>: ndcg@10=<value>"`.
  5. `git checkout exp/<NNN>-<slug>` — return to the experiment branch for decision + optional promotion.
  
  This avoids `git stash --include-untracked` fragility and conflicts on `ledger.tsv`.
- Never delete experiment branches. Never force-push `main`.

### `.gitignore` authoritative list *(Fix F2)*

```
music-crs-baselines/exp/inference/
music-crs-baselines/exp/scores/
music-crs-baselines/cache/
music-crs-evaluator/exp/inference/
music-crs-evaluator/exp/scores/
experiments/cache/
experiments/runs/*/weights.pt
*.log
!*.log.gz
# Kept as explicit exceptions
!music-crs-evaluator/exp/ground_truth/
```

Top-5 weights are not tracked via git whitelist (too fragile). Instead: all `weights.pt` ignored; disk is the source of truth, `weights_kept` column in ledger declares which runs should have weights on disk. `evict_weights.py` reconciles disk ↔ ledger.

## 10. Crash / retry / stop rules

- **Crash** (exception / OOM / NaN): retry once with `max(batch_size // 2, 1)`. If second attempt crashes (including at bs=1), `status=crashed, kept=crashed`, ledger notes `"crashed@bs=1"`, move on. *(Fix L3.)*
- **Regression** (valid nDCG@10 below parent or guardrail fails): `status=done, kept=rejected`. Never retry. Signal is the point.
- **Silent eval failure** (evaluator writes no JSON or NaN score): `status=no_score`, flagged for human.
- **Per-idea retries**: max 2. Then `status=abandoned, kept=rejected` with the attempted run IDs in notes.
- **`PAUSE` sentinel** checked between experiments at the top of the loop. Any non-empty `PAUSE` file halts cleanly after the current run finishes.
- **All hard budget caps** (wallclock, tokens, tool calls, watchdog, anti-spin) are specified in **§17 — Runaway protection & circuit breakers**. Any budget violation from §17 results in `status=abandoned|crashed`, ledger row written, loop continues (except the loop-wallclock cap and watchdog hard stop, which halt the loop).

## 11. What this plan does **NOT** include (safety / CLAUDE.md compliance)

- No edits to any existing `music-crs-baselines/config/*.yaml`.
- No edits to `music-crs-evaluator/` (evaluator, `make_ground_truth.py`, `metrics/`, `exp/ground_truth/`).
- **Design choice** *(Fix P6)*: no edits to `music-crs-baselines/run_inference_devset.py`. A sibling `run_inference_devset_retrieval_only.py` is added instead. This is self-imposed discipline, not a hard CLAUDE.md rule — it keeps the official baseline reproducible and its git history clean.
- No experiment sets `track_split_types` to anything other than `["all_tracks"]`.
- No committing of datasets, caches, `.pt` weights, or inference/score JSONs. (`.gitignore` §9 enforces.)
- No Colab; Mac CPU/MPS only. (User directive supersedes CLAUDE.md's Colab preference.)
- No new deps except `lightgbm` at v6 and `sentence-transformers` at v10. No `faiss`, no MLX, no `rank_bm25`.
- No perturbation of challenge splits (`train / test (=devset) / Blind-A / Blind-B` exactly as defined by HF `talkpl-ai/` datasets).

## 12. Critical files

**Writable by the loop** (new files or factory registrations only):

- `/Users/orrimoch/PythonProjs/recsys2026/music-crs-baselines/mcrs/retrieval_modules/__init__.py` — add new retriever keys.
- `/Users/orrimoch/PythonProjs/recsys2026/music-crs-baselines/mcrs/lm_modules/__init__.py` — only if a new LM loader is added.
- `/Users/orrimoch/PythonProjs/recsys2026/pyproject.toml` — `lightgbm` at v6, `sentence-transformers` at v10.
- `/Users/orrimoch/PythonProjs/recsys2026/.gitignore` — list in §9.
- All new files under `mcrs/rerankers/`, `mcrs/query_rewriters/`, `mcrs/embedders/`, `config/`, `experiments/`, `documents/research/`, `documents/analysis/`.

**Read-only references** (reused, never edited):

- `/Users/orrimoch/PythonProjs/recsys2026/music-crs-baselines/run_inference_devset.py`
- `/Users/orrimoch/PythonProjs/recsys2026/music-crs-baselines/mcrs/crs_baseline.py`
- `/Users/orrimoch/PythonProjs/recsys2026/music-crs-baselines/mcrs/retrieval_modules/{bm25,bert}.py`
- `/Users/orrimoch/PythonProjs/recsys2026/music-crs-baselines/config/qwen1.5b_bm25_devset_mps.yaml` (template to copy from)
- `/Users/orrimoch/PythonProjs/recsys2026/music-crs-evaluator/evaluate_devset.py`
- `/Users/orrimoch/PythonProjs/recsys2026/music-crs-evaluator/metrics/metrics_recsys.py`
- `/Users/orrimoch/PythonProjs/recsys2026/music-crs-evaluator/exp/ground_truth/devset.json`
- `/Users/orrimoch/PythonProjs/recsys2026/.claude/agents/{researcher,experimenter,code-reviewer,data-explorer,prompt-engineer}.md`

## 13. Verification (end-to-end test plan)

After v0 scaffolding + v1 baseline reproduction:

1. `wc -l experiments/ledger.tsv` = 2 (header + `001-baseline-repro`). `ndcg@10 ≈ 0.0627 ± 0.001`. *(Fix L5 — tid is `001`, not `002`.)*
2. `ls experiments/runs/001-baseline-repro/` contains `config.yaml`, `code.diff`, `run.log.gz`, `metrics.json`, `curves.png?`, `weights.pt?`.
3. `git branch` shows `main` and `exp/001-baseline-repro`.
4. Predictions record count: `jq 'length' music-crs-baselines/exp/inference/devset/001-baseline-repro.json` = **8000**. *(Fix P10.)*
5. Dedup check: `jq '[.[] | select(.predicted_track_ids | length != 20 or (.predicted_track_ids | unique | length) != 20)] | length'` = **0** (no under-20 and no duplicates).
6. Determinism check: re-run `cd music-crs-baselines && python ../music-crs-evaluator/evaluate_devset.py --tid 001-baseline-repro`. `exp/scores/devset/001-baseline-repro.json` byte-identical to first run.
7. PAUSE test: `touch PAUSE && <run orchestrator>` — exits without starting a new experiment, logs `"paused at boundary"`.
8. Evict test: add 6 fake weight files and ledger rows, run `evict_weights.py`; only top-5 by `ndcg@10` remain on disk; rows flipped to `weights_kept=evicted`.
9. Schema test: fabricate a prediction with a duplicate `track_id`; code-reviewer catches it before the evaluator would reject.
10. Bucket analysis: `python experiments/bucket_analysis.py --tid 001-baseline-repro` produces a `buckets.json` with 5 × 2 × 2 = 20 cells, each reporting `{n_queries, ndcg@10}`.

## 14. Open assumptions (flag for redirection)

- **Headline metric** = `nDCG@10`; @1/@20 are guardrails. If the challenge weights all three equally or averages, §1 is a one-line flip.
- **Subsampling scope** = retrieval-only mode only. LLM-path experiments run full dev set once. If this proves too slow on Mac, add an LLM-path subset wrapper later.
- **Serial execution**: one experiment at a time on the Mac (no git worktrees). If parallelism is wanted (e.g., BM25-only + LightGBM train concurrently), switch to worktrees — minimal scaffolding change.
- **PAUSE sentinel** is one of the two hard stops; the other is the daily wallclock cap (§17.5, default 12 h). All other caps (§17.1–§17.9) abort the current work unit only and let the loop continue.
- **`parent_tid` semantics** = current champion at claim time (not branch-off commit). Simplifies lineage tracking.
- **Budget defaults** in §17.10 are first-pass heuristics. Re-tune after ~20 experiments once real token/wallclock distributions are visible in ledger `tokens_used_k` and `wallclock_s`.

## 15. Execution order post-approval

1. Copy this plan to `documents/autoresearch_recsys_26_plan.md`. Commit on `main`: `"docs: autoresearch recsys26 plan v2"`.
2. Create `documents/research/` and `documents/analysis/` (empty). Commit.
3. v0 scaffolding on `main`:
   - `experiments/{program.md, orchestrator.md, ledger.tsv (header only), backlog.md (v1–v5 seeded, v6+ placeholders), run_experiment.py, bucket_analysis.py, evict_weights.py, plot_curves.py, cache/}`
   - `music-crs-baselines/run_inference_devset_retrieval_only.py`
   - `music-crs-baselines/mcrs/{rerankers,query_rewriters,embedders}/__init__.py` (empty packages)
   - Updated `.gitignore`
   - Commit: `"scaffold: autonomous research system"`.
4. Run v0 verification subset (steps 1,3,5,7,8 from §13 skipped since no experiment has run yet; check scaffolding exists and runner dispatches `--help` correctly).
5. Hand off to orchestrator loop. First iteration = v1 (`tid=001-baseline-repro`).

---

## 16. Alignment with the autoresearch blueprint

Mapping each of autoresearch's eight core principles (from `documents/autoresearch_summary.md`) to this plan. Principles are either **Match**, **Justified divergence** (intentional adaptation to our pipeline / Mac constraint), or **Adopted via adjustment** (v2 fix closing a prior gap).

| # | autoresearch principle | This plan | Why |
|---|---|---|---|
| 1 | **Single editable file (`train.py`)** | **Justified divergence** — editable surface = new files under `mcrs/{retrieval_modules,rerankers,query_rewriters,embedders}/` and new YAMLs under `config/`. | Our task is a multi-stage pipeline (retriever → reranker → LLM), not a single training script. Compensating discipline: additions only, no edits to existing modules; factory registration is the only contact with pre-existing code. |
| 2 | **Fixed 5-minute time budget** | **Adopted via adjustment** — per-experiment hard timeout = `2 × expected_cost_min` *(§5, A4)*. | Heterogeneous pipeline (3-min retrieval vs. 4-h training) makes a single fixed budget impractical. Per-idea budgets preserve the "bounded wall-clock" guarantee that makes throughput predictable. |
| 3 | **One scalar metric (`val_bpb`)** | **Match with guardrails** — nDCG@10 is the decision scalar; @1/@20 only kill obvious regressions. | autoresearch accepts occasional noise in a single scalar because experiments are cheap (12/hour). Our experiments are 10–100× more expensive; guardrails prevent the loop from committing to a phantom win. |
| 4 | **Frozen evaluation harness** | **Match** — `music-crs-evaluator/` is read-only; `track_split_types=["all_tracks"]` hard rule. | Same reasoning as autoresearch: prevents Goodhart. |
| 5 | **Git as keep/discard** | **Match** — `exp/<NNN>-<slug>` branch per experiment, merge `--no-ff` on win. | Straight adoption. One small enhancement: we never `git reset` rejected branches away — the failed branch is kept for the record. |
| 6 | **Flat results.tsv, untracked** | **Justified divergence** — ledger tracked on `main`. | Our project has durability requirements (losing 1000 experiments' metadata is catastrophic; autoresearch can re-run in an hour). Ledger writes happen only on `main` via the clean pattern in §9, so cross-branch conflicts are avoided. |
| 7 | **Simplicity criterion** | **Adopted via adjustment** — explicit clause now lives in `experiments/program.md` *(A1, §4 content block)*. | Previously implied; now verbatim from autoresearch's playbook. |
| 8 | **NEVER STOP** | **Adopted via adjustment** — rejects and empty-backlog re-seed the researcher, never halt the loop. Only `PAUSE` sentinel or wallclock budget stops *(A2, §2)*. | Previously had a soft-pause on 5 rejects, which violated the spirit. Closed. |

### autoresearch's 9-step loop mapped to ours

| autoresearch step | This plan |
|---|---|
| 1. Inspect current branch/commit | `git checkout main && git checkout -b exp/<NNN>-<slug>` (§2 loop) |
| 2. Modify train.py with one idea | `experimenter` adds new module(s) + new config YAML (§3 lifecycle) |
| 3. `git commit` | One code commit per experiment (§3 "Commit discipline") |
| 4. `uv run train.py > run.log 2>&1` (never tee) | `run_experiment.py` redirects everything to `run.log` → `run.log.gz`; no `tee`, no stream-to-context (§4 "Log discipline", A6) |
| 5. `grep "^val_bpb:" run.log` | `grep "^ndcg@10:" run.log` on the standard summary block (§5 "Standard stdout summary block", A3) |
| 6. If empty → crashed | `status=crashed` + `bs/2` retry with floor 1 (§10) |
| 7. Append row to results.tsv | Append to `experiments/ledger.tsv` on `main` (§9) |
| 8. If val_bpb improved → advance | If `Δ nDCG@10 ≥ threshold(champion)` (progress-decaying, §1) AND guardrails pass → `git merge --no-ff` (§2, §9) |
| 9. Else → git reset | Branch kept, weights evicted to maintain top-5 (§3, §9 — preserves the record) |

### Hardware-budget comparison

| | autoresearch | this plan |
|---|---|---|
| Hardware | 1 × H100 (~80 GB VRAM) | MacBook CPU/MPS (~16 GB RAM budget) |
| Model scale | 50 M params GPT, trained from scratch | Pretrained OSS retrievers (up to BGE-small 33 M) and rerankers (MiniLM-L6 22 M); Qwen-1.5B for generation only |
| Experiments/hour | ~12 | ~1–15, mode-dependent (retrieval-only: ~15/h; LLM-path: ~1–3/h; trained: <1/h) |
| Overnight yield | ~100 experiments | ~10–50 experiments |

This is the central reason we diverge on budgets and editable surface: we're exchanging throughput for pipeline realism.

## 17. Runaway protection & circuit breakers *(A10)*

The autonomous loop must never burn unbounded wall-clock or tokens on a single bad idea, a hung agent, or a pathological retry pattern. **Multiple independent layers of hard stops** are defined below. Any one firing aborts the current work unit cleanly; the loop continues unless the cap was a loop-level cap.

### 17.1 Per-run process timeout *(already specified in §5, A4)*

- Hard kill at `2 × expected_cost_min`. Sequence: `SIGTERM` → 30 s grace → `SIGKILL`.
- Covers the full sum of attempts (original + OOM retry), not per-attempt.
- Ledger row: `status=crashed, notes="timeout@<N>min"`. Loop continues.

### 17.2 Per-agent-dispatch tool-call cap

Each subagent dispatch receives a hard tool-call budget in its prompt. If the agent approaches the cap without a concrete result, it is instructed to return early with `"insufficient: <reason>"` rather than keep flailing:

| Agent | Max tool calls | Rationale |
|---|---|---|
| `researcher` | 40 | ≈ 20 WebSearch/WebFetch + 20 Read/Write |
| `experimenter` | 80 | File edits + Bash dominate |
| `code-reviewer` | 30 | Read-only; just reviews one diff |
| `data-explorer` | 50 | Dataset scans + Bash |
| `prompt-engineer` | 20 | Prompt edits + a few reads |

These are self-discipline caps baked into the dispatch prompt. An agent that returns `"insufficient"` does not veto the experiment; the orchestrator proceeds with whatever partial result it has (e.g., ship the code change without prompt polish, or dispatch another agent if critical path).

### 17.3 Per-experiment cumulative wallclock cap

From branch creation (`git checkout -b exp/<NNN>-<slug>`) to ledger-row commit: hard cap at `3 × expected_cost_min` total. This covers the sum of every agent dispatch + every run + every retry.

On breach: the current in-flight subprocess is killed (SIGTERM→SIGKILL), ledger row `status=abandoned, notes="cum_wallclock_exceeded(<M>min)"`, loop continues.

### 17.4 Per-experiment cumulative token budget

Total Claude tokens consumed across all agent dispatches for one experiment (from claim to promotion decision): **hard cap 200 K tokens**.

Implementation: the orchestrator tracks cumulative tokens from each `Agent` tool result's usage metadata (when surfaced) and maintains its own rough estimate otherwise. If the running sum crosses 200 K before the experiment commits, the orchestrator forcibly closes the experiment: `status=abandoned, notes="token_budget_exceeded(<N>k)"`, loop continues.

### 17.5 Daily loop wallclock cap

Configurable at launch; default **12 h**. When the elapsed time exceeds the cap, the loop finishes the current experiment (which is already bounded by §17.1 and §17.3) and halts. A `PAUSE` sentinel is auto-written so a naive resume is a no-op until the user removes it.

### 17.6 Watchdog — no-progress timeout

A lightweight watchdog monitors `experiments/ledger.tsv` mtime:

- **Level 1 (warning)**: no new ledger row in `2 × max(expected_cost_min)` minutes → emit a one-line warning and dump the in-flight subprocess tree to `experiments/watchdog_stuck.log`.
- **Level 2 (hard kill)**: still no new row after another `1 × max(expected_cost_min)` minutes → kill the entire subprocess tree, commit a partial ledger row with `status=crashed, notes="watchdog_kill"`, and **halt the loop** (requires user to investigate and restart).

### 17.7 Retry caps *(reinforced)*

- **Per-idea**: max 2 claims. After the second `status=crashed` or `status=abandoned` on the same backlog slug, the backlog row is marked `status=abandoned, kept=rejected` with `tried_tids` in notes and is no longer picked.
- **Per-run OOM**: 1 retry with `bs = max(bs // 2, 1)`. Second crash → `status=crashed`.
- **Per-code-fix during agent dispatch**: `experimenter` may apply one trivial fix (typo / missing import) and re-run. A second code fix required → abandon the idea.

### 17.8 Loop-level anti-spin heuristic

If the orchestrator observes itself claiming **the same backlog slug 3 times in a row** (irrespective of `tid` suffix), that signals a bug — most likely the backlog item wasn't marked `status=abandoned` after previous failed attempts. Response: halt the loop with an explicit error message and require user intervention. Prevents silent infinite re-claiming.

### 17.9 Agent-internal tool-loop detection *(self-discipline)*

Agents are instructed (via `experiments/program.md` and each agent file) to bail out early if they notice:

- The **same tool call** (same tool + same key args) issued 3 times without progress → stop, return `"loop_detected"`.
- **Same file read** 5 times in one session → stop, return with the current state.

The orchestrator cannot enforce this from outside; it relies on agent self-discipline. Observed violations surface in `run.log` and are used to refine prompts over time.

### 17.10 Default budget summary (for `experiments/program.md`)

```
per_run_timeout_s         = 2 * expected_cost_min * 60          # §17.1, hard kill
per_experiment_wallclock  = 3 * expected_cost_min * 60          # §17.3, hard kill + abandon
per_experiment_tokens     = 200_000                              # §17.4, abandon on breach
per_agent_tool_calls      = {researcher: 40, experimenter: 80,   # §17.2, self-discipline
                             code-reviewer: 30, data-explorer: 50,
                             prompt-engineer: 20}
loop_wallclock_cap_h      = 12                                   # §17.5, halt loop
watchdog_warn_min         = 2 * max(expected_cost_min)           # §17.6, warn
watchdog_kill_min         = 3 * max(expected_cost_min)           # §17.6, halt loop
retries_per_idea          = 2                                    # §17.7
retries_per_run_oom       = 1  (floor bs=1)                      # §17.7
anti_spin_same_slug       = 3  → halt loop                       # §17.8
agent_repeat_call_limit   = 3  → agent self-bails                # §17.9
agent_repeat_read_limit   = 5  → agent self-bails                # §17.9
```

### 17.11 Budget recording

Every ledger row gets these derived columns (added to the §8 schema at scaffolding):

- `tokens_used_k` — cumulative tokens across all agent dispatches for this experiment (in thousands).
- `attempts` — how many claim attempts preceded success/abandonment for this backlog slug.
- `hit_cap` — one of `{none, run_timeout, cum_wallclock, tokens, oom, retries, watchdog}`; blank if the experiment finished naturally.

This makes budget-pressure patterns visible in the ledger so the human can later tune budgets post-hoc.

### Intentionally **not** adopted

- autoresearch's `analysis.ipynb` post-hoc plotting notebook. Replaced by `experiments/plot_curves.py` + bucketed analysis (`bucket_analysis.py`), driven by the orchestrator rather than a human.
- autoresearch's "give up after a few attempts" on crashes. We formalize to `max 2 retries with bs/2`, then status=crashed.
- autoresearch's `uv` project manager. We stay with the existing `pyproject.toml` flow to keep drift minimal.

---

### Changelog from v1 of this plan

- **F1** Evaluator `cd` pattern: `cd music-crs-baselines && python ../music-crs-evaluator/evaluate_devset.py` — no copy, no symlink.
- **F2** `.gitignore` paths corrected (evaluator scores under `music-crs-evaluator/`, not `music-crs-baselines/`).
- **F3 / L5** ID numbering clarified; v1 = `001`, not `002`.
- **F4** Training lifecycle step added for v6/v10.
- **F5** Smoke/full split bound to retrieval-only mode; LLM path runs once full.
- **F6** Retriever requests `topk=40`, dedup + trim to 20, backfill if short.
- **F7** Per-run artifacts live on `main`, committed separately after the experiment's code commit. No more `git stash` dance.
- **L1** Unified status vocabulary; `kept` is derived.
- **L2** Guardrail floor: max(10 % relative, 0.003 absolute).
- **L3** OOM retry floor at bs=1.
- **L4** `experiments/bucket_analysis.py` added.
- **L6** `parent_tid = current champion at claim time`.
- **P1** v2 split into v2a (field expansion) + v2b (imputation).
- **P2** Multilingual encoder note for v7/v10.
- **P3** Shared `experiments/cache/` dir pinned for retrieval-only iteration.
- **P4** Replaced `sleep 60` with one-line status + PAUSE check.
- **P5** "LRU" → "top-5-by-score" eviction; `weights_kept ∈ {yes, no, evicted}`.
- **P6** Stance on `run_inference_devset.py` clarified as design choice, not CLAUDE.md rule.
- **P7** `seed` column added to ledger.
- **P8** `documents/research/` convention for backlog citations.
- **P9** macOS `ru_maxrss` unit note.
- **P10** 8000-record and dedup-correctness checks in verification plan.

### Alignment adjustments from autoresearch audit (this revision)

- **A1** Simplicity criterion added verbatim to `experiments/program.md` spec.
- **A2** NEVER STOP hardened: rejects and empty-backlog re-seed researcher, no pause.
- **A3** Standard grep-able stdout summary block printed by the runner (mirrors `grep "^val_bpb:" run.log`).
- **A4** Hard per-experiment timeout = `2 × expected_cost_min` (preserves bounded wall-clock).
- **A5** In-scope file re-reads at the top of each experiment (program.md spec).
- **A6** "Never `tee`, never stream to context" discipline in program.md.
- **A7** RAM growth policy: soft constraint, rejected on complexity grounds if `peak_rss_mb > 2×` champion AND gain < 0.003.
- **A8** New §16 documenting match / adjustment / justified divergence vs. autoresearch's eight principles and 9-step loop.
- **A11** Lifted four general policy statements from autoresearch `program.md` into the `experiments/program.md` spec (§4): (0) one-line goal statement, (10) "one idea per experiment" rule for clean attribution, (11) reproducibility clause forbidding subset-size / seed drift on headline comparisons, (12) explicit "escape-from-stuck" tactics menu for the researcher re-seed path — re-read in-scope files → paper related-work → near-miss combinations → radical category shift → cross-bucket targeting. Plus reinforced "cold start never asks permission" under rule 5.
- **A10** New **§17 Runaway protection & circuit breakers** — six independent hard-stop layers: per-run process timeout (§17.1), per-agent-dispatch tool-call cap (§17.2), per-experiment cumulative wallclock (§17.3), per-experiment cumulative token budget (§17.4), daily loop wallclock cap (§17.5), watchdog no-progress timeout (§17.6). Plus reinforced retry caps, an anti-spin heuristic, and agent-internal tool-loop self-bail rules. Ledger now records `tokens_used_k`, `attempts`, `hit_cap` so budget-pressure patterns are visible. Default budgets listed in §17.10.
- **A9** Win threshold replaced with a **progress-based decaying function** — `threshold = max(0.002, 0.015 × (1 − progress))` where `progress = (champion − 0.0627) / (1 − 0.0627)`. Starts at 0.015 (demand meaningful wins early when +0.01–+0.03 gains are available), decays to the 0.002 noise floor near ceiling. Plus a **prerequisite pass**: backlog items declaring `unlocks: [vN, ...]` get the noise-floor threshold for that run only, so enabler steps (e.g., v3 enabling v4) don't get rejected during the high-bar phase. Guardrails (@1/@20 regression) stay fixed — they're about measurement noise, not strategy.
