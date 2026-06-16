# P0 — EDA & Recall-Ceiling Probe

> Phase-0 analysis deliverable. Notebook `nb/phase0_eda.ipynb`, config `config/eda.yaml`.
> Answers plan §5's 8 questions from real data and **sizes every Phase-1+ design parameter**
> (fusion-K, `topk_internal`, context cap, cold/warm threshold, channel keep-list).
> Analysis-only — **no training**, no model artifacts. See `000_INDEX.md` for the catalogue and build graph.

## 1. Purpose
Know the data cold and measure — not assume — the numbers that fix every downstream design choice: catalog size/integrity, ground-truth structure, conversation anatomy → context cap, cold/warm split, metadata/embedding sanity, the **recall-ceiling table** (per-channel and fused recall@{50,100,200,500}), and the leakage audit. Output `reports/eda.md` and the recall-ceiling table; hand decided parameters to R7, K1, R1/§8, K2, K3, L1, S1.

## 2. Interface / contract
P0 is a **notebook + report**, not an importable module. It *consumes* F1 loaders and F3 metric/recall helpers; it *produces* a markdown report and a parameters block that downstream modules read (manually, into their configs — P0 does not write other modules' YAML).

**Inputs (read-only):**
- `mcrs.data` (F1): the class-based loaders — `Catalog` (canonical `track_id` set, `id_to_index`, `id_to_metadata`), `TrackEmbeddings`, `UserEmbeddings`, `Users.profile`, `Conversations.turns(split)` / `Conversations.gold(...)`, plus `canonical_track_id`/`canonical_track_ids` and `segment_for`. See `10_F1_data_access.md` interfaces — P0 does **not** re-implement loaders.
- `mcrs.eval` (F3): `recall_at_k(retrieved, gold, k)` and `hit_rank(...)` primitives, the official-parity metric helpers, and the session-disjointness / no-leak split asserts. See `12_F3_eval_harness.md` — P0 reuses F3's recall function so probe numbers match the harness exactly.
- `mcrs.contracts` (F2): `TurnContext`, `UserProfile`, `Query` — P0 builds causal `TurnContext`s per dev turn to feed the probe; it does **not** define new types.
- Channel implementations for the probe (salvage, ported **read-only** for measurement, not yet wired into serve): BM25, dense-text, content-kNN-from-history, CF (see §4, §5 channel list).

**Outputs:**
- `reports/eda.md` — findings, plots, and the **DESIGN PARAMETERS** block (§6 deliverable).
- `reports/recall_ceiling.{md,csv}` — the recall-ceiling table (rows = channels + fused; cols = recall@{50,100,200,500}, split by cold/warm and overall).
- A `cache/eda/*.parquet` of per-turn retrieved-id lists per channel (so the fusion-K / weight sweeps in R7 re-fuse cheaply without re-running retrieval).

**Wiring (who reads P0's decisions):**
| Decision | Consumer module |
|---|---|
| `topk_internal`, fusion-K, cross-encoder-K, RRF weight priors, channel keep-list | **R7** (`46_R7_rrf_fusion.md`) — and per-channel R3/R4/R5/R6 inherit their pull depth |
| cold/warm threshold (history-length N), `is_cold` feature hint | **K1** (`50_K1_rerank_feature_builder.md`), **K2/K3**, channel gating in R5/R6 |
| context cap + truncation policy (recency window) | **R1** (`40_R1_query_construction.md`) + the §8 alignment rule; reranker doc-side cap → K3 |
| gold ∈ history rate → history-filter A/B default | **L1** (`60_L1_filter_assembly.md`) |
| catalog integrity / id-join coverage, missing-embedding set | **F1** (asserts), all retrieval channels |
| metadata coverage (tags/era/popularity), enrichment need | **A1** (`30_A1_catalog_assets.md`), K1 feature design |
| dialogue language(s), goal-field format/coverage | **R1**, **R2** (`41_R2_query_refinement_llm.md`), **S1** (`70_S1_responder.md`) |

## 3. Dependencies
- **Data/modules:** F1 (loaders + canonical id set) and F3 (recall/metric helpers + split asserts) **must exist first** — P0 is on the critical path right after them (`F1+F2+F3 → P0`, `000_INDEX.md` graph). F2 contracts for `TurnContext`.
- **Models/assets (probe-only, read-only):** the four candidate generators of §5.7 — BM25 index over baseline track docs, a dense-text encoder (default BGE-large-en-v1.5, per plan §7.2), the provided track embeddings (content-kNN), the provided user embeddings (CF). These are ported salvage/baseline impls used **only to measure ceiling**; the final serve channels are chosen+built in Phase 1 from P0's keep-list.
- **External APIs:** none. P0 is pure analysis; no LLM calls (enrichment/refinement levers are *informed* by P0 but not run here).
- **Config:** `config/eda.yaml` — probe knobs (§9).
- **Libraries:** `numpy`, `pandas`/`polars`, `matplotlib`, a tokenizer matching the chosen encoder (for token-length measurement, §5.3), `rank_bm25` or the baseline BM25, `transformers`/`sentence-transformers` for the dense encoder.

## 4. Design & logic
The notebook is 8 sections, one per plan-§5 question. Each section ends with an explicit **decision line** that lands in the §6 parameters block. Every probe runs over the **dev** split (the only split with golds we may inspect) using **causal** `TurnContext`s (utterances `1..t`, history `≤t`, no future, no gold in inputs — F2 guarantees this by construction).

**1. Catalog size & integrity.** Count distinct `track_id` in metadata; confirm **50k vs 1M** (the plan flags this as an open question — measure it, do not assume). Track/user embedding dims. Set difference: `track_id`s appearing in conversations/golds but **missing** from metadata or embeddings (the un-retrievable set — a hard recall ceiling). Duplicate-track detection (same metadata, different id). → hands F1 the integrity asserts and the missing-embedding id set.

**2. Ground-truth structure.** Inspect `music-crs-evaluator/make_ground_truth.py` output directly to confirm **exactly one gold per (session, turn)**. Define what gold is (track played/selected at turn `t`). Measure **rate that gold ∈ the user's history `≤t`** — this decides the L1 history-filter A/B (if golds are frequently re-listens, blanket history-filtering destroys recall). → hands L1 the A/B default; asserts single-gold-per-turn.

**3. Conversation anatomy.** Turn-count distribution (confirm 1–8). Per-turn utterance token-length **and cumulative context length** distribution (p50/p90/p99/max) under the chosen encoder's tokenizer → the **truncation policy** (§8): smallest recency-window/cap that keeps the latest utterance verbatim + goal field while fitting the 512-token encoder limit at, say, p95. Language detection. Goal-field presence/format/coverage. Fraction of utterances carrying explicit metadata (artist/genre/era terms) vs pure mood/semantic intent → informs R1 entity extraction and BM25-vs-dense weighting. → hands R1/§8 the context cap; R1/R2/S1 the goal+language facts.

**4. User profiles & cold/warm split.** History-length distribution → choose the **cold/warm threshold N** (e.g. cold = history ≤ N or degenerate CF embedding). Share of users with empty/degenerate (zero/constant-norm) CF embedding. Demographic coverage (country/age/gender non-null %). Compute the **cold/warm population share on dev** (so every later experiment can report segment-split metrics, plan §10). → hands K1/K2/K3 the threshold + `is_cold` hint; R5/R6 the gating threshold.

**5. Track metadata quality.** Tag vocabulary size + per-track coverage; release-date range + missing %; popularity distribution (long-tail shape, head/tail split); missing-field rates per column. → hands A1 the enrichment target (which fields are sparse enough to need doc2query/LLM blurbs) and K1 the usable feature columns.

**6. Embedding sanity.** Are provided track embeddings **text or audio** (or both)? Norm distribution per embedding type (flag un-normalized / degenerate). Nearest-neighbor coherence: for a sample of tracks, do top-k NN share artist/tags (musical sense)? Does `user_emb · track_emb` yield sensible personalized neighbors for warm users? → confirms the content-kNN and CF channels are *worth probing*; flags whether to normalize before cosine (feeds R4/R5).

**7. Recall-ceiling probe (CENTERPIECE — see §5 below for the exact procedure).**

**8. Leakage audit.** Verify Train/Dev/Blind **session disjointness** (no `session_id` overlap; reuse F3's split asserts). Confirm no future-turn field leaks into per-turn inputs (every `TurnContext` exposes only ≤t — verify the dev-turn builder against F2's causal guards). Confirm the gold is never in any channel's *input* query/history. → a leakage report section; any violation is a stop-the-line finding.

**Edge cases / no-leak constraints:** the probe builds the query from utterances `1..t` only; history-kNN/CF use history `≤t`; the gold is excluded from every channel input and only used to *score* recall. Turn-1 ("cold-turn") rows have empty/short context — report turn-1 recall separately (it stresses the dense/BM25 channels and is the hardest segment). Probe is **deterministic** (fixed dev order, fixed tie-break) so R7's later weight sweep is reproducible against the cached id-lists.

## 5. The recall-ceiling probe (plan §5.7 + §7.3.1) — exact procedure
**Goal:** for the dev golds, measure recall@{50,100,200,500} of **each candidate generator alone** and of **RRF fusion**, split overall / cold / warm / turn-1 — so the table directly sets `topk_internal`, fusion-K, and the channel keep-list, and confirms whether **recall@200 ≥ 0.90** (the §7 DoD gate) is reachable.

**Channels probed (each over the full `all_tracks` universe, canonical ids only):**
1. **BM25** over baseline track docs (`track_name + artist + album + tags + release_date`).
2. **Dense-text** bi-encoder (default BGE-large-en-v1.5; respect prefixes + 512-cap from §3 decision).
3. **Content-kNN from history** (recency-weighted pool of the user's track embeddings → catalog NN) — warm-only meaningful.
4. **CF** (`user_emb · track_emb` top-N) — warm-only meaningful.
5. **Fused** — weighted RRF over the kept channels (start `w_r ≡ 1`, `k=60`; P0 reports *unweighted* fusion as the ceiling and a small weight-sensitivity check, leaving the full weight sweep to R7 over the cached id-lists).

**Procedure (measured at the actual pull depth — this is the subtle part):**
- Pull each channel to depth **500** (`topk_internal_probe = 500`) **once**, cache the ordered id-list per (session, turn, channel) to `cache/eda/`. Recall@{50,100,200,500} are then prefix-cuts of that one pull — so the table reflects what each channel *can* deliver at depth, not a shallow default-60 truncation (the §7.3.1 silent-ceiling trap).
- Assert **every channel output ⊆ canonical catalog id set** before scoring (fusion-correctness req #1, §7.3) — a channel emitting non-canonical ids invalidates its recall number.
- Recall is computed with **F3's `recall_at_k`** (single gold per turn ⇒ recall ∈ {0,1} per turn; report the mean = hit-rate@K). Report per-channel **unique-recall contribution** (golds this channel hits that no other does) — the §7.3 req #2 signal for the keep/drop decision.
- Report a **pool-saturation curve**: fused recall@K for K ∈ {50,100,200,300,400,500}; the smallest K where the next 100 adds **< +0.5% recall** is the saturation point.

**Decisions the table SETS (plan §7.3.1):**
- **fusion-K** = smallest K where fused recall@K ≥ 0.90 (the §7 gate); if 0.90 is unreachable at 500, report the actual ceiling and flag it (drives the Phase-1 channel-enrichment priority — A1/R6).
- **`topk_internal`** = the per-channel depth at which each kept channel's recall plateaus, with the constraint **`topk_internal ≥ fusion-K`** (target 300–500). A gold a channel only surfaces at rank 300 must be pulled at ≥300 or fusion never sees it.
- **cross-encoder-K** prior = the K (≈100–200) capturing most of the reranker-recoverable golds, for K3's two-tier hand-off (`topk_internal ≥ fusion-K ≥ cross-encoder-K`).
- **channel keep-list** = channels with non-trivial unique-recall; drop or ~0-weight noise channels (req #2). Report cold/warm separately (CF/content-kNN may earn their place only on warm).

## 6. Eval & acceptance gate
**Gate:** the **recall-ceiling table is delivered** (`reports/recall_ceiling.{md,csv}`, all channels × {50,100,200,500} × {overall,cold,warm,turn-1}) **and every Phase-1+ design parameter in the §6 block is justified by a specific EDA number** with the cell/figure it came from. Not a model-metric gate — it is a *completeness + traceability* gate.

**DESIGN PARAMETERS block (the contract `reports/eda.md` must contain):**
- `fusion_k` — value + the fused recall@K that justifies it (and whether ≥0.90 was reached).
- `topk_internal` — value + per-channel plateau evidence; assert ≥ `fusion_k`.
- `cross_encoder_k` — value + the recoverable-golds rationale.
- `context_cap` / truncation policy — token cap + recency window + the cumulative-length percentile that justifies it.
- `cold_threshold` (history-length N) + cold/warm dev share + degenerate-CF share.
- `channel_keep_list` (+ per-channel unique-recall and overall/cold/warm recall) — keep/drop rationale per req #2.
- `gold_in_history_rate` → L1 history-filter A/B default.
- `missing_embedding_ids` count + `catalog_size` (50k vs 1M resolved) + dup-track count.
- metadata-coverage summary (tags/era/popularity/missing%) → A1 enrichment need.
- language(s) + goal-field coverage/format.
- leakage-audit result (split-disjoint PASS/FAIL; causal PASS/FAIL).

Each line is **cross-checked**: a downstream module's config value that contradicts P0's number is a review failure.

## 7. Tests
- **Schema / counts:** catalog row count, embedding-matrix shapes match `len(catalog)`; id-join coverage assert (conversation/gold ids ⊆ catalog ∪ reported-missing-set).
- **Single-gold-per-turn:** assert exactly one gold per (session, turn) over the whole dev split (from `make_ground_truth.py` output).
- **No-leak split:** Train/Dev/Blind `session_id` sets pairwise disjoint (reuse F3 asserts).
- **Causal probe:** for a sample of turns, assert the built `TurnContext` contains no turn-`>t` utterance and the gold ∉ (query, history, any channel input) — fires on violation (F2 constructor guards exercised).
- **Channel-id canonicality:** every probed channel's output ⊆ canonical catalog id set (fusion req #1).
- **Probe determinism:** re-running the probe with the fixed seed reproduces the recall-ceiling table byte-identically; prefix-cut recall@{50,100,200} is monotone in K.
- **Metric parity:** P0's `recall_at_k` is F3's function (import-identity check, not a re-implementation).

## 8. Failure modes & guards
- **Shallow-pull ceiling** (the §7.3.1 trap): probing at default depth 60 understates every channel → **pull at 500 once, prefix-cut**; guard = the depth assert in §5.
- **Non-canonical ids inflate/deflate recall:** a channel emitting raw/prefixed ids scores wrong → the ⊆-catalog assert before scoring; canonicalize at the channel boundary.
- **Train-eval bug masquerading as model failure** (prior-work hazard: raw UUIDs in context, role-filter mismatch): the causal-probe test + manual spot-check of 5 rendered queries before trusting any recall number.
- **Optimistic in-sample numbers:** P0 measures on **dev only** (golds we may see); no Train recall is reported as a ceiling (it would leak the same-split memorization trap). Probe channels are read-only — none is *trained* in P0.
- **Cold-turn under-report:** turn-1 (empty-history) recall reported separately so the warm average doesn't hide a cold collapse.
- **0.90 unreachable:** if fused recall@500 < 0.90, P0 does **not** silently pick a smaller gate — it flags the gap and routes it to A1/R6 (enrichment / extension channels) as the Phase-1 priority. The probe MEASURES the ceiling; it doesn't move it.
- **Stale numbers:** all reported numbers carry the data revision + commit hash; re-run on any data refresh.

## 9. Config knobs (`config/eda.yaml`)
- `splits.eval = "dev"`, `splits.leak_check = ["train","dev","blind_a","blind_b"]`.
- `probe.topk_internal_probe = 500`, `probe.recall_ks = [50,100,200,500]`, `probe.saturation_ks = [50,100,200,300,400,500]`.
- `probe.channels = ["bm25","dense_text","content_knn","cf"]`, `probe.fusion = {k: 60, weights: null}` (unweighted ceiling; weight sweep deferred to R7).
- `encoder.dense_model = "BAAI/bge-large-en-v1.5"`, `encoder.max_tokens = 512`, `encoder.query_prefix`/`passage_prefix`.
- `coldwarm.threshold_candidates = [0,1,2,5,10]` (sweep → pick N), `coldwarm.degenerate_cf_norm_eps`.
- `context.length_percentiles = [50,90,95,99,100]`.
- `paths.*` (catalog, embeddings, conversations, ground_truth, cache, reports), `seed`, `data_revision`.
- Reads F2's config schema shape; unknown keys rejected by the F2 loader.

## 10. Definition of Done & review checklist
- [ ] `reports/eda.md` written with the full **DESIGN PARAMETERS** block, each value traced to a cell/figure number.
- [ ] `reports/recall_ceiling.{md,csv}` delivered: all channels + fused × recall@{50,100,200,500} × {overall,cold,warm,turn-1}, plus per-channel unique-recall and the saturation curve.
- [ ] `catalog_size` (50k vs 1M) resolved; missing-embedding id set + dup count reported.
- [ ] `fusion_k`, `topk_internal` (≥ fusion_k), `cross_encoder_k`, `context_cap`, `cold_threshold`, `channel_keep_list`, `gold_in_history_rate` all decided **with numbers**.
- [ ] Leakage audit PASS (split-disjoint + causal) or a logged stop-the-line finding.
- [ ] All §7 tests green (incl. metric-parity import check against F3 and probe determinism).
- [ ] Decisions logged to `reports/experiments.md` and the project memory; the handoff table (§2) is filled so R7/K1/R1/L1/A1 can pull their inputs.
- [ ] Code/notebook review approved; no number carried over from prior work — every figure re-measured on the current data revision.

## 11. Build order & dependencies
**Built right after the foundation, on the critical path:** `F1 + F2 + F3 → P0`. Requires F1 loaders/canonical-id set, F2 `TurnContext`/contracts, and F3 recall-metric + split-asserts to all exist first (`000_INDEX.md` graph). **Blocks all of Phase 1+:** R7 (K/weights), K1/K2/K3 (cold/warm + features), R1/§8 (context cap), L1 (history A/B), A1 (enrichment target) each wait on P0's decided parameters. Analysis-only — produces a report + cached id-lists, no trained artifact, so it does not follow the §6.1 training-notebook skeleton (it is the `phase0_eda.ipynb` analysis exception).
