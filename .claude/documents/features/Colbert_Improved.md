# ColBERT Improved — query upgrades + fine-tuning readiness (R8 → R8-FT)

> Successor plan to `47_R8_colbert_late_interaction_channel.md`. R8 shipped a pretrained
> GTE-ModernColBERT probe; it was marginal (+0.76% fused recall@200, 1.45% unique) and did NOT
> break the ~0.66@500 recall wall. This spec is the path from "pretrained, marginal" to
> "fine-tuned, recall-positive", plus the query-side upgrades that make ColBERT earn its place in
> the fused pool. It is organized so each lever is independently gated — we only keep what measures.
>
> Prereqs / context to read first: `47_R8_colbert_late_interaction_channel.md` (mechanics, MaxSim,
> per-channel query), `40_R1_query_construction.md` (QueryBuilder), `46_R7_rrf_fusion.md` (fusion),
> `53_K3b_ce_lora_finetune.md` (the LoRA CE teacher we reuse). Memory: probe result, data facts,
> "use PLAID not Voyager".

## 1. Why (motivation + current verdict)

Measured probe (pretrained `lightonai/GTE-ModernColBERT-v1`, A1-enriched 47k docs, 1000 dev sessions):

| Channel | r@50 | r@100 | r@200 | r@500 | unique |
|---|---|---|---|---|---|
| ColBERT (pretrained) | 0.226 | 0.278 | 0.336 | 0.421 | 0.0145 |
| Dense (bge-large) | 0.288 | 0.352 | 0.417 | 0.507 | 0.0185 |
| Fused w/o ColBERT | 0.413 | 0.491 | 0.559 | 0.660 | — |
| Fused w/ ColBERT | 0.411 | 0.496 | 0.567 | 0.661 | — |

Read of the result:
- Pretrained ColBERT is below dense at every depth → no transfer to music CRS out of the box. This is the textbook signal that fine-tuning (domain adaptation) is the lever, not architecture.
- ColBERT is cold-skewed: cold r@500 = 0.547 ≫ warm 0.378 — intent text carries cold turns where CF/history is empty. That is exactly the segment our fused pool is weakest on, so even a modest ColBERT lift is differentially valuable.
- The recall wall (~0.66@500) is fundamental to content-only retrieval. We are NOT promising to break it; we are promising (a) unique recall on the cold segment and (b) a stronger reranker base.

So: two coupled workstreams. Query upgrades (Section 4) make the most of the model we have; fine-tuning (Section 7) closes the transfer gap. Document-side (5) and serving robustness (6) are the supporting fixes that must land first or the measurements lie.

## 2. Goals / non-goals / success gates

Goals
- G1 (query): a focused, per-channel, faceted query for ColBERT that beats the full-dialog query on dev recall@{100,200}.
- G2 (fine-tune readiness): a reproducible, train/serve-parity-checked pipeline to fine-tune a ColBERT on TalkPlay query→gold with hard negatives + a cross-encoder teacher.
- G3 (fine-tune result): a checkpoint that beats pretrained ColBERT AND beats dense on dev recall@200, OR adds ≥ +1.0% fused recall@200 / ≥ +2% unique recall over the current fused pool.
- G4 (serving): PLAID index, self-healing cache, `topk_internal ≥ fusion_K`, segment-aware fusion weight.

Non-goals
- Breaking the recall wall (that needs R6 LLM-propose / non-content signals — out of scope).
- Replacing dense everywhere — ColBERT replaces R4 query-dense ONLY if the §6 ablation in R8 holds post-fine-tune; otherwise it is additive on the cold segment.
- Online/real-time query rewriting with an LLM in the serving loop (we stay offline/batch).

Decision gate (kill criteria). After Phase 2 (fine-tune), if the best checkpoint does not clear G3 on a clean dev split (no train-session leakage), we stop investing in ColBERT recall and keep it only as a cheap weight-swept cold-segment channel (or drop it). Recall is the real lever; do not pour effort past a failed gate.

## 3. Current state (what exists — do not rebuild)

- Channel: `mcrs/retrieval/colbert_channel.py` — `ColBERTChannel` (brute-force chunked MaxSim, content-only, deterministic top-k), `colbert_doc_text(cat, t, expansion_first=)`, `from_pylate(...)`. 19 tests green. This is the serving channel.
- Query: `mcrs/retrieval/query.py` — `QueryBuilder` (plain + enriched/`markers` template, recency, cap, `taste:` clause via `track_label_fn`). Does NOT yet populate `Query.per_channel`.
- Probe: `nb/phase1_colbert_probe.ipynb` — now restored to PyLate PLAID with a self-healing cache (see §6; Voyager was the source of the `pad_sequence` and empty-index crashes).
- Fine-tune scaffolding (salvage, parked, original branch `recall-union-lgbm`):
  - `salvage/scripts/build_colbert_train_data.py` — mines query→gold + hard negatives from TRAIN; `MOVES_TOWARD_GOAL` turn filter (turn-1 always); SASRec-free union pool (BM25+dense+same-artist); `--compact-query`, `--enrich-tags` parity flags; emits JSONL triples.
  - `salvage/scripts/train_colbert.py` — PyLate `Contrastive` loss, `SentenceTransformerTrainer`, in-loop dev-eval callback (turn-1 recall@k model selection), `--force` overwrite guard.
  - `salvage/scripts/build_colbert_index.py` — PLAID index builder with train/index doc-recipe parity banner.
  - `salvage/mcrs/retrieval_modules/colbert_late.py` — `make_colbert_doc_text_fn` (single source of doc text), `ColbertIndexRetriever` (PLAID), tag-enrichment helpers.
- Data facts (TalkPlay): in-session-only history; list-valued metadata; `goal_progress` train/test shift (TRAIN 44/43 MOVES/DOES_NOT, TEST 77/10); 74% seen-users; warm enriched-query tokens p50≈360/p99≈866; ~47,071 tracks; TRAIN 15,199 sessions / TEST 1,000.

## 4. Workstream A — Query improvements

Ordered cheapest-first. A1 is a blocker (without per-channel routing every lever below is untestable because the channel still gets the full dialog).

### A1. Per-channel query routing (blocker)
R1 must populate `Query.per_channel["colbert"]`; R7 must route each channel to its `query_key` override when present, else fall back to `Query.text`. Today R7 passes the same full query to every channel — which feeds ColBERT a long multi-turn string that additive MaxSim punishes (R8 §3.2). Implement the routing in R1/R7, add an integration test (mock encoder), and only then run the A/B sweeps below.

### A2. Right-size `query_maxlen` (cheap, do first)
ColBERT [MASK] query augmentation is non-linear: collapse below ~4 expansion tokens, sharp gains 4–24, peak near the training length (~32 for classic ColBERT; GTE-ModernColBERT tolerates longer with diminishing returns), slight decline past it. The [MASK] tokens mostly re-weight the real query terms — they do not add new search vocabulary. Action: measure focused-query token lengths (notebook §4b) at p95, set `query_maxlen` to fit the goal + latest utterance + facet tokens (never truncate the goal). Likely 48–64 for a faceted query. Sweep {32, 48, 64, 96}.
Source: ColBERT [MASK] term-weighting analysis (arxiv 2408.13672); PyLate paper (2508.03555).

### A3. Faceted / structured query (cheap)
Because MaxSim matches each query token independently, every informative facet token is an extra probe that can find its own best doc token. Build the focused query as: goal + latest utterance, then deduped facet tokens (genre / mood / era / activity / artist-style) extracted from the dialogue, trimmed to `query_maxlen`. Structure mainly guarantees facets are present and not crowded out; hard "labels-in-one-string vs free-text" superiority is unproven, so keep it light. This maps onto the existing `markers=True` enriched template in `QueryBuilder` — extend it with a `facets:` line.

### A4. Taste-profile injection (cheap win)
Turn in-session history track-ids into a short, recency-weighted, deduped `taste:` clause (artist – title, plus aggregated top genres/tags from the enriched catalog) — already supported by `QueryBuilder.taste_items` + `track_label_fn`. Bounded by `query_maxlen`; prefer the few most informative recent terms over many (MASK-reweighting finding). Place goal + current utterance first so taste tokens are the ones truncated, never the goal. This is the warm-segment lever (cold turns have no history).

### A5. Multi-query + RRF (the biggest evidenced recall lever — MQ4CS)
Generate 3–5 facet-focused query variants per turn (one emphasizing genre/mood, one era/artist, one activity/goal, etc.), run each through ColBERT independently, fuse the ranked lists with RRF. MQ4CS reports +6.8–9.6% Recall@100 over strong baselines; the gain comes from independent retrieval + rank fusion (vector-averaging blurs facets, fusion preserves them). This reuses R7's RRF stack directly. Offline LLM generation (paid Gemini, async bulk — see memory) so it is batch-friendly. Cap N and lean on RRF to down-weight an off-topic variant (drift guard). Gate: must beat single faceted query (A3) on dev recall@200 by a margin that justifies N× the encode cost.
Source: MQ4CS (arxiv 2403.19302).

### A6. HyDE / query2doc expansion (optional, gated, drift-risk)
Append a short LLM-generated hypothetical track description (genre/mood/artist-style prose) to the query — concatenate, do not replace the utterance+goal (query2doc style). Helps zero-shot recall but can hallucinate off-intent specifics (wrong artist/era) → recall loss on the gold. Mitigate: attribute-level descriptions over fabricated proper nouns; generate multiple and fuse (A5) rather than commit. Only pursue if A5 underdelivers.
Source: query2doc (2303.07678), HyDE survey.

### A7. CHIQ-style history cleanup (optional, offline)
Before building the query: disambiguate the latest user turn (resolve coref/acronyms), prune turns before a topic switch (drift control), optionally add one short pseudo-response. +2–5.5% MRR in conversational retrieval; runs offline once. Lower priority than A2–A5; revisit if multi-turn (warm) queries underperform.
Source: CHIQ (arxiv 2406.05013).

Query A/B matrix to run (after A1): {full-dialog (baseline)} × {focused A3} × {+taste A4} × {multi-query A5} at `query_maxlen` ∈ {32,48,64}. Report dev recall@{100,200,500}, unique recall, and cold/warm split. Keep the winner per segment (cold and warm may differ → feeds segment-aware fusion weight, §6).

## 5. Workstream B — Document-side

- B1. Keep expansion-first doc text (`colbert_doc_text(..., expansion_first=True)`): reorder enriched doc to `expansion | base` so right-truncation at `document_length` keeps the doc2query text. Guarantee A1 emits doc2query LAST in the raw doc, or the reorder is a no-op. Monitor `count_docs_over_budget`.
- B2. doc2query for recall, not precision: when filtering expansions, keep them broad — Doc2Query-- shows aggressive filtering can HURT recall while helping precision. Our objective is recall.
- B3. Right-size `document_length`: GTE-ModernColBERT trained at doc_len=300 but serves up to 32k (train-short/infer-long holds for ModernBERT). Set `document_length` to enriched-doc p99 (~866 warm) so we stop truncating tags/expansions; cost is index size, not quality. Measure with notebook §4b.
- B4. Tag enrichment parity: if using curated genre/mood tags (`make_colbert_doc_text_fn(enrich_tags=True, tag_min_freq, tag_top_k)`), the SAME recipe must be used in train-data, index, and serve. Print the recipe banner everywhere; add a preflight parity assert (§7.6).
- B5. Embedding dim economics (efficiency, not recall): ColBERTv2 @128-dim ≈ 732MB/10k docs; 64-dim halves storage at negligible BEIR loss; 48-dim can still beat ColBERTv2. If/when we fine-tune, train a 64-dim projection — at 47k docs and growing this materially cuts index/RAM with no measured recall cost. Token pooling × dim × PQ are independent multipliers.

## 6. Workstream C — Index & serving robustness (mostly landed)

- C1. Use PyLate PLAID, never Voyager. PLAID is end-to-end (`is_end_to_end_index=True`) — it scores inside the index and bypasses the token-path `rank.rerank → pad_sequence`. Voyager gave us two production crashes: (a) `retrieve(k=MAXK)` bumps `k_token` to MAXK; if `ef_search < k_token` Voyager returns no neighbors → `pad_sequence([])` "received an empty list of sequences"; (b) an interrupted build leaves an empty index that errors only at query time ("Entry point … element 4294967295, but only 0 elements present"). Both are gone with PLAID. (See memory: use-plaid-not-voyager.) Reference impl: `ColbertIndexRetriever` + `build_colbert_index.py`.
- C2. Self-healing cache: after loading a cached index, run a 1-query smoke test; rebuild on failure. Already added to the probe; replicate in the serving index builder so an interrupted build can never wedge a run.
- C3. `topk_internal ≥ fusion_K`: assert at R7 init for every channel — otherwise ColBERT silently never surfaces golds beyond rank `topk_internal`.
- C4. Segment-aware fusion weight: ColBERT is cold-strong / warm-weak. Use R7's `cold_weight`/`warm_weight` to up-weight ColBERT on cold turns. Run the per-segment weight sweep in P0 before committing.
- C5. Enrichment coverage alarm: log enriched-doc coverage; warn if <95% (heterogeneous enriched/raw doc text degrades MaxSim).

## 7. Workstream D — Fine-tuning (the core)

The pretrained gap is the whole reason for this spec.

DECISION (2026-06-18): keep the ORIGINAL feedback mechanism — PyLate `losses.Contrastive` (mined
hard negatives + in-batch negatives), with turn-1 dev-recall callback for checkpoint selection (the
original `salvage/scripts/train_colbert.py` approach). Distillation from the K3b cross-encoder (D3)
is documented as a DEFERRED upgrade we may layer on later; it is NOT in the initial build.

### D1. Training data pipeline (reuse + harden the salvage scripts)
Use `build_colbert_train_data.py`. Each row = (query, gold doc, hard negatives). Critical parity rule: the query string at TRAIN must match the query at SERVE (the §4 focused/faceted query), and the doc recipe must match the index recipe (§5). The salvage builder already has `--compact-query` and `--enrich-tags` — wire them to emit the §4-winner query format. Turn filter: turn-1 always; turns >1 require `MOVES_TOWARD_GOAL` (strips noisy off-goal mid-conversation targets; note the train/test goal-progress shift — do not filter the TEST/dev eval the same way).

### D2. Hard negative mining (the quality driver)
Mine hard negatives from multiple first-stage retrievers (BM25 + dense + same-artist union pool — already done), 15–30 per positive. Two upgrades:
- Multi-source + denoise: blend retriever-mined hard negs with random negatives (mxbai recipe ≈ 35% BM25-hard + 30% random + mined). Random negatives stabilize; all-hard can teach false distinctions.
- False-negative filtering: drop candidate negatives that the teacher cross-encoder scores above a threshold (likely unlabeled positives) — prevents penalizing true matches.
Skip rows where the gold is not in the pool (an unrecoverable recall miss for the ranker).

### D3. Teacher scores + distillation (DEFERRED upgrade — not the initial build)
Use the K3b LoRA cross-encoder (bge-reranker-v2-m3) as teacher. For each (query, [gold + negatives]) generate teacher relevance scores; train ColBERT with PyLate `pylate.losses.Distillation` (KL/margin-MSE on normalized teacher vs student scores). Data schema: per row a query, a list of documents, and a parallel list of teacher scores. This transfers the cross-encoder's fine-grained relevance into the cheap late-interaction student — the lever that most reliably lifts recall/ranking for ColBERT. Keep `Contrastive` (in-batch + hard negs) as the fallback/auxiliary objective for rows without teacher scores.
Source: PyLate docs (lightonai.github.io/pylate); JaColBERTv2.5 / answerai-colbert distillation writeups.

### D4. Base model + adaptation strategy
- Base: start from `lightonai/GTE-ModernColBERT-v1` (already an LI checkpoint, long-context ModernBERT) — continue-fine-tune rather than train a ColBERT from a bare encoder (cheaper, less data-hungry). Alternative to benchmark: `answerdotai/answerai-colbert-small-v1` (smaller, strong).
- Adaptation: LoRA on the backbone (matches the K3b pattern). LoRA makes catastrophic forgetting near-impossible by construction — important because TalkPlay is small (15k sessions). If full fine-tune, use mixed-batch (blend a slice of MS MARCO triplets) as the forgetting defense.
- Output dim: train a 64-dim projection (§B5) for index economics, after confirming no recall regression vs 128.

### D5. PyLate training config (starting point)
- Trainer: `SentenceTransformerTrainer` + `SentenceTransformerTrainingArguments`, `data_collator=ColBERTCollator(model.tokenize)`, `bf16=True`, `max_grad_norm=1.0`.
- Loss: `Distillation` (primary), `Contrastive` (fallback). Init `models.ColBERT(model_name_or_path, query_length=Q_LEN, document_length=D_LEN)` with the §4/§5-sized lengths.
- Hyperparams (small-domain): lr 1e-5 (full) / 1e-4–3e-4 (LoRA), epochs 1–3, batch 32 (bf16), warmup 0.1, weight-decay 0.01. Watch for overfitting — few epochs, early-stop on dev.
- Eval: `PyLateInformationRetrievalEvaluator` in-loop + the existing turn-1 dev-recall callback for model selection. SELECT ON A CLEAN DEV SPLIT with zero train-session overlap (the K3b stacking pivot showed a leak when selection scored on train sessions — do not repeat).

### D6. Train / index / serve parity (the silent-failure class)
One source of truth for both query format (§4 winner) and doc text (`make_colbert_doc_text_fn`). Preflight assert that train `--enrich-tags/--tag-min-freq/--tag-top-k/--d-len/--q-len` equal the index builder's; print the recipe banner in every script. A flag mismatch silently desyncs the model from the corpus it serves.

### D7. Evaluate the recall thesis on the FULL catalog
Pool-reranking cannot raise recall (it only reorders the union's own pool). Build a full-catalog PLAID index with the fine-tuned model (`build_colbert_index.py`) and measure recall@k over all 47k tracks — the only test that shows whether fine-tuned ColBERT surfaces golds the union missed. This is the G3 gate.

## 8. Evaluation & gates (summary)

- Channel-own: F3 `recall@{50,100,200,500}` on dev via the §4 focused query, overall + cold/warm + unique recall.
- Fused: recall lift vs current fused pool (R7), per segment.
- Reranker downstream: dev nDCG@20 with ColBERT pool feeding K2/K3 (recall@pool ≈ ceiling on nDCG).
- Gates: G1 (query beats full-dialog), G3 (fine-tune beats dense @200 OR +1% fused / +2% unique). Kill if Phase 2 misses G3 on a clean split.
- Always measure on train-only first; final blind = retrain on train+dev (submission protocol).

## 9. Phased rollout (each phase gated)

1. Phase 0 — unblock + parity: A1 per-channel routing; C1–C5 serving robustness; §B doc sizing; notebook §4b length measurement. (Mostly landed: PLAID + self-heal.)
2. Phase 1 — query upgrades (no training): A2 length, A3 facets, A4 taste, A5 multi-query+RRF. Run the A/B matrix; keep per-segment winners. Gate G1.
3. Phase 2 — fine-tune: D1 data + D2 negatives + D3 teacher scores → D4/D5 train (LoRA, distillation) → D6 parity → D7 full-catalog recall. Gate G3.
4. Phase 3 — efficiency + integration (only if G3 passes): 64-dim projection (B5), segment-aware fusion weight (C4), replace-R4 ablation (R8 §6), wire into R7, re-probe fused ceiling. Optionally MUVERA for fixed-dim retrieval (community PyLate bridges sionic-ai/muvera-py, smarthi/pymuvera; not in core PyLate, which uses FastPLAID/WARP).

## 10. Risks & mitigations

- Train/serve query or doc skew → silent recall loss. Mitigate: single source of truth + preflight parity assert (D6).
- Selection leakage (train sessions in the dev/selection metric) → inflated offline, regressed real. Mitigate: clean dev split, session-disjoint (D5). Already burned once (K3b).
- Query drift from LLM expansion (A5/A6) → wrong-facet retrievals. Mitigate: keep utterance+goal, fuse multiple, attribute-level not proper-noun.
- Overfitting small domain data → forgetting. Mitigate: LoRA / mixed-batch (D4), few epochs, dev early-stop.
- Chasing the recall wall. The wall is content-fundamental; do not over-invest past the G3 gate. ColBERT's realistic prize is cold-segment unique recall + a stronger reranker base, not 0.90 recall.

## 11. Open decisions (resolve before Phase 2)

- Primary objective: RESOLVED (2026-06-18) — contrastive only (the original way); distillation deferred (D3).
- Build basis: RESOLVED — reuse/promote the original implementation code (salvage scripts + `recall-union-lgbm` branch), adapt to the fresh-start data contracts, rather than writing from scratch.
- Base model: continue GTE-ModernColBERT vs benchmark answerai-colbert-small. (Recommend GTE-ModernColBERT first.)
- Adaptation: LoRA vs full + mixed-batch. (Recommend LoRA — reuses K3b infra, forgetting-safe.)
- Output dim: 128 then shrink, vs train 64 directly. (Recommend confirm 128, then 64.)
- Multi-query N and whether it runs at serve (latency) or is precomputed for dev only.

## 12. References (research)

Query: MQ4CS multi-aspect query + fusion (arxiv 2403.19302); CHIQ history cleanup (2406.05013); LLM4CS (2303.06573); ConvGQR (ACL 2023 2305.15645); ColBERT [MASK] term-weighting (2408.13672); query2doc (2303.07678); Doc2Query-- (less-is-more); ColBERT original (2004.12832); PyLate paper (2508.03555).
Fine-tune: PyLate docs (lightonai.github.io/pylate) — Distillation/Contrastive losses, evaluator; JaColBERTv2.5 / answerai-colbert (distillation, dim reduction); mxbai-edge-colbert (mixed-batch negatives, index economics); MUVERA bridges (sionic-ai/muvera-py, smarthi/pymuvera); GTE-ModernColBERT (train-short/infer-long).

## 13. File map (what to touch)

- `mcrs/retrieval/query.py` — per-channel + faceted/taste query (A1,A3,A4).
- `mcrs/retrieval/fusion.py` / `46_R7` — per-channel routing, `topk_internal≥K` assert, segment weight (A1,C3,C4).
- `mcrs/retrieval/colbert_channel.py` — `from_pylate` PLAID + smoke-test cache (C1,C2); 64-dim (B5).
- `nb/phase1_colbert_probe.ipynb` — PLAID restored + self-heal (done); A/B matrix harness (A2–A5).
- `salvage/scripts/build_colbert_train_data.py`, `train_colbert.py`, `build_colbert_index.py` — promote out of salvage; add teacher-score / Distillation path (D3), parity preflight (D6).
- `salvage/mcrs/retrieval_modules/colbert_late.py` — `make_colbert_doc_text_fn` single-source doc text.
