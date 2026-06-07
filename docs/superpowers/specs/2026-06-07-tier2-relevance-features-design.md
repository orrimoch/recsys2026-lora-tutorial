# Tier-2 #4.1 — leak-free relevance features (qwen_meta_cos + bm25_score)

Date: 2026-06-07
Status: approved (design)

## Context

Stage 11 + the recall-by-turn diagnostic showed the reranker is at/below
recall-only and collapses on later turns even though recall@100 is flat across
turns — a pure RANKING failure. Root cause: the reranker has NO continuous
query->candidate relevance feature (only wrrf_rank + static columns), so it can't
tell a deep-but-relevant gold (recall@100 0.50 vs recall@20 0.32 => ~18% of golds
parked at ranks 21-100) from a deep distractor.

## Decision (approved)

Add TWO leak-free relevance features, wired at BOTH the feature builder AND serve
(the parity step the bge work skipped):
- `qwen_meta_cos`: cosine(frozen pretrained Qwen3-instruct(query), precomputed
  metadata-qwen3 catalog emb[tid]). Pretrained encoder => no in-sample label leak
  (distinct from the fine-tuned bge cosine that leaked).
- `bm25_score`: raw BM25 score of the query for the candidate (0 if outside the
  bm25 top-K). Lexical relevance axis.

## Parity-by-construction

A single shared `RelevanceScorer` (mcrs/rerankers/relevance_scorer.py) computes
both features; build_lgbm_features and crs_baseline both call it, so train and
serve magnitudes are identical by construction. The features are PASSTHROUGH on
both sides (mirrors bge_cos): the scorer injects them into candidate dicts;
extract_features and lgbm_rerank just read them.

- Pure core: `relevance_feats_for_candidates(q_vec, catalog_norm, tid_to_idx,
  bm25_map, cand_tids) -> [{qwen_meta_cos, bm25_score}]` (unit-tested).
- Wrapper: `RelevanceScorer(dataset, splits, corpus, cache_dir)` loads a frozen
  DENSE_PRECOMPUTED(metadata, instruct) (shared-singleton encoder, no double
  load) + BM25_MODEL; `feats_for_batch(queries, cand_tids_per_query)`.

## Wiring

- `extract_features`: gated `with_relevance` -> row["qwen_meta_cos"] =
  float(c.get("qwen_meta_cos", 0.0)); row["bm25_score"] = float(c.get("bm25_score", 0.0)).
- `build_lgbm_features.build`: `--with-relevance` -> per-chunk, RelevanceScorer
  injects the two keys into candidate dicts before extract_features (mirrors the
  bge chunk injection).
- `lgbm_rerank`: add qwen_meta_cos / bm25_score to extended_keys; read each behind
  the f_idx guard (default 0.0).
- `crs_baseline.batch_chat`: build a RelevanceScorer once; inject the two keys
  into extra_features_per_candidate at serve (TRUE parity).

## Regression safety

- Off by default: extract_features omits the columns unless with_relevance; the
  reranker reads them only if the trained model lists them (f_idx guard). Existing
  parquets/models/config 194 unchanged. Full-suite gate. Tier-end QA.

## Tests (TDD)

- relevance_feats_for_candidates: cos = normalized dot; bm25 = map-or-0; missing
  tid -> cos 0.0.
- train/serve parity (mirror test_lgbm_bge_feature): identical (qwen_meta_cos,
  bm25_score) in a candidate dict -> identical values from extract_features and
  lgbm_rerank._compute_feature_matrix.
- off-by-default: no columns without with_relevance; reranker default 0.0 when a
  model lists the feature but the candidate lacks it.

## Validation gate (Colab)

Rebuild features with --with-relevance, retrain (early-stop on the temporal
holdout, NOT the leaky val), report turn-stratified nDCG@20 vs the recall-only
baseline and the prior reranker. Ship only if turn-1 nDCG beats recall-only.

## Non-goals

The early-stop-on-temporal-holdout change is a separate follow-up (kept out of
this increment per the user). No serve config change beyond the injection.
