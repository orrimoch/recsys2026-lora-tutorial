# Design: Close the nDCG gap via a recall union + LightGBM aggregator

Date: 2026-05-25
Status: Approved (brainstorming) — pending implementation plan
Owner: Or Rimoch

## Problem / motivation

Our conversational music retriever scores nDCG@20 ~0.09 on Blind-A vs leaders'
0.49-0.57. A deep raw-data audit (memory `project_ndcg_gap_closer_recall_union_2026_05_25`)
established:

- Recall is a UNION problem. The live retriever fuses only 2 channels
  (BM25 + the multi-modal dense tower) -> recall@100 = 0.336. The union of
  trivial complementary signals reaches 0.498.
- The dominant signal — the next track is by an artist already in the session
  (38% of golds; same-artist recall@100 = 0.358) — is not a retrieval channel
  at all; it only leaks in as text inside the dense query.
- User-level CF is near-inert (NN recall@100 = 0.070); session-level CF works
  (centroid recall@100 = 0.240).
- The fine-tuned multi-modal dense tower underperforms plain BM25 (0.336 < 0.396)
  — it does not earn its training cost. (See also
  `project_stage_a_retrieval_gap_diagnosis_2026_05_24`.)

A lot of the target architecture is already half-built in the repo: a Stage C
LightGBM LambdaRank aggregator (nb 72, `mcrs/rerankers/lgbm_rerank.py`,
`scripts/build_lgbm_features.py`, configs `182-+lgbm-v5kto-blindA.yaml`,
`027-wrrf-lgbm-...`), plus `cf_bpr.py`, `sequential_rerank.py`, and a `chain.py`
reranker. This work completes that pipeline rather than building anew.

## Goal and success criteria

Complete the signals->specialists->aggregator pipeline with NO GPU training, so
each iteration is minutes (frozen embeddings + rules + a CPU tree ranker).
Gates, measured on the corrected nb 71 cell 6/7 dev eval (production query path):

- G1 — Stage A recall@100: 0.34 -> >= 0.46 (approach the 0.498 union ceiling).
- G2 — end-to-end dev nDCG@20: meaningfully above the current ~0.09-0.10
  (target a clear, CI-backed lift; stretch toward the 0.49-0.57 leader band).
- G3 — Blind-A composite nDCG@20: beat the current 0.09 submission.

## Architecture (two stages; cross-encoder retired)

Stage A — Recall = weighted RRF union of 4 complementary channels, each a cheap
specialist over frozen data (no training):

1. BM25 — lexical / stated intent. (Existing.)
2. Frozen Qwen-metadata-embedding ANN — conversation -> nearest catalog tracks in
   the provided Qwen text-embedding space. (Frozen; replaces the fine-tuned
   tower's semantic role.)
3. Same-artist generator — candidates = tracks by artists already played in the
   session. (New, rule-based.)
4. Session-CF ANN — centroid of the session's played tracks in CF-BPR space ->
   nearest catalog tracks by CF-BPR. (New; reuses `cf_bpr.py` vectors.)

Fused by weighted reciprocal-rank fusion into one top-100 pool. Channel weights
tuned on dev.

Stage C — Rerank = LightGBM LambdaRank aggregator over a per-(query, candidate)
feature vector; directly optimizes nDCG@20; trains in minutes on CPU. The
multi-modal cross-encoder (Stage B) is dropped entirely — not even a feature.

Feature set (per query-candidate pair):
- Per-channel: similarity score, rank, and RRF contribution from each of the 4
  channels.
- Modality cosines: Qwen-text, CLAP audio, CF-BPR (and optionally SigLIP image),
  computed query/session-side vs the candidate track.
- Structural: same-artist flag, same-album flag, count of times the candidate's
  artist appears in the session, candidate-in-user-train-history flag.
- Context: global popularity, position/turn in session, session length.

Stage B (multi-modal cross-encoder) is retired for this pipeline.

## Data flow

```
turn (conversation + session history + user_id)
  -> [4 recall channels each return top-K]
  -> weighted RRF -> top-100 candidate pool
  -> per-(query,candidate) feature extraction (build_lgbm_features.py, extended)
  -> LightGBM LambdaRank score -> top-20
  -> nDCG@20
```

## Reuse vs new

Reuse:
- nb 72 (LambdaRank training loop), `scripts/build_lgbm_features.py`,
  `mcrs/rerankers/lgbm_rerank.py`.
- The wRRF factory (`mcrs/retrieval_modules/__init__.py`) and `cf_bpr.py`.
- Config `182-+lgbm-v5kto-blindA.yaml` as the inference-wiring template.
- Corrected nb 71 cell 6/7 as the eval harness (recall@K, per-stream, nDCG,
  paired-bootstrap CI).

New / changed:
- A same-artist candidate generator and a session-CF-centroid retriever, wired
  as channels into a NEW fused `retrieval_type` whose channels are exactly
  {BM25, frozen-Qwen-ANN, same-artist, session-CF}. This replaces the
  fine-tuned multi-modal dense tower (used by `wrrf_bm25_multimodal_v1`) with the
  frozen-Qwen channel — the fine-tuned tower is dropped from recall (it loses to
  BM25 and isn't worth its cost). CLAP/CF/image survive only as LGBM features.
- Extend the LGBM feature extractor with the new channel signals + structural
  features above.
- Remove the cross-encoder from the rerank chain.
- A config for the full Stage A(union) + Stage C(LGBM) pipeline.
- Dev tuning of wRRF channel weights.

## Validation

- Every channel and feature group added incrementally; kept only if it produces
  a recall (channel) or nDCG (feature) lift whose paired-bootstrap 95% CI
  excludes 0 (ablation discipline).
- Offline gates G1/G2 on the corrected dev eval before any Blind-A submission.
- Final: one Blind-A submission via the config above; compare nDCG@20 to the
  0.09 baseline (G3).

## Scope guardrails (YAGNI)

In scope: the 4 recall channels + the LGBM aggregator + eval/tuning.

Out of scope (deferred to a separate Phase-B spec): the learned
session-transformer retriever; CLAP and SigLIP image as standalone recall
channels (they remain LGBM features only); any encoder fine-tuning; reviving the
cross-encoder.

## Risks / tradeoffs

- Frozen embeddings can't task-adapt — accepted, since the fine-tuned tower
  already loses to BM25; the signal is in the fusion, not the encoder.
- A tree ranker has no token-level query<->track text attention — accepted; most
  signal is structured, and BM25 + Qwen-text cover the lexical/semantic text role.
- Dropping the cross-encoder forgoes its (small, +0.0186) lift — accepted for
  simplicity and GPU-free iteration; it can return as a feature later if needed.
- Same-artist / session-CF channels add candidates but also noise; the LGBM
  aggregator + CI-gated ablation control for that.

## References

- Memory: `project_ndcg_gap_closer_recall_union_2026_05_25` (the evidence + plan),
  `project_stage_a_retrieval_gap_diagnosis_2026_05_24`,
  `project_leaderboard_gap_diagnosis_2026_05_24`,
  `project_modular_architecture` (signals->specialists->aggregator).
- Corrected dev eval: `colab/71_train_cross_encoder.ipynb` cells 6-7.
