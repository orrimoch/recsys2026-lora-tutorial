# Content-fused SASRec recall channel — design (2026-05-27)

A session-based sequential next-item model whose items are represented by fused
frozen multimodal content (so brand-new-artist tracks are scorable), added as a
new channel of the recall union. This is P0 of the ML-audit roadmap — see
`memory/project_ml_audit_sasrec_pivot_2026_05_27.md`.

## Problem & motivation

Full-dev decomposition: `nDCG@20 = recall@100 × conditional-on-hit = 0.4375 ×
0.3257 = 0.1425`. The binding constraint is recall, and the root cause is that
62% of golds are brand-new artists (not in session history) that every current
channel structurally misses — BM25/dense are lexical-ish, `same_artist` only
returns same-session artists, and there is no behavioral/sequential channel at
all. The task is literally next-track-in-session prediction; a sequential model
is the canonical fit, and content-based item representations are what let it
score cold (new-artist) items.

## Goal & success criteria

- Add a `sasrec_seq` channel to `wrrf_union_v1` (opt-in, ensemble-first) and
  measure its recall@100 lift on full dev, channel-on vs channel-off.
- Target: meaningful recall@100 lift over the 0.4375 baseline (the audit's
  speculative projection is 0.55–0.65 — treat as a hypothesis to test, not a
  promise). Clear the 0.46 G1 gate comfortably.
- Leakage-safe: the model is fit on train-split sessions only; dev/Blind
  sessions are encoded at inference only.

This spec is P0 (model + recall channel). P1 — feeding the model's score into
the LGBM reranker as the relevance feature — is a separate spec, gated on this
channel's recall result (ablation discipline: move one term at a time).

## Architecture — content-fused, dialog-conditioned SASRec (session-based, no user-ID dependence)

Item tower (the cold-start fix). For each catalog track, concatenate three
frozen precomputed per-track embeddings from
`talkpl-ai/TalkPlayData-Challenge-Track-Embeddings` — `metadata-qwen3_embedding_0.6b`
(1024) + the LAION-CLAP audio embedding (512) + `cf-bpr` (128) = 1664-dim — and
pass through a small trained MLP (Linear → GELU → Linear) to a shared dim
`d = 128`. Only the MLP is trained; inputs are frozen. Missing modalities are
imputed (artist-mean → category-mean → global-mean, per the project rule), so no
track is dropped. The result is a 47K×d item-representation matrix that is
computed identically for warm and cold items — that identity is what makes a
brand-new artist's track scorable. (Exact CLAP column key is confirmed against
the dataset schema in the first implementation task.)

Sequence encoder. Causal self-attention (2 layers, dim `d`) over the sequence
`[context_token, item_1, …, item_t]`, producing a session-state vector from the
final position. Position 0 is a conversation context token: the Qwen3 embedding
of the dialog so far (the same `role: content` text the BM25/dense channels
query on, via the shared Qwen3-Embedding-0.6B) projected to `d`. It conditions
the model on the user's stated intent and makes it productive from turn 1, when
no tracks have been played yet. Positions 1..t are the played-track item-reprs.
Learned positional embeddings index the played-track SUBSEQUENCE position (1st
played, 2nd played, …), NOT the raw `turn_number` (which has gaps from
interleaved user turns) — this guards the off-by-one that field would invite.
Sessions are short (~8 turns), so `max_len ≈ 50` is ample; the model is
small/fast.

Scoring. `score(session, item) = dot(session_state, item_repr)`. The same model
serves both downstream uses: rank all 47K item-reprs → recall channel (this
spec); score a single candidate → reranker feature (P1, later).

Deliberate v1 calls (approved):
- No per-item trainable "delta" in v1. A delta only helps warm items (it is zero
  for the cold 62%) and adds delta-dropout training complexity. v1 uses pure
  fused content; the delta is a v2 lever if warm-item memorization is the limit.
- Loss: next-item prediction with temperature-scaled full-catalog softmax CE
  (47K classes is small enough to be exact; its gradient already concentrates on
  the highest-scoring — i.e. same-vibe — negatives). This is the strong objective,
  not the naive one (naive = original SASRec's 1-negative BCE, which gSASRec
  showed is weak/overconfident). Documented first refinement, if recall@20 lags
  recall@100 (the same-vibe-confusion symptom): switch to sampled softmax with
  mined HARD negatives (content-nearest-neighbours + same-artist/album non-next
  tracks) plus a logQ sampling-bias correction. Full softmax needs no such
  correction (no sampling).
- No user embedding. The audit showed user-level signal is inert (0.070); the
  model relies on session dynamics + the dialog context token only.
- Exclude `conversation_goal` (category/specificity/listener_goal). Per
  crs_baseline.py:663 it is NOT available at Blind inference, so conditioning on
  it would reintroduce train/serve skew on the scored set; it is also coarse and
  largely subsumed by the conversation context token.

## Data flow

Train: train-split sessions → per session, the ordered music-turn track_ids plus
the dialog text up to each turn → examples `[context_token(dialog 1..t),
item-reprs of played 1..t]` → predict target t+1 → temperature-scaled
full-softmax CE.
Serve (channel): a dev/Blind session's dialog text + played track_ids →
`[context_token, item-reprs]` → session state → dot against the precomputed
47K item-repr matrix → top-K track_ids.

## Components / files

- Create `music-crs-baselines/mcrs/retrieval_modules/sasrec_seq.py`: the model
  (item-fusion MLP + SASRec encoder) and the retriever-interface channel
  (`batch_text_to_item_retrieval(queries, topk, user_ids, batch_context)`, where
  `queries` is the dialog text for the context token — embedded via the shared
  Qwen3-Embedding-0.6B from `dense_precomputed.py` — and `batch_context['history_tids']`
  supplies the played history like `same_artist`; loads trained weights +
  precomputes the item-repr matrix once).
- Create `scripts/train_sasrec.py`: build sequences from train sessions, train,
  save weights + config to the cache/Drive.
- Reuse the frozen-embedding loaders (the `Challenge-Track-Embeddings` load +
  imputation patterns in `dense_precomputed.py` / `cf_bpr.py`).
- Modify `mcrs/retrieval_modules/__init__.py`: a `sasrec_seq` build branch + a
  gated 4th channel in `_wrrf_union_v1_specs` (e.g. `use_sasrec`), mirroring the
  existing `use_hyde` pattern.

## Validation (ablation-disciplined)

- Recall channel: full-dev recall@{20,100} with `use_sasrec` on vs off (and
  SASRec standalone), via the nb72 cell-8-style ablation harness.
- Also report SASRec standalone recall to see how much the new-artist segment
  improves (optionally stratified by same-artist vs new-artist golds).
- Decision gate: if recall@100 lifts meaningfully, proceed to P1 (the LGBM
  relevance feature); if flat, inspect the item-fusion / training before
  abandoning.

## Risks & mitigations

- The 0.55–0.65 recall projection is speculative — validate before trusting it.
- Cold-item quality depends entirely on the frozen embeddings; if CLAP/metadata
  don't separate new-artist golds from neighbors, recall won't move. Mitigation:
  the standalone + stratified recall readout tells us which modality carries the
  signal; we can reweight or drop a modality.
- Missing-modality rows (e.g. cf-bpr drops ~616 empties; CLAP/metadata may have
  gaps) → impute, never drop.
- Full-softmax memory on a small GPU → sampled-softmax fallback.
- Leakage: fit on train sessions only; never touch dev/Blind during training.
  Frozen content embeddings are catalog content, not interaction labels.

## Out of scope (YAGNI)

Per-item trainable delta (v2). The P1 LGBM relevance feature (separate spec,
gated on this recall result). SPLADE (P2) and CLAP-only session-centroid (P3)
channels. HyDE (demoted to opt-in; do not let it block this). User-ID
personalization (inert). The generative-retrieval head (P4 contingency).
