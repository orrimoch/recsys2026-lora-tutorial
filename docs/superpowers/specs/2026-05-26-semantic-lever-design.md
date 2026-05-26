# Semantic lever — design (2026-05-26)

Dual-use conversation→track semantic signal: an LLM (Qwen2.5-7B-Instruct) turns
the conversation into a HyDE pseudo-track, which we use both as a new recall
channel and (phase 2) as the discriminating feature the reranker lacks.

## Problem & motivation

Full-dev (n=8000) diagnosis factors our score exactly:

    dev nDCG@20 = recall@100 × conditional-nDCG-on-hits
    0.1425      = 0.4375     × 0.3257

Both terms are low and they multiply, so neither lever alone reaches the leaders
(nDCG 0.49–0.57): perfect reranking at current recall caps at 0.44; perfect
recall at current ranking caps at 0.33. We need ~0.7 × ~0.7.

Both deficiencies share one root cause: we have no strong model of "which track
fits this conversation." Recall dies on the 62% brand-new-artist vibe→track
cases (gold-name-in-context only 2.1%), where our dense retriever (0.336) loses
to BM25 (0.396); and the reranker is inert (conditional lift +0.007 over the raw
union order) because its features (same_artist/same_album/user-CF) are redundant
with the fusion and don't separate the gold from ~99 same-vibe distractors.

One semantic signal, computed once per turn, fixes both. See
`memory/project_semantic_lever_decision_2026_05_26.md`.

## Goal & success criteria

- Phase 1 (recall channel): on full dev, adding the HyDE channel to
  `wrrf_union_v1` lifts recall@100 from 0.4375 by a meaningful margin
  (target ≥ +0.03, comfortably clearing the 0.46 G1 gate). Measured channel-on
  vs channel-off, same pool depth.
- Phase 2 (reranker feature): with the semantic relevance + session-CF centroid
  features added and the LGBM retrained, conditional-nDCG-on-hits rises from
  0.3257 with a significant paired-bootstrap CI (lower bound > 0).
- Combined: dev nDCG@20 improves over 0.1425; if it clears, run Blind-A
  (full pipeline + official evaluator — offline eval mirrors online).

Each axis is measured independently (ablation discipline — one change at a time).

## Approach

Channel-first. Build and validate the recall channel before touching the
reranker, so the expensive LGBM rebuild is gated on the channel actually helping.

### Component 1 — HyDE generation (extends `mcrs/query_rewriters`)

Reuse `CMQR_REWRITER`'s machinery: the Qwen backend (vLLM/HF auto-select),
per-conversation disk caching, and chat-template plumbing. Add a HyDE generation
mode whose prompt asks the model to (a) infer an explicit intent query
(genre / mood / era / instrumentation / activity) and (b) write 1–3 plausible
"ideal next track" descriptions in catalog-metadata style — descriptions of
attributes, never real track or artist names. Modern prompting per house style:
brief CoT for intent, few-shot conversation→pseudo-track examples, low temp for
the intent query, slightly higher temp for HyDE-doc diversity, thinking mode off.

Interface:

    rewriter = HydeRewriter(model="Qwen/Qwen2.5-7B-Instruct", cache_root=...)
    out = rewriter.generate_batch(conversations)   # cached by hash(conversation)
    # out[i] = {"intent_query": str, "hyde_docs": list[str]}

### Component 2 — recall channel (`mcrs/retrieval_modules/hyde_qwen3.py`)

A new sub-retriever with the standard `batch_text_to_item_retrieval(queries,
topk, ...)` shape. Per query: HyDE-generate → embed each hyde_doc with the
existing metadata-qwen3 embedder (so they land in the precomputed catalog
embedding space — no re-embedding the catalog) → cosine search → fuse the 1–3
per-doc hit lists (RRF over the docs) → return top-K.

Wire into the union factory (`mcrs/retrieval_modules/__init__.py`, `wrrf_union_v1`
~L63) as a 4th sub_spec — ensemble-first, keep bm25 + qwen + same_artist:

    {"type": "hyde_qwen3", "topk_internal": 100,
     "weight": float(extra_config.get("w_hyde", 1.0))}

Cheap comparison baseline within Phase 1: CMQR's existing query-rewrite mode as a
channel, to confirm HyDE (pseudo-doc) beats plain query reformulation here.

### Component 3 — reranker features (phase 2)

Two new LGBM columns, computed identically in the builder
(`scripts/build_lgbm_features.py`) and at serve time
(`mcrs/rerankers/lgbm_rerank.py`) via one shared helper (train/serve-skew
discipline):

- `hyde_relevance`: max cosine(candidate metadata-qwen3 embedding, hyde_doc
  embeddings) — the discriminating "fits this conversation" signal.
- `session_cf_centroid_sim`: cosine(candidate cf-bpr, mean of played-track
  cf-bpr) — the session-level 0.240 signal, replacing the inert user-level
  `cfbpr_score` (top gain in training but doesn't generalize).

Retrain the LambdaRank model; measure conditional-nDCG lift.

## Data flow

conversation → HyDE LLM (cached) → {intent_query, hyde_docs} → qwen3 embed →
(a) cosine search → union recall channel;  (b) per-candidate max-cosine → LGBM
feature.

## Caching

Cache the LLM output (text) keyed by a hash of the conversation, persisted to
Drive, so the 8K-turn dev pass and re-runs never re-call the model. HyDE-doc
embeddings are recomputed per run (cheap with the 0.6B embedder). Reuse CMQR's
existing cache layout.

## Validation

- Phase 1: full-dev recall@{20,100} with the HyDE channel on vs off (and HyDE
  standalone), plus CMQR-rewrite-channel as the baseline. Extend nb72's recall
  reporting.
- Phase 2: full-dev conditional-nDCG-on-hits with vs without the new features
  (retrained LGBM), paired-bootstrap CI. Then combined dev nDCG@20.
- Validate the LLM pass on a 1–2K-turn subsample before the full 8K run.

## Risks & mitigations

- Weak qwen3 embedder is the known bottleneck. HyDE specifically helps a weak
  embedder by making the query look like a document. If recall doesn't move,
  fall back to a stronger embedding model for this channel only.
- LLM cost/latency over 8K turns → caching + subsample dry-run first.
- HyDE drifting to real/incorrect track names → prompt for attributes only;
  it's a retrieval cue, not an answer.

## Out of scope (YAGNI)

Cross-encoder (tried, dropped — do not resurrect). Sequential next-track model
(separate later bet). Responder/LLM-judge changes (nDCG-first). Increasing
training sample count (ruled out three ways — revisit only after a
discriminating feature lands). Re-tuning the bi-encoder (proven wrong lever).
