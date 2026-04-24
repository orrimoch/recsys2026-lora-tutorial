# Final Model Design — signals, specialists, aggregator

**Version**: 2 · **Date**: 2026-04-18 · **Status**: draft (iterating)

**v2 changes (user directive 2026-04-18):** explicit modular hybrid
architecture. Each data modality/signal gets a dedicated specialist;
components are reusable across experiments; a learned aggregator
blends all specialist scores. Aligns with the supplied repo tips at
`music-crs-baselines/tips/add_reranker.md` which explicitly recommends
cross-modal reranking (text + audio + user preference) and LLM-based
rerankers. Adds: sequential recsys, LLM-based preference/attribute
extraction, and expanded audio/CF roles.

Modular architecture for the final submission. Each signal in the data
has a dedicated specialist model; a learned aggregator blends all their
scores into the final per-(query, candidate) ranking. Principle from the
user: *"the sum of all components is greater than the sum"* — ensemble
wins over single-model replacements.

---

## Architecture diagram

```
                                     ┌─────────────────────┐
                                     │  query + history    │
                                     └──────────┬──────────┘
                                                │
           ┌──────────────┬──────────────┬──────┴───────┬──────────────┬──────────────┐
           │              │              │              │              │              │
   ┌───────▼────────┐ ┌───▼───────┐ ┌────▼─────┐ ┌──────▼────┐ ┌───────▼────┐ ┌───────▼────┐
   │ BM25 (5-field) │ │dense_meta │ │dense_lyr │ │CLAP-audio*│ │ cf-bpr**   │ │user-emb ** │
   │  LEXICAL       │ │  SEMANTIC │ │  LYRIC   │ │  AUDIO    │ │  COLLAB    │ │  PROFILE   │
   └───────┬────────┘ └───┬───────┘ └────┬─────┘ └──────┬────┘ └───────┬────┘ └───────┬────┘
           │              │              │              │              │              │
           └──────────────┴──────────────┴──────┬───────┴──────────────┴──────────────┘
                                                │
                                    per-query ranked list ×6
                                                │
              ┌─────────────────────────────────┴──────────────────────────────────┐
              │                      CANDIDATE POOL                                │
              │    union of top-60 from each specialist (usually ~60-200 unique)   │
              └─────────────────────────────────┬──────────────────────────────────┘
                                                │
                                    per-(query, candidate) features
                                                │
                         ┌──────────────────────┴─────────────────────┐
                         │ scores from each specialist                │
                         │ + popularity_log, artist_in_history,       │
                         │   cold_flag, release_year_delta,           │
                         │   bm25_rank, dense_cos, cross_encoder_logit│
                         └──────────────────────┬─────────────────────┘
                                                │
                                   ┌────────────▼────────────┐
                                   │    AGGREGATOR MODEL     │
                                   │  (LightGBM LambdaRank)  │
                                   └────────────┬────────────┘
                                                │
                                        final top-20

*  CLAP is STANDALONE-NULL (v5.1 rejected); exclude from ensemble unless
   we revisit with a better query encoder.
** cf-bpr and user-emb not yet integrated (v5.2/v5.5 queued).
```

---

## Specialists (current + planned, hybrid-modular v2)

Each specialist owns one signal. Interface is standardised so they
plug into the aggregator without bespoke glue:

- **Retriever**: `batch_text_to_item_retrieval(queries, topk) → list[list[str]]`
- **Scorer**: `batch_score_pairs(queries, candidates) → list[list[float]]`
- **Feature producer**: `batch_feature(queries, candidates) → list[list[float]]`

| signal | specialist | tid | status | interface | standalone ndcg@10 | ensemble role |
|---|---|---|---|---|---|---|
| Lexical | BM25 5-field (bm25s) | 002 | used | retriever | 0.0753 | first-stage retriever; RRF stream weight 1.0 |
| Semantic (metadata) | Qwen3-Embedding metadata + instruct | 006 | used | retriever+scorer | 0.0471 | RRF stream weight 0.4; LGB feature |
| Lyrical | Qwen3-Embedding lyrics + instruct | 010 | used | retriever+scorer | (in champion) | RRF stream weight 0.4; LGB feature |
| Audio (text↔audio) | LAION-CLAP MiniLM-L-6 text | 011 | rejected flat | retriever+scorer | ~flat vs champion | **exclude** — domain mismatch |
| Audio (alt) | MERT / music-specific audio encoder | — | future candidate | scorer | TBD | alternative to CLAP if audio signal is wanted |
| Fine-grained relevance | cross-encoder MS-MARCO-MiniLM | 013 | **rejected as standalone** | scorer | 0.0466 smoke | exclude as standalone; CE-logit feature **maybe** (expensive, deprioritised) |
| Collaborative | cf-bpr (precomputed 128-dim) | — | queued (v5.2) | scorer (user·track) | TBD | RRF stream; LGB feature |
| User profile | user-embedding (precomputed) | — | queued (v5.5) | scorer (user_emb·track_emb) | TBD | RRF stream (warm-users only); LGB feature |
| User profile (dem) | age/country/gender × track-popularity-by-demographic | — | queued | feature producer | TBD | LGB feature set |
| Sequential history | SASRec-like sequential recsys on chat-history music turns | — | queued (new) | scorer (history-seq → track-sim) | TBD | LGB feature |
| LLM preference extraction | Qwen3-0.6B extracts `{genres, moods, keywords, artists}` from chat history | — | queued (new; extends v8) | feature producer + query rewrite | TBD | BM25 corpus augmentation + LGB features |
| Query rewrite (HyDE) | Qwen3-0.6B generates pseudo-track metadata for BM25 | — | queued (v8) | query rewrite | TBD | feeds BM25 first stage |
| Popularity prior | log1p(play_count_train) | — | queued (v12) | feature producer | TBD | LGB feature |
| Cold-start routing | bucket(warm_user × warm_track) gate | — | queued (v11) | gate over weighted RRF | TBD | gates other specialists' weights |

**Notes on the new signal types called out 2026-04-18:**

- **Raw audio encoder (alternative to CLAP).** CLAP rejected flat; MERT
  (`m-a-p/MERT-v1-330M`, 330M params) is a music-domain self-supervised
  audio encoder that may capture timbre/genre better than CLAP's
  general-audio pretraining. Only worth trying if cf/sequential/LLM
  specialists also need a raw-audio complement — low priority per the
  current ledger.

- **Simple CF model (user × item).** Two candidate forms:
  1. Use `talkpl-ai/TalkPlayData-Challenge-Track-Embeddings[cf-bpr]`
     (precomputed 128-dim from the challenge org, trained on their
     train interactions). **Safe and cheap — v5.2 plan.**
  2. Train our own ALS / BPR / implicit-MF on the train-split
     music-turns as (user, item, 1) interactions. ~15k users × 47k
     tracks × sparse → fits comfortably in RAM; `implicit` library is
     Mac-compatible. Only if the precomputed cf-bpr is weak.

- **Sequential recsys on chat history.** The conversations have music
  turns in order; a SASRec-like self-attention model can embed "the
  sequence of tracks played so far" and score candidate tracks by
  next-item probability. Pytorch + MPS, ~1M params. New training
  experiment (v10.5 or similar). Complements CF by modelling
  within-session order, not just bag-of-interactions.

- **LLM preference/attribute extraction.** Prompt Qwen3-0.6B on the
  chat-history to produce structured fields — `{moods: [chill, upbeat],
  genres: [pop, indie], keywords: ['90s', 'female vocal'], artists: [...]}`.
  Two uses:
  1. **Query augmentation**: concatenate extracted keywords to the
     query before BM25 / dense retrieval. Improves lexical recall on
     underspecified queries.
  2. **Feature producer**: extract similar structured fields per track
     (once, offline) and compute tag-overlap scores as LGB features.
  Offline cost: 8000 dev queries × ~2s each on MPS ≈ 4h one-time,
  fully cached.

---

## Aggregator evolution

### Current: Weighted RRF (champion, tid=010)

```
score(q, d) = 1.0/(60 + rank_bm25(q, d))
            + 0.4/(60 + rank_dense_metadata(q, d))
            + 0.4/(60 + rank_dense_lyrics(q, d))
```

- Pros: no training, robust, catches our current signals.
- Cons: fixed weights (can't adapt per query or per cold/warm flag);
  can't use pointwise scores (cross-encoder logit, popularity_log).

### Next: Learned LambdaRank aggregator (v6-final)

Once enough specialists exist (≥4 orthogonal streams), retrain
`train_lgbm_reranker.py` with the full feature set:

```
features = [
  # from each specialist's score/rank
  bm25_rank, bm25_score_normalized,
  dense_metadata_cos, dense_metadata_rank,
  dense_lyrics_cos, dense_lyrics_rank,
  crossencoder_logit,       # from v7
  cf_bpr_score,             # from v5.2
  user_emb_cos,             # from v5.5
  # content features
  log1p_popularity,
  artist_in_history_count,
  release_year_present,
  tag_overlap_count,
  # cold-start flags
  cold_user, cold_track,
]
# objective = lambdarank; metric = ndcg@10
```

This is v6 reimagined under the aggregator lens — v6's MVP failed
because it had only 3 weak features. With 10-15 features spanning
orthogonal signals, it should beat the RRF-only champion.

### After learned aggregator: stacking / MoE

Only if (v6-final) plateaus. Typical stacking adds a second-level
meta-learner over specialist predictions. Heavy; reserve for late.

---

## Submission pipeline (for the blind sets)

Phase 2 per `project_submission_prep.md`:

1. Run each specialist on BLIND-A test queries.
2. Concatenate specialist scores per (query, candidate).
3. Apply the AGGREGATOR (trained on train+dev combined) to produce
   final top-20.
4. Write to `music-crs-baselines/exp/inference/blindset_A/*.json`.
5. Submit via CodaBench.

All specialists must be Mac-MPS-inferenceable at submission time.
No specialist that fails `feedback_sota_models_mac.md` enters the
final ensemble.

---

## Open questions

1. **Cross-encoder cost at submission scale**: ~21 min per 1000-session
   inference on M4. On blind-A this is fine (1000 sessions = one-time
   20-min pass). Worth it.

2. **Aggregator training data**: if the aggregator uses cross-encoder
   scores as features, we need to run cross-encoder on the train split
   during aggregator training. 15000 × 100 candidates × ~50 ms / batch
   = ~10 hours. Feasible overnight. Cached for any subsequent aggregator
   retrain.

3. **Feature normalization**: BM25 scores, dense cosines, and
   cross-encoder logits live on different scales. LGBM handles
   monotonic scales automatically (tree-based), so explicit
   normalization not required — but if the aggregator becomes an MLP,
   it is.

4. **Per-bucket aggregators?** Cold-user queries have different optimal
   weights than warm-user. Option A: single aggregator with cold_flag
   as a feature (simpler). Option B: two aggregators routed by
   cold-user flag. Start with A; try B only if bucket analysis shows
   large residual on cold-user cells.

---

## Action plan (hybrid-modular v2, 2026-04-18)

Re-ordered under the modular directive. Each new specialist provides
ONE signal with a standardised interface, so they reuse cleanly across
the RRF aggregator (now) and the LGB aggregator (v6-final, later).

1. **v7 (cross-encoder) — DECIDED to exclude as standalone specialist.**
   Smoke on 100 sessions showed ndcg@10=0.0466 << champion 0.0782;
   MS-MARCO→music-metadata domain gap too large. Branch kept; CE-logit
   stays a low-priority optional LGB feature (would cost 85 min cold
   run to populate — deferred unless aggregator needs the signal).

2. **v5.2 — cf-bpr collaborative specialist.** Cheap (precomputed
   128-dim embeddings already in `Challenge-Track-Embeddings`); same
   pattern as dense retrieval but with user_id lookup as the "query".
   Interface: scorer `(user_id, track) → dot(user_cf, track_cf)`.
   Ensemble role: RRF stream weight TBD; LGB feature.

3. **v5.5 — user-embedding profile specialist.** Precomputed from
   `Challenge-User-Embeddings`. Similar interface to v5.2 but
   different embedding space (profile ≠ interaction history).

4. **v8 (HyDE) + LLM preference extraction.** Single offline pass
   where Qwen3-0.6B both (a) generates pseudo-track metadata for query
   augmentation AND (b) extracts `{moods, genres, keywords, artists}`
   as structured features. Cache the output per (session, turn) so
   downstream specialists reuse.

5. **v10.5 — sequential recsys on chat-history music turns (NEW).**
   Small SASRec-style self-attention model; train on train split's
   conversation sequences. Feature producer: `(history_tracks → next_track)`
   probability. Novel signal not captured by other specialists.

6. **v11 — cold-start routing.** Gate on user-warm/track-warm buckets;
   dampen collaborative specialists for cold users, promote content
   specialists. Implements as weight-modulator over the existing wRRF.

7. **v12 — popularity prior.** Cheap score-level feature; critical for
   LGB aggregator; risky as standalone (v6 MVP confirmed).

8. **v6-final (AGGREGATOR) — LGBM LambdaRank over all specialist
   features.** Inputs: scores/ranks from specialists 1–7 above + user
   profile features (age/country/gender) + content features
   (popularity, release_year, tag_overlap). Train on full 15k-session
   train split. This is the FINAL RANKER; replaces weighted RRF once
   it beats the champion.

9. **v6-submission** — v6-final retrained on train+dev combined, for
   blind-set submission. See `project_submission_prep.md`.

10. **v13+ exotic** — defer indefinitely; the modular ensemble is
    expected to reach the ceiling well before generative-retrieval
    / DSI becomes necessary.

**Dispatch discipline going forward:** researcher agent focuses on
**designing specialist submodules** (per `feedback_researcher_synthesis.md`
v2), not proposing disconnected monolithic architectures. Each
researcher output specifies the signal, interface, integration point,
and reusability.
