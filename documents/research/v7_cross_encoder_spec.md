# v7 — Cross-Encoder Reranker over BM25 Top-K (Spec)

Status: ready-to-code. Author: researcher agent, 2026-04-17. Target experiment tid: `013-bm25-then-crossencoder-v1`. Parent champion at write time: tid=010 (wRRF + lyrics, nDCG@10 = 0.0782). This spec does NOT ship any code; the orchestrator implements from here.

Pipeline shape (same as v4.1 / tid=008 but with a deeper reranker):

```
query  ──► BM25 (5-field corpus, top-100) ──► CrossEncoder scores (query, doc_str) ──► top-20
```

Cross-encoder jointly attends over `[CLS] query [SEP] doc [SEP]` and returns a scalar relevance score per pair. This is the classic retrieve-then-rerank pattern (Nogueira & Cho 2019, MonoBERT; Thakur et al. 2021, BEIR) — complementary to v6 (feature-based LambdaMART) because the cross-encoder learns token-level interactions BM25 features cannot represent.

---

## 1. Model pick

**Default: `cross-encoder/ms-marco-MiniLM-L-6-v2`** (Hugging Face repo id).

Rationale:
- 22.7M params (6-layer MiniLM), fp32 weights ≈ 91 MB on disk — comfortable on 16 GB unified RAM with Qwen3-Embedding already loaded (memory feedback `feedback_sota_models_mac.md` explicitly lists this model as the v7 pre-approved pick).
- Pretrained on MS MARCO passage ranking (≈40M pairwise labels): the de-facto SOTA-for-size checkpoint in the sentence-transformers ecosystem; widely-cited baseline (thousands of downstream uses). Stays well within the SOTA-and-Mac-feasible rule (memory `feedback_sota_models_mac.md`).
- MPS-compatible out of the box via `transformers.AutoModelForSequenceClassification` — same op-set we already run for Qwen3-Embedding, no exotic ops.
- Standard ~50 ms / batch-32 × 512 tokens on M4 MPS in published benchmarks (sentence-transformers docs). See §7 for our projection.

**Alt (larger, only if L-6 gives strong @1 signal):** `cross-encoder/ms-marco-MiniLM-L-12-v2` (33M params, +10 pp NDCG@10 over L-6 on MS MARCO dev at ~1.5× latency). Treat as v7.1 follow-up, not a first shot.

**Multilingual (deprioritized):** `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (118M). EDA check run at spec time:
- 200 random dev sessions: 3 (≈1.5%) have any non-ASCII content, and those hits were mostly curly quotes / isolated Portuguese/K-pop title references — NOT full non-English sessions.
- 200 random catalog tracks: 0 non-ASCII characters in `track_name/artist_name/album_name`.
- Conclusion: catalog is English-dominant, conversations are English-dominant. Pay the 5× parameter tax of the multilingual checkpoint only if L-6 is promoted AND an error analysis shows non-English queries as a concentrated weakness.

---

## 2. Scoring path

Cross-encoder consumes `(query, doc)` text pairs and returns one logit per pair (higher = more relevant). For our task, `doc` is the stringified track metadata — identical format to what `MusicCatalogDB.id_to_metadata(track_id)` produces, using the v2a 5-field corpus for consistency with the BM25 first stage:

```
doc = id_to_metadata(tid) over corpus_types=[
    "track_name", "artist_name", "album_name", "release_date", "tag_list",
]
# Example: "track_id: abc123, track_name: lose control, artist_name: missy elliott,
#           album_name: ..., release_date: 2005, tag_list: rap, female voc, ..."
```

Pseudo-Python for single-pair scoring (real implementation should batch — §3):

```python
def _score_pair(self, query: str, candidate_tid: str) -> float:
    doc = self.catalog.id_to_metadata(candidate_tid)  # uses 5-field corpus
    enc = self.tokenizer(query, doc, truncation=True,
                         max_length=512, return_tensors="pt").to(self.device)
    with torch.no_grad():
        logit = self.model(**enc).logits.squeeze(-1)   # shape (1,)
    return float(logit.cpu())
```

Notes:
- `AutoModelForSequenceClassification` returns a single regression logit for this checkpoint (num_labels=1). Do NOT softmax; the raw logit is the rank score.
- `truncation=True, max_length=512` — MS-MARCO checkpoints were trained on 512-token inputs. Our doc strings are ~30–80 tokens; query + doc fits comfortably.
- `id_to_metadata` already produces a stable lowercased comma-separated string — reuse it unchanged to keep corpus-stringification identical across experiments.

---

## 3. Batching strategy

Per query we retrieve `topk_internal=100` BM25 candidates (same as v4.1). Flatten to `100 × B_queries` (query, doc) pairs, run the cross-encoder in mini-batches of `BATCH_SIZE=32` pairs on MPS, reshape back to `(B_queries, 100)` scores.

Recommended values:
- `BATCH_SIZE = 32` (start). If MPS memory allows, bump to 64 — MiniLM-L-6 has tiny activations (~6 × 32 × 512 × 384 fp32 ≈ 38 MB per batch).
- `topk_internal = 100` (copy v4.1). Rerank to caller's `topk` (usually 20).
- Within one query, run all 100 pairs contiguously (same query string tokenized 100× — cheap, and sorting later is O(100 log 100)).

Memory budget (fp32):
- Model weights: ≈ 91 MB.
- Activations for batch 32 × 512 tokens: ≈ 32 × 512 × 384 × 6 layers × 4 bytes ≈ 150 MB worst-case (forward-only, no gradients).
- Total resident footprint: ≈ 250 MB on top of Qwen3-Embedding's ≈ 1.2 GB. Well within 16 GB unified RAM.

For smoke tests, use `fp16` weights (cast on load) to halve model+activation memory. Production run stays fp32 to avoid MPS fp16 numerical quirks in `AutoModelForSequenceClassification` (some ops fall back to CPU in mixed precision, killing throughput).

---

## 4. Caching strategy

Cross-encoder scores depend on BOTH query AND candidate, so the cache key is the tuple `(query_str, track_id)`. No key collapses are possible — every new query forces a fresh inference over its ~60–100 candidates.

Storage layout:
- Single shared pickle per model checkpoint: `experiments/cache/rerank/ms-marco-MiniLM-L-6-v2.pkl`
- In-memory type: `dict[tuple[str, str], float]` where `str(float)` is fp32.
- Load pattern: mirror `DENSE_PRECOMPUTED._load_query_cache` — load on init, mark dirty on writes, `os.replace(tmp)` atomic save in `batch_text_to_item_retrieval` and `__del__`.

Capacity estimate (devset):
- 1000 dev sessions × avg ~8 music turns per session ≈ 8000 queries.
- 100 candidates per query → 800k entries.
- Each entry: ~80 B tuple key + 8 B float → ~90 B. Total ≈ 70 MB unpickled, ~40 MB on disk (pickle compresses keys).
- Even if storage blows to 200 MB, still comparable to existing dense caches (already ~60 MB each for query embeddings).

Cache reuse across experiments:
- Key is raw `(query, tid)` — independent of first-stage retriever choice. A future v7.1 (L-12-v2) or v7.2 (BM25→dense→cross-enc cascade) will hit this cache whenever the same (query, tid) pair appears, which happens heavily since BM25 candidate lists are stable.
- Invalidation: checkpoint name is embedded in filename. Never mutate existing checkpoint's cache.

Do not cache PER-QUERY shas — tuple-keyed dict is simpler and already small enough. Skip the sha split the user proposed.

---

## 5. Factory key design

New factory key in `mcrs/retrieval_modules/__init__.py`:

```python
elif retrieval_type == "bm25_then_crossencoder_rerank_v1":
    # BM25 (v2a 5-field corpus) first-stage top-100 → cross-encoder rerank → top-20.
    # Analogous to bm25_then_dense_rerank_v1 (v4.1 / tid=008) but reranker is
    # a jointly-attended cross-encoder (Nogueira & Cho 2019, MonoBERT pattern).
    from .cross_encoder_rerank import CROSS_ENCODER_RERANK  # new module, §6
    return CROSS_ENCODER_RERANK(
        dataset_name, track_split_types, corpus_types, cache_dir,
        first_stage_spec={
            "type": "bm25",
            "corpus_types": [
                "track_name", "artist_name", "album_name",
                "release_date", "tag_list",
            ],
            "topk_internal": 100,
        },
        reranker_spec={
            "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "batch_size": 32,
            "max_length": 512,
            "corpus_types": [
                "track_name", "artist_name", "album_name",
                "release_date", "tag_list",
            ],
        },
    )
```

Rationale for composition: wrap a class (not modify `SEQUENTIAL_RERANK`) so the reranker step is swappable (dense-cosine vs cross-encoder vs LambdaMART) without branching inside one module. Same design principle as v6's `LGBM_RERANKER`.

---

## 6. Implementation files

| file | status | purpose |
|---|---|---|
| `music-crs-baselines/mcrs/retrieval_modules/cross_encoder_rerank.py` | NEW | Class `CROSS_ENCODER_RERANK` with `batch_text_to_item_retrieval(queries, topk)` matching `BM25_MODEL`'s signature. Loads the HF checkpoint once (lazy, on first call), runs first stage, batch-scores pairs, sorts, trims. Owns the `(query, tid) → score` cache. Interface lives under `retrieval_modules/` (not `rerankers/`) to keep it callable as a retriever by the existing factory. Keep it parallel to `sequential_rerank.py`. |
| `music-crs-baselines/mcrs/retrieval_modules/__init__.py` | EDIT | Add `bm25_then_crossencoder_rerank_v1` branch (see §5). Edit allowed per plan §12 (factory-key additions are additive). |
| `music-crs-baselines/config/013-bm25-then-crossencoder-v1.yaml` | NEW | Copy of `008-bm25-then-dense-rerank.yaml` with `retrieval_type: "bm25_then_crossencoder_rerank_v1"`. All other fields unchanged (including `cache_dir`, `device: "mps"`). |

Within `cross_encoder_rerank.py`:

- Construct `MusicCatalogDB(dataset_name, split_types, reranker_spec["corpus_types"])` for stable `id_to_metadata`.
- Lazy-load HF model+tokenizer on first `batch_text_to_item_retrieval` call (same pattern as `DENSE_PRECOMPUTED._get_encoder`). Keeps imports cheap for unit tests.
- On each batch call:
  1. `first.batch_text_to_item_retrieval(queries, topk=first_topk)` — 100 cands per query.
  2. Build a flat list of `(query, tid)` pairs NOT in cache.
  3. Tokenize + score in mini-batches of 32 (`with torch.no_grad()`).
  4. Write back to cache; save atomically.
  5. For each query, sort 100 cached scores descending, take top-`topk`.

- Split discipline (per `feedback_no_data_leakage.md`): the cross-encoder is pretrained on MS MARCO — a catalog-independent corpus — so no train/dev leakage risk from the checkpoint itself. We do not fine-tune it. `MusicCatalogDB` uses `track_split_types=["all_tracks"]` which is item-level metadata, safe across splits.

---

## 7. Expected wallclock on M4 MPS

Back-of-envelope:

- ~8000 queries on devset × 100 candidates = 800k (query, doc) pairs.
- MiniLM-L-6 on MPS: ~50 ms per batch of 32 pairs (conservative; sentence-transformers benchmarks show ~30 ms on M-series).
- 800k / 32 = 25k batches × 50 ms = **1250 s ≈ 21 min** for a cold cache full pass.
- Warm cache: cache hit-rate ≈ 100% on re-runs → <60 s (pickle load + sort).

Smoke-100 is worthwhile here because 21 min is non-trivial and the MSMARCO→music domain gap is untested:
- Run `subset=100` first (~125 s → ~2 min). If @1 signal is positive and ndcg@10 shows a directionally sensible number (say > 0.075), proceed to full 1000-session dev eval.
- If @1 or ndcg@10 is flat/regressed on the smoke, stop and log the null before burning 21 min.

Comparison to v4.1 (tid=008): dense rerank ran in 36 s warm. Cross-encoder is ~35× slower first pass due to joint-attention over pairs. One-time cost; warm re-runs are free.

---

## 8. Risks / watch-outs

1. **Domain gap (PRIMARY risk):** MS MARCO is web passage ranking; our "docs" are comma-joined track metadata with no prose. Cross-encoders trained on natural-language passages MAY assign uniform low scores and lose to BM25's lexical precision. Mitigation: the smoke-100 in §7 surfaces this cheaply; if the ndcg drops below the parent (0.0782) the rerank is likely over-ranking tracks whose metadata happens to contain BM25 stop-words.

2. **Signature: strong @1 vs flat:** v4.1 (dense rerank) showed +128% @1 signal. If cross-encoder shows similar or stronger @1 lift (say ndcg@1 > 0.025), we promote even if @10 is near the floor — matches Pareto-dominance rule (program.md §16). If ALL metrics flat like v5.1 CLAP, drop the experiment and do not queue v7.1 (L-12).

3. **Multilingual sub-bucket blindspot:** the 1.5% non-ASCII sessions won't move aggregate metrics but could silently degrade that sub-bucket. Flag for post-eval bucket analysis (`experiments/bucket_analysis.py`): if the non-English slice regresses >5% vs BM25-only, queue the mmarco multilingual checkpoint as v7.2.

4. **Cache corruption on crash mid-write:** use `os.replace(tmp)` atomic swap exactly as `dense_precomputed._save_query_cache` does. On MPS, long runs occasionally OOM; an atomic cache survives.

5. **No data leakage introduced:** checkpoint is frozen, no fine-tuning, doc strings come from `MusicCatalogDB` (item-level, split-independent). Popularity counts / user embeddings are NOT features here — strictly (query, doc) text pair. No train/dev overlap risk. Passes `feedback_no_data_leakage.md` checklist.

6. **Interaction with v6 (LightGBM):** v6 uses BM25 features directly. v7 is complementary — if both promote, queue a v13 ensemble (average v6's LGBM score and v7's CE score normalized to [0, 1], then sort). Do NOT stack them into a single pipeline until both land independently.

---

## 9. Expected gain

Per the researcher synthesis v1 portfolio and plan §6:
- **+0.015 to +0.025 on nDCG@10** over parent champion (tid=010, 0.0782 → ~0.08–0.10).
- Lower bound (+0.015) = L-6-v2 on an unseen domain; upper bound (+0.025) = if music metadata happens to be close enough to passage text for MS-MARCO supervision to transfer cleanly. Empirical anchor: Thakur et al. 2021 (BEIR) show +0.04–0.08 nDCG@10 gains from MS-MARCO cross-encoders on out-of-domain retrieval when first-stage is BM25, which is a strong prior for a positive effect even under domain shift.

Promotion decision:
- +0.015 ≥ progress-scaled threshold 0.01475 → promote.
- Below 0.015 but strict-Pareto improves on ≥7/10 secondary metrics → retro-promote under program.md §16 (same rule that promoted v5).
- All-flat (like v5.1 CLAP) → reject, no v7.1 L-12 queued.

---

## 10. Citations

1. **Nogueira & Cho 2019 — "Passage Re-ranking with BERT"**. arXiv:1901.04085. Foundational MonoBERT pattern: BM25 top-K → BERT cross-encoder logit → re-sort. Gains of +0.05–0.10 MRR@10 over BM25 on MS MARCO passage. Our v7 is a parameter-efficient MiniLM distillation of this recipe.
2. **Thakur, Reimers, Rücklé, Srivastava, Gurevych 2021 — "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models"**. NeurIPS Datasets. arXiv:2104.08663. Documents that retrieve-then-rerank with MS-MARCO-pretrained cross-encoders (including MiniLM-L-6) generalizes positively to 18 OOD datasets; directly supports the domain-gap risk being survivable (§8.1).

---

## 11. Open items (for the experimenter, not blocking claim)

- Decide whether to persist cache as pickle of tuple-keyed dict (simple) or SQLite (more robust to partial writes). Start with pickle; switch only if >2 GB.
- Confirm `transformers` minimum version supports `AutoModelForSequenceClassification` for this checkpoint on MPS eager attention (should be fine with the version already pinned by Qwen3-Embedding).
- If batch_size=32 underutilizes MPS, bump to 64 in a follow-up; first pass should land the experiment.
