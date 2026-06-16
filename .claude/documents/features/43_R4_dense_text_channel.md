# R4 — Dense-Text Channel

> Phase-1 retrieval channel. The **semantic** counterpart to R3's lexical BM25: embeds the
> constructed `Query` and the (enriched) catalog docs into one vector space and ranks by cosine.
> One of the union inputs to R7 fusion. Grounded in plan §7.2.2 (dense bi-encoder; BGE/E5/GTE
> ablation; `query:`/`passage:` prefixes; 512-token cap), §3 insight 2 (brute-force cosine over the
> full ~47k catalog = one matmul, no ANN), and §8 (context-length discipline). See `000_INDEX.md`.

## 1. Purpose
A single dense bi-encoder retrieval channel: score the F2 `Query` against the full `all_tracks`
catalog by L2-normalized cosine (one matmul over ~47k rows), and return the top-`topk_internal`
**canonical** `track_id`s. It implements the F2 `RetrievalChannel` contract; it does not fuse and it
does not rerank — it only ranks one channel's view of relevance for R7 to fuse.

## 2. Interface / contract
Lives in `mcrs/retrieval/dense_text.py`. Implements F2 `RetrievalChannel` (`11_F2_…`):

```python
class DenseTextChannel:                       # F2 RetrievalChannel
    label: str                                # unique, e.g. "dense_bge" (config-set)

    def __init__(
        self,
        catalog: "Catalog",                   # F1 — id space + id_to_metadata(enriched=True)
        embeddings: "TrackEmbeddings | None",  # F1 — provided-modality source (ablation), else None
        cfg: "DenseTextConfig",               # F2 config slice (§9)
    ) -> None: ...

    # F2 surface — queries are Query.text (or Query.per_channel[label]); returns CANONICAL ids.
    def batch_text_to_item_retrieval(
        self, queries: list[str], topk: int,
        batch_context: list[dict] | None = None, user_ids: list[str] | None = None,
    ) -> list[list[str]]: ...
```

**Wiring:** R1/R2 build the `Query`; R7 hands this channel either `Query.text` or, if a `query_key`
is configured, `Query.per_channel[label]` (R7 owns that resolution — this channel just receives the
already-resolved strings). The channel returns `list[list[track_id]]` (one ranked list per query,
length ≤ `topk`), every id passed through F1 `canonical_track_id`. R7 fuses this list with R3/R5/R6
by weighted RRF (`46_R7_…`); `topk` here is the channel's `topk_internal` (≥ `fusion_K`, P0-sized).
`batch_context`/`user_ids` are accepted for signature parity and **ignored** in the default
(query-only) channel — they are used only by the optional user-aware multimodal variant (§4.6).

**Two backends, one contract** (config-selected, output-identical shape):
- **`source: "encoded"`** — a fresh text encoder (default BGE-large-en-v1.5) embeds the enriched
  catalog docs (A1) once offline + caches; encodes the query online. Plan §7.2.2 primary path.
- **`source: "provided"`** — reuse a precomputed F1 modality matrix (`TrackEmbeddings.matrix(modality)`,
  e.g. `metadata-qwen3_embedding_0.6b`) as the catalog side; query encoded by the matching encoder.

## 3. Dependencies
- **Modules:** F1 (`Catalog`: `track_ids`, `id_to_index`/`index_to_id`, `id_to_metadata(tid, enriched=True)`,
  `canonical_track_id`; `TrackEmbeddings.matrix(modality)` for the `provided` backend), F2
  (`RetrievalChannel`, `Query`, config), F3 (`recall_at_k` for the gate), A1 (enriched doc corpus +
  the offline catalog-embedding artifact for the `encoded` backend), R1 (`Query.text`), R7 (consumer),
  P0 (sets `topk_internal`, the context-length cap, the cold/warm split, the encoder short-list).
- **Models / libs:** `sentence-transformers` (or `transformers`) for the encoder; `numpy` for the
  matmul + top-k. GPU only to *embed* (catalog once offline; queries online, batched + cached); the
  cosine matmul itself is CPU-fine on a 47k×1024 matrix (plan §3 insight 2, §8 footprint).
- **Data:** A1 enriched-doc corpus (read via `Catalog.id_to_metadata(enriched=True)` — **the channel
  never builds doc text itself**, per the Phase-1 shared contract) and/or the F1 track-embedding
  modalities. No conversations/gold (causal: the channel sees only the `Query` string).
- **Config:** `retrieval.channels[]` entry (`label`, `type: dense_text`, `weight`, `topk_internal`,
  `query_key?`, `extra`) + the §9 `DenseTextConfig` block.

## 4. Design & logic

### 4.1 Scoring (plan §3 insight 2 — brute force, no ANN)
- The catalog matrix `M` is `(n_tracks ≈ 47k, dim)`, **row-aligned to `Catalog.id_to_index`** (the
  load-bearing F1 guarantee). Query batch `Q` is `(b, dim)`. Both L2-normalized ⇒ cosine = dot:
  `scores = Q @ M.T` → `(b, n_tracks)`. Top-`topk` per row via `np.argpartition(-row, topk)[:topk]`
  then `argsort` of that slice (the salvage pattern), mapping index→`track_id` via `index_to_id`.
- 47k×1024 fp32 ≈ 190 MB; one matmul per batch is milliseconds. **No FAISS/ANN.** The §8 / plan §11
  contingency: *if* P0 ever confirms a true ~1M catalog, swap the matmul for an ANN index behind the
  same `batch_text_to_item_retrieval` surface — F1's contract and this channel's interface are
  unchanged. Default = brute force (verified 47,071, not 1M).

### 4.2 Catalog-side row alignment (THE guard)
- The matrix MUST be ordered by `Catalog.index_to_id`, not by the embedding file's native row order.
  - **`encoded` backend:** A1 embeds docs in `index_to_id` order (or returns `{track_id: vec}` that
    this channel re-stacks via `id_to_index`); init asserts every catalog id has exactly one row and
    `M.shape[0] == len(catalog)`.
  - **`provided` backend:** `TrackEmbeddings.matrix(modality)` is already row-aligned to
    `id_to_index` (F1 §4.3 guarantee). Init still asserts `M.shape[0] == len(catalog)` and
    spot-checks `M[id_to_index[tid]] == TrackEmbeddings.vector(tid, modality)` for a sample.
- Salvage `dense_local`/`dense_precomputed` carry their **own** `track_ids` list inside the pickle
  and zip it to rows — on port this is replaced by the F1 id space: build/verify `M` *from*
  `id_to_index` so a channel id can never point at the wrong track. This is the row-alignment guard
  that F1 §6.4 and §8 demand; a misalignment silently scores the wrong track and is invisible to
  recall numbers until audited. See §8.

### 4.3 Encoder choice = an ablation (plan §7.2.2, §7 E2)
- **Default `encoder: BAAI/bge-large-en-v1.5`** (English dialogues). **Alternatives** (config swap,
  no code change): `BAAI/bge-m3` (if P0 finds multilingual turns), `intfloat/e5-large-v2`,
  `thenlper/gte-large`. P0/§7-E2 picks the winner on dev recall@K; the chosen `encoder` + `revision`
  lock into `config/<exp>.yaml` (train==serve) and `model_revisions{}` (D1 records the hash).
- **Prefix discipline (asymmetric retrieval):** each encoder family wants the right query/passage
  prefix or symmetric cosine breaks. Config carries `query_prefix` / `doc_prefix`:
  - E5: `query_prefix="query: "`, `doc_prefix="passage: "` (mandatory — E5 is asymmetric).
  - BGE-large-en-v1.5: retrieval `query_prefix` = the model's recommended instruction
    (`"Represent this sentence for searching relevant passages: "`), `doc_prefix=""`.
  - BGE-M3 / GTE: no prefix (`""`/`""`).
  The **same** prefix is used to embed docs offline (A1) and queries online here — a prefix
  train/serve mismatch is a silent recall killer (the salvage `instruct`/`instruct_label` knob exists
  for exactly this; ported as `query_prefix` + `prefix_label` so different strategies cache to
  different files). Surface a VISIBLE log of the active `(encoder, query_prefix, doc_prefix)` at init.

### 4.4 Context-length cap (plan §8)
- Encoders cap at **512 tokens**; track docs are short (safe). The **dialogue query grows with
  turns** — R1 already applies the §8 recency policy (recent turns verbatim, older compressed, goal
  always kept, never truncate the latest utterance). This channel sets the tokenizer
  `max_length = query_max_len` (config, default 512) with `truncation=True`; if R1's query still
  exceeds the cap, truncation is **left**-sided so the most-recent intent at the tail survives (the
  salvage multimodal `truncation_side="left"` lesson). Log a counter of queries that hit the cap (a
  P0/§8 signal that R1's compression needs tightening, not a silent drop).

### 4.5 Normalization & determinism
- L2-normalize **catalog rows once at load** (`norms = max(‖·‖, 1e-9)`) and **query rows per batch**
  → cosine = dot (salvage pattern, kept). Store `M` as `float32`.
- **Determinism:** encoding under `torch.no_grad()` + `model.eval()`, fixed `seed`, fixed batch size,
  fp32 query output (downcast fp16 weights' output to fp32 before the matmul so scores are
  reproducible across CPU/GPU). Ties in the top-k sort broken by ascending index (stable) so repeated
  runs yield byte-identical id lists. The cosine matmul is order-deterministic for a fixed `M`.
- **Query cache** (ported, throttled): cache `query_text → embedding` keyed by `(encoder, prefix_label)`;
  persist at most every `cache_save_every` new entries + an `atexit` flush (the salvage Drive
  write-amplification fix). Cache is a speed optimization only — it never changes results.

### 4.6 Provided modalities vs. a fresh text encoder (both gated by recall)
- F1 exposes **6 provided modalities**: `metadata-qwen3_embedding_0.6b`, `lyrics-qwen3_embedding_0.6b`,
  `attributes-qwen3_embedding_0.6b`, `audio-laion_clap`, `image-siglip2`, `cf-bpr`. For a **text**
  dense channel the candidate catalog sources are the three **qwen3** text modalities (default
  consideration: `metadata-qwen3`, which the salvage `dense_precomputed` used) — query-encoded by the
  matching `Qwen/Qwen3-Embedding-0.6B` (last-token pooling, left padding — ported verbatim from
  salvage). `audio-laion_clap`/`image-siglip2`/`cf-bpr` are **not** this channel's concern: CLAP is a
  separate R6 audio channel, `cf-bpr` is R5's CF channel; SigLIP image is not a text source.
- **Decision is an ablation, not a default-by-fiat.** Treat the *freshly-encoded BGE-large doc/query*
  path (`source: "encoded"`) as the **default** dense source (plan §7.2.2 primary), and the
  *provided qwen3 modality* path (`source: "provided"`) as the **ablation alternative** (cheaper — no
  catalog embedding pass — but fixed-encoder, fixed-doc-text). P0/§7-E2 keeps whichever gives higher
  dev recall@K and unique recall; both are wired behind this one channel so the sweep is a config flip.
- **User-aware variant (optional, off):** the salvage `dense_multimodal_local` injects the user CF
  vector into the query embedding (per-user query). It is a distinct trained model; if ever revived it
  ports as a `source: "multimodal"` backend that *uses* `user_ids` (cache key = `(query, user_id)`).
  Default off — it is gated separately under R5/R6, not the plain text channel.

### 4.7 Causal / no-leak constraints
- The channel sees only the `Query` string (built causally by R1 from utterances ≤ t) and the static
  catalog. No gold, no future turns, no `thought` — those never reach the `Query` (F1/F2 enforce it
  upstream). The catalog embedding is content-only (doc text / provided modality), independent of any
  session, so there is no train/serve skew in the index. `batch_context`/`user_ids` ignored by
  default = no accidental personalization leak in the plain channel.

## 5. Reuse
- **Port + extend** `salvage/mcrs/retrieval_modules/dense_local.py` (fresh-encoder catalog pickle +
  sentence-transformers query encoding + matmul top-k) and `dense_precomputed.py` (provided-modality
  catalog + Qwen3-Embedding last-token/left-pad query encoding). Keep: the matmul/top-k path, the
  L2-normalize-on-load, the shared-encoder singleton, the throttled `atexit` query cache.
- **Pristine reference:** `music-crs-baselines/.../retrieval_modules/bert.py` — the simpler dense
  baseline; consult for the canonical bi-encoder shape, but start from salvage (caching + prefix +
  multi-instance sharing already solved).
- **Changes on port (the F2/F1 adaptation):** (1) drop the in-pickle `track_ids` zip — build/verify
  `M` *from* F1 `id_to_index` (§4.2 guard); (2) read doc text via `Catalog.id_to_metadata(enriched=True)`
  instead of the salvage doc builder; (3) replace `instruct`/`instruct_label` with config
  `query_prefix`/`doc_prefix`/`prefix_label`; (4) canonicalize every returned id via
  `canonical_track_id`; (5) match the exact F2 `batch_text_to_item_retrieval(queries, topk,
  batch_context, user_ids)` signature; (6) merge `dense_local` + `dense_precomputed` behind one
  `source` switch. **Do not** carry over salvage's artist-mean imputation as default — that was a
  qwen3-empty-row workaround; only enable it for the `provided` backend if P0 finds empty modality rows.
- **`dense_multimodal_local.py`** is **referenced, not ported now** — kept as the blueprint for the
  gated `source:"multimodal"` user-aware variant (§4.6) under R5/R6.

## 6. Eval & acceptance gate
**Channel-own metric (F3 `recall_at_k`), measured on dev via R1 queries:**
1. **Primary:** `recall@{50, 100, 200, 500}` reported **overall + cold + warm** (the §6 segment split
   from F1). The channel must clear P0's per-channel dense floor and not regress vs. the chosen-encoder
   baseline; the encoder ablation (E2) keeps the highest-recall encoder.
2. **Unique recall:** golds this channel surfaces that **no other kept channel** (R3/R5/R6) finds at
   the same K — the real fusion-value test (a channel with high recall but zero unique recall is RRF
   noise, plan §7.3 req #2). Logged per encoder/source so R7's weight/keep decision is evidence-based.
3. **Source A/B:** `encoded` (BGE) vs `provided` (qwen3 metadata) on the same dev queries — recall@K +
   unique recall — to justify which is default vs ablation.
- Numbers logged to `reports/experiments.md` + the recall-ceiling table (P0). The channel is "done"
  only when its recall@K + unique-recall are recorded and the encoder/source are locked in config.
  (R7 owns the *fused* gate recall@20 ≥ 0.75 / recall@200 ≥ 0.90 — not this module.)

## 7. Tests
- **Unit / scoring:** on a tiny fixture catalog, `batch_text_to_item_retrieval` returns ids sorted by
  descending cosine; a query identical to a doc ranks that doc #1; `topk` truncates correctly;
  `topk ≥ n_tracks` returns the full ranking.
- **Row-alignment (the guard):** `M[id_to_index[tid]]` equals the embedding for `tid` for a sampled
  set; shuffling the source file's row order does **not** change results (because `M` is built from
  `id_to_index`); `M.shape[0] == len(catalog)`; a missing/duplicate catalog row raises at init.
- **Canonical ids (plan §7.3 req #1):** every returned id ∈ `Catalog.track_ids`; a raw/prefixed id
  fixture is canonicalized before return.
- **Prefix discipline:** with E5 config, query/doc get `query: `/`passage: `; with BGE the search
  instruction prefix; a prefix-strategy change writes to a different query-cache file (no aliasing).
- **Context cap:** an over-length query is left-truncated to `query_max_len` (tail/latest-intent
  preserved) and the cap-hit counter increments.
- **Normalization / determinism:** catalog + query rows are unit-norm (±1e-5); two runs with the same
  seed/batch give byte-identical id lists; tie ids broken by ascending index.
- **Source parity:** `encoded` and `provided` backends both satisfy the F2 Protocol and the
  `⊆ catalog` / shape asserts; the query cache never alters output (cache-hit run == cold run).
- **Wiring:** R7 can construct this channel from a `retrieval.channels[]` entry and fuse its output
  with a stub BM25 channel with no shape change.

## 8. Failure modes & guards
- **Embedding/id row misalignment** (channel scores the wrong track; invisible to recall totals) →
  build `M` from `id_to_index`, never trust file order; assert shape + spot-check vectors at init
  (§4.2). The single highest-severity failure for this module.
- **Prefix mismatch train↔serve** (docs embedded without a prefix, queries with one, or E5 used
  symmetric) → cosine collapses, recall craters silently. Guard: one config-driven prefix used for
  both doc (A1) and query embedding; visible init log; prefix baked into the cache key.
- **Wrong dim / empty provided rows** (qwen3 empty embeddings, or mixing a 768-dim encoder query with
  a 1024-dim catalog matrix) → matmul error or garbage. Guard: assert `Q.dim == M.dim`; for `provided`,
  detect empty rows and either impute (artist-mean, salvage path) or fail loudly per config.
- **Unnormalized vectors** → dot ≠ cosine, popular/long docs dominate. Guard: L2-normalize on load +
  per batch; assert unit norm in tests.
- **Silent over-length truncation of the latest utterance** → loses active intent. Guard: left-side
  truncation + cap-hit counter; R1 keeps the latest turn verbatim (§8).
- **Noise channel** (high recall, no unique recall) → R7 down-/zero-weights it via the §6.2 unique-recall
  evidence; don't ship a channel that only duplicates BM25.
- **Catalog-size surprise (47k vs 1M)** → brute force too slow. Guard: P0 asserts the count; ANN swap
  behind the same interface is the documented contingency, not a silent default.
- **GPU OOM on a 4B encoder** → not applicable to BGE-large/E5-large (≤0.5B) at fp16; if a larger
  encoder is ablated, the salvage `resolve_st_dtype` fp16-on-CUDA + sub-batching path bounds memory.

## 9. Config knobs (types validated by the F2 loader; values from P0/§7-E2)
Per-channel `retrieval.channels[]` entry: `label` (e.g. `"dense_bge"`), `type: "dense_text"`,
`weight` (R7 fusion), `topk_internal` (≥ `fusion_K`, P0-sized 300–500), `query_key?` (optional
`Query.per_channel` override). Plus the `extra`/`DenseTextConfig` block this module reads:
- `dense.source` — `"encoded"` (default) | `"provided"` | `"multimodal"` (gated, off).
- `dense.encoder` — default `"BAAI/bge-large-en-v1.5"`; alts `bge-m3` / `intfloat/e5-large-v2` /
  `thenlper/gte-large`; for `provided`, the matching query encoder (`Qwen/Qwen3-Embedding-0.6B`).
- `dense.encoder_revision` — pinned commit hash (recorded by D1 in `model_revisions{}`).
- `dense.modality` — for `source:"provided"`: which F1 modality matrix (default `"metadata-qwen3_embedding_0.6b"`).
- `dense.query_prefix` / `dense.doc_prefix` / `dense.prefix_label` — asymmetric-retrieval prefixes (§4.3).
- `dense.query_max_len` — tokenizer cap (default `512`, plan §8); `dense.truncation_side` (default `"left"`).
- `dense.normalize` (default `true`), `dense.batch_size` (default `32`), `dense.dtype` (default `"auto"`).
- `dense.cache_dir`, `dense.cache_save_every` (default `2000`) — query-embedding cache throttle.
- `dense.enriched_docs` (default `true`) — read A1-enriched doc text vs raw `corpus_types`.
- `dense.impute_empty` (default `false`) — provided-modality empty-row imputation (salvage artist-mean).
- inherited: `seed`, `segment.cold_threshold` (for the cold/warm recall split in §6).

## 10. Definition of Done & review checklist
- [ ] `DenseTextChannel` implements F2 `RetrievalChannel`; `encoded` + `provided` backends behind one
      `source` switch; signature exactly `batch_text_to_item_retrieval(queries, topk, batch_context, user_ids)`.
- [ ] Catalog matrix built/verified **from** F1 `id_to_index`; row-alignment asserts + spot-check pass.
- [ ] Every returned id `canonical_track_id`-normalized and `⊆ Catalog.track_ids`.
- [ ] Query/doc prefixes config-driven, identical across train/serve, baked into the cache key; init
      logs the active `(encoder, query_prefix, doc_prefix, source)`.
- [ ] Doc text read via `Catalog.id_to_metadata(enriched=True)` — channel builds no doc text itself.
- [ ] L2-normalization (load + per-batch) + determinism tests green; query cache never alters output.
- [ ] `recall@{50,100,200,500}` overall + cold/warm + unique recall logged for the chosen encoder;
      `encoded` vs `provided` A/B logged; encoder/source locked in `config/<exp>.yaml`.
- [ ] Context-cap (left-truncation + cap-hit counter) implemented; latest utterance preserved.
- [ ] Code review approved; no `Any` in public signature; no salvage in-pickle `track_ids` zip left.

## 11. Build order & dependencies
**Built in Phase 1 after A1 (enriched docs + catalog embedding for the `encoded` backend) and R1
(`Query`), in parallel with R3.** Depends on: F1 (id space, `id_to_metadata`, `TrackEmbeddings`),
F2 (contracts/config), F3 (`recall_at_k`), A1, R1, P0 (`topk_internal`, context cap, cold/warm split,
encoder short-list). **Blocks:** R7 (fusion consumes this channel's `list[list[track_id]]`), and
transitively K1/K2 (rerank the fused pool) and the first end-to-end submission
(`…→ R4 ⊕ R3 → R7 → L1 → responder → D1`, plan §17). On the critical path via R7.
