# R8 — ColBERT Late-Interaction Channel (HOT LEVER, parked)

> Phase-1/§12 retrieval channel — a **token-level semantic** retriever that replaces the
> single-vector R4 dense channel's *current-intent* role. Encodes the focused conversational
> query and the A1-**enriched** catalog docs into **one vector per token** (no pooling) and ranks
> by **MaxSim** (sum of per-query-token max cosine). One union input to R7 fusion.
>
> **Status: PARKED hot lever, not in the active plan.** Current plan stays BM25 (R3) + dense (R4)
> → R7 RRF → K2 LightGBM. This doc is the ready-to-build design for when we attack the recall
> ceiling. Motivation: enriched BM25 tops out ~0.36 recall@100 (P0/A1, 2026-06-16) — far below the
> recall@K ≥ 0.90 the nDCG-0.55 target needs; late interaction is the lever to break it. Grounded
> in plan §9.2 (ColBERT/cross-encoder rerank menu), §7.2.2 (dense retrieval), §8 (length discipline),
> Khattab & Zaharia 2020 (ColBERT), Santhanam 2022 (ColBERTv2/PLAID), Chaffin 2025 (PyLate).
> See `000_INDEX.md`. Memory: `project_colbert_hot_lever.md`.

## 1. Purpose
A single late-interaction retrieval channel: score a **focused** F2 `Query` (recent turn + goal +
cultural attributes, NOT the full session) against the full `all_tracks` catalog by **MaxSim** over
token embeddings of the A1-enriched docs, and return the top-`topk_internal` **canonical**
`track_id`s. Implements the F2 `RetrievalChannel` contract; it does not fuse and does not rerank.
It is the **semantic current-intent** channel — it **replaces** R4's query-dense role (the two are
redundant; we never run both as query channels — see §4.7).

## 2. Interface / contract
Lives in `mcrs/retrieval/colbert_channel.py`. Implements F2 `RetrievalChannel` (`11_F2_…`):

```python
class ColBERTChannel:                         # F2 RetrievalChannel
    label: str                                # unique, e.g. "colbert" (config-set)

    def __init__(
        self,
        catalog: "Catalog",                   # F1 — id space + id_to_metadata(enriched=True)
        cfg: "ColBERTConfig",                 # F2 config slice (§9)
    ) -> None: ...

    # F2 surface — queries are Query.per_channel[label] (focused), else Query.text; CANONICAL ids out.
    def batch_text_to_item_retrieval(
        self, queries: list[str], topk: int,
        batch_context: list[dict] | None = None, user_ids: list[str] | None = None,
    ) -> list[list[str]]: ...
```

**Wiring:** R1/R2 build the `Query`; this channel should be fed a **per-channel** query string
(`query_key` configured → `Query.per_channel["colbert"]`) that is tighter than the dense/BM25 query
(§4.4). Returns `list[list[track_id]]` (one ranked list per query, length ≤ `topk`), every id passed
through F1 `canonical_track_id`. R7 fuses with R3/R5/R6 by weighted RRF (`46_R7_…`); `topk` here is
`topk_internal` (≥ `fusion_K`, P0-sized). `batch_context`/`user_ids` accepted for signature parity
and **ignored** (content-only channel; personalization stays in R5 — see §4.7).

## 3. How the model works (mechanics — so we know exactly what we're wiring)

### 3.1 Token-level encoding (no pooling)
- A transformer backbone (BERT/ModernBERT) encodes query and document **independently**. Unlike a
  bi-encoder (R4), there is **no pooling** — every token keeps its own contextual vector, then a
  linear layer projects each to a low dim (`dim`, default **128**) and L2-normalizes it.
- **Special markers:** a `[Q]` token is prepended to queries, `[D]` to documents (the model learns
  asymmetric query/doc representations from the same backbone).
- **Query augmentation:** the query is padded to `query_maxlen` with `[MASK]` tokens (not discarded).
  These MASK positions still produce embeddings — a learned "query expansion" that lets the model
  match doc tokens the literal query didn't contain. This is why `query_maxlen` is a *real* knob, not
  just a truncation limit: larger = more expansion capacity (and more cost/noise).
- **Punctuation masking** (`mask_punctuation`, default true): document punctuation token embeddings
  are dropped, so they can't absorb MaxSim mass.

### 3.2 MaxSim scoring
```
S(Q, D) = Σ_{i=1..|Q|}  max_{j=1..|D|}  (q_i · d_j)        # q_i, d_j are L2-normalized
```
Each **query** token takes its single best-matching **document** token (max cosine); those maxes are
**summed** over query tokens. Two consequences that drive our design:
1. **Aspect independence** — "workout", "90s", "grunge" each find their own best doc token, so a
   multi-aspect query isn't blurred into one centroid (the win over R4's pooled vector).
2. **Additive, non-negative** — every query token only *adds* to the score (max of cosines, ≥ ~0
   after the doc side). So junk/stale/long-context query tokens inflate scores indiscriminately and
   blur ranking → **keep the query short and focused** (§4.4). This is the opposite failure mode to a
   bi-encoder and the reason ColBERT gets its own tight query, not the kitchen-sink dense query.

### 3.3 Retrieval at serve time (two viable paths at our 47k scale)
- **PLAID (library default):** offline, cluster all doc-token embeddings into centroids (k-means);
  compress residuals to `nbits` (default 2). At query time: (a) find candidate docs whose tokens fall
  in probed centroid **cells** (`ncells`), (b) filter by `centroid_score_threshold`, (c) take `ndocs`
  candidates, (d) decompress and compute **exact MaxSim** to rank. Approximate but scales to millions.
- **Brute-force MaxSim (recommended first at 47k):** mirror R4's "one matmul, no ANN" insight (plan §3
  insight 2). 47k docs × ~`doc_maxlen` tokens × 128-dim fp16 token matrix is a few GB; an exact
  batched MaxSim per query is feasible on a GPU and removes PLAID's approximation + tuning surface.
  Start exact (correctness, no recall lost to ANN), only adopt PLAID if memory/latency forces it.

## 4. Design & logic

### 4.1 Channel role — replaces R4 query-dense (not additive to it)
ColBERT and R4 both do *semantic match on the current query*; running both as query channels is
redundant (plan §7.3 req #2: a channel with no **unique** recall is RRF noise). So ColBERT **takes
over** the current-intent role and R4's query-dense channel is dropped (or zero-weighted) **iff** the
§6 ablation confirms ColBERT-alone ≥ dense-alone recall@K and dense adds ~no unique recall.

### 4.2 Indexes the A1-ENRICHED docs (hard constraint)
Doc text is read via `Catalog.id_to_metadata(tid, enriched=True)` — the channel **never** builds doc
text itself (Phase-1 shared contract). The A1 doc2query expansion (validated +60% BM25 recall@100,
2026-06-16) gives ColBERT more aspect-tokens to MaxSim against; the two levers compound.

### 4.3 Document-side length discipline (plan §8: truncate the DOC, preserve the query)
Some enriched docs are long (raw tag dumps + the doc2query sentences — e.g. the RHCP sample). The
high-value text is the **doc2query expansion** (after the ` | `), not the noisy raw tag list. Policy:
build the ColBERT doc as `expansion + trimmed-metadata` capped at `doc_maxlen` (default **300**),
trimming the raw tag dump first. Log a counter of docs that hit the cap (a signal to prune doc text,
not silently truncate). Truncate the **document** side, never the query side.

### 4.4 Query-side: a FOCUSED per-channel query (the key decision)
ColBERT gets a **tighter** query than dense/BM25 via `Query.per_channel["colbert"]`:
**recent utterance + goal + cultural attributes** (+ at most the last 1 turn for follow-ups like
"more upbeat"), capped under `query_maxlen`. Rationale = §3.2 (additive MaxSim punishes long queries)
**and** the model's small `query_maxlen` (default 32 in PyLate / 60 Stanford) which would otherwise
hard-truncate — and, if truncation is right-sided, chop the *latest* (most important) intent. R1
produces this focused string (recency_window/context_cap knobs already exist); never drop the latest
utterance. Dense/BM25 may still take the fuller context — `per_channel` is built for exactly this
divergence, and the ablation stays honest because each channel uses its best-suited input.

### 4.5 Pretrained-first, gate before fine-tuning
- **Phase A (default):** an **off-the-shelf** late-interaction model — `colbert-ir/colbertv2.0`,
  `answerdotai/answerai-colbert-small-v1`, or `lightonai/GTE-ModernColBERT-v1` (ModernBERT backbone,
  8k context, strong BEIR) — index enriched docs, measure recall@K via P0. **No training yet.**
- **Phase B (only if A clears its gate but needs more):** fine-tune on **Train** conversational
  query→gold pairs (contrastive in-batch + optional KD from a cross-encoder teacher; §3.x training
  knobs). The prior (ceilinged) approach invested in ColBERT fine-tuning **early** — do NOT repeat
  that; gate before sinking GPU. (`mcrs/training/colbert_*.py` + `nb/phase2_colbert_finetune.ipynb` are the training reference.)

### 4.6 Canonical ids + row/space alignment
Every returned id → F1 `canonical_track_id`, and the doc index is built over `Catalog.index_to_id`
so a ColBERT internal pid maps back to the right `track_id`. Assert index size == `len(catalog)` and
spot-check a pid→track_id round-trip at load (the same row-alignment guard R4 §4.2 demands).

### 4.7 Causal / no-leak + one-history-channel constraint
- Content-only: sees the focused `Query` string (causal, turns ≤ t) and the static enriched catalog.
  No gold, no future turns, no `thought`. `batch_context`/`user_ids` ignored ⇒ no personalization leak.
- **Hard constraint — exactly ONE history channel.** The user's "dense over history" idea already
  exists as R5 `ContentKNNChannel` (recency-pooled history embeddings → nearest tracks). We do **not**
  build a new dense-over-history channel. History/taste stays = R5 (ContentKNN + CF + same-artist are
  distinct representations, not dupes). ColBERT is current-intent only; personalization is orthogonal.

## 5. Reuse
- **Port behind the gate:** the active ColBERT channel `mcrs/retrieval/colbert_channel.py` (late-interaction
  scoring + a `strip_track_id_prefix` doc-text helper — note plan §7.3: it's a doc-text helper, **not**
  an id normalizer; canonicalize ids via F1, not this) for index build, and `mcrs/training/colbert_*.py`
  + `nb/phase2_colbert_finetune.ipynb` (Phase-B fine-tune). Prior nbs `82_colbert_conversational_retrieval`
  / `90_colbert_dev_experiments` (what was tried + where it ceilinged — recoverable from the old git
  branches; mine for the failure mode, don't trust wholesale).
- **Prefer a maintained library** over hand-rolled: **PyLate** (Sentence-Transformers-based, modular
  train/index/serve) or Stanford **ColBERT** (PLAID). Build the channel as a thin F2 adapter over the
  library so the index/search internals stay swappable.
- **Changes on port:** (1) read doc text via `Catalog.id_to_metadata(enriched=True)`; (2) build the
  index over F1 `index_to_id`, canonicalize returned ids; (3) match the exact F2 signature; (4) feed
  the focused `per_channel` query, not raw turns; (5) drop the in-pickle id zip in favor of the F1 id
  space; (6) start brute-force MaxSim at 47k, PLAID only if forced.

## 6. Eval & acceptance gate
**Channel-own metric (F3 `recall_at_k`), measured on dev via R1 focused queries:**
1. **Primary:** `recall@{50,100,200,500}` overall + cold + warm; must clear P0's per-channel floor.
2. **Replace-R4 ablation (the decision test):** ColBERT-alone vs R4-dense-alone vs both-fused —
   recall@K **and unique recall**. Replace R4 iff ColBERT-alone ≥ dense-alone and dense's unique
   recall on top of ColBERT is ~0. (If complementary, keep both; if pretrained ColBERT < dense, that's
   the signal to consider Phase-B fine-tuning.)
3. **Pretrained vs fine-tuned (Phase A vs B):** recall@K + unique recall, to justify any GPU on training.
- Numbers → `reports/experiments.md` + the P0 recall-ceiling table. "Done" when recall@K + unique
  recall are logged and the model/source are locked in config. (R7 owns the *fused* gate, not this.)

## 7. Tests
- **Unit / scoring:** tiny fixture catalog — a query identical to a doc's expansion ranks that doc #1;
  `topk` truncates; multi-aspect query surfaces a doc matching on a non-title aspect.
- **MaxSim correctness:** hand-computed MaxSim on a 2-doc/3-token fixture equals the channel's score
  ordering; L2-normalized token embeddings (‖·‖ ≈ 1).
- **Canonical ids (plan §7.3 req #1):** every returned id ∈ `Catalog.track_ids`; prefixed-id fixture
  canonicalized before return; pid→track_id round-trip correct after a shuffled index build.
- **Query focus / cap:** the per-channel query is the focused string (not full session); an
  over-length query truncates without losing the latest utterance; `query_maxlen` respected.
- **Doc cap:** an over-length enriched doc is trimmed to `doc_maxlen` keeping the doc2query expansion;
  cap-hit counter increments.
- **No-leak:** channel output is invariant to `batch_context`/`user_ids` (content-only).
- **Determinism:** fixed seed/batch → byte-identical id lists; ties broken by ascending index.
- **Wiring:** R7 constructs the channel from a `retrieval.channels[]` entry and fuses with a stub BM25.

## 8. Failure modes & guards
- **Long/kitchen-sink query** → additive MaxSim inflates scores, recall + precision drop. Guard:
  focused `per_channel` query (§4.4); cap-hit counter; latest utterance preserved.
- **Doc truncation drops the doc2query expansion** → loses the very text that lifts recall. Guard:
  order doc as expansion-first, trim raw tags first (§4.3).
- **pid↔track_id / row misalignment** → scores the wrong track, invisible to recall totals. Guard:
  build index from `index_to_id`; assert size; round-trip spot-check (§4.6).
- **PLAID ANN under-retrieval** (ncells/threshold too tight → golds never reach exact rerank) → use
  brute-force MaxSim at 47k, or widen `ncells`/lower `centroid_score_threshold`/raise `ndocs` and
  re-measure recall. Never let ANN silently cap recall.
- **query_maxlen too small** → right-truncates the latest intent. Guard: left-truncation + a
  query_maxlen large enough for the focused query (measure token lengths in P0/EDA).
- **Two history channels** → double-counts taste in RRF. Guard: ColBERT is content-only; history =
  R5 only (§4.7).
- **Redundant with R4** → RRF noise. Guard: the §6 replace-ablation; don't ship two query channels.
- **Running Phase-B fine-tune before the pretrained gate** → repeats the prior ceiling. Guard: §4.5.
- **GPU/memory on the token matrix** → fp16 token embeddings + sub-batching; or PLAID `nbits=2`.

## 9. Config knobs (types validated by the F2 loader; values from P0/§9-ablation)
Per-channel `retrieval.channels[]` entry: `label` (`"colbert"`), `type: "colbert"`, `weight` (R7),
`topk_internal` (≥ `fusion_K`, 300–500), `query_key` (e.g. `"colbert"` → focused `per_channel` query).
Plus the `ColBERTConfig` block this module reads — **the full parameter surface:**

**Model / encoding**
- `colbert.model` — backbone/checkpoint; default `"colbert-ir/colbertv2.0"`; alts
  `"answerdotai/answerai-colbert-small-v1"`, `"lightonai/GTE-ModernColBERT-v1"`.
- `colbert.model_revision` — pinned commit hash (D1 records it).
- `colbert.dim` — projection dim per token; default **128**.
- `colbert.similarity` — `"cosine"` (L2-normalize; default) | `"l2"`.
- `colbert.query_maxlen` — query token cap incl. `[MASK]` augmentation; default **32** (PyLate) /
  **60** (Stanford). Tune to the focused query length.
- `colbert.doc_maxlen` — document token cap; default **300** (PyLate) / **120** (Stanford).
- `colbert.mask_punctuation` — drop doc punctuation embeddings; default **true**.
- `colbert.attend_to_mask_tokens` — default **false**.
- `colbert.query_prefix` / `colbert.doc_prefix` — the `[Q]`/`[D]` markers (model-defined; rarely changed).
- `colbert.bsize` — encode batch size; default **32**.
- `colbert.dtype` — `"auto"`/`fp16`/`bf16`.

**Indexing (PLAID; only if not brute-force)**
- `colbert.index_backend` — `"bruteforce"` (default at 47k) | `"plaid"`.
- `colbert.nbits` — residual quantization bits; default **2**.
- `colbert.kmeans_niters` — centroid k-means iters; default **4**.
- `colbert.pool_factor` — hierarchical token-pool compression (PyLate); default **1** (off).
- `colbert.index_name` / `colbert.index_root` / `colbert.override`.
- `colbert.nranks` / `colbert.gpus` — indexing parallelism.

**Search / retrieval (PLAID; scale with k)**
- `colbert.k` — top-k (= `topk_internal`).
- `colbert.ncells` — centroid cells probed; ColBERTv2 default scales with k (≈1/2/4 for k≤10/≤100/else).
- `colbert.centroid_score_threshold` — cell prune threshold; default ≈ 0.5/0.45/0.4 by k tier.
- `colbert.ndocs` — candidates taken to exact MaxSim; default ≈ 256/1024/4096 by k tier.
  *(Verify exact defaults against the installed library version; widen if recall caps.)*

**Training (Phase B only — `mcrs/training/colbert_*.py` + `nb/phase2_colbert_finetune.ipynb`)**
- `colbert.train.loss` — `"contrastive"` (in-batch negatives) | `"kldiv"` (KD from a cross-encoder teacher).
- `colbert.train.teacher` — e.g. `BAAI/bge-reranker-v2-*` for distillation scores.
- `colbert.train.nway` — negatives per query (in-batch + hard).
- `colbert.train.use_ib_negatives` — default **true**.
- `colbert.train.lr`, `warmup`, `maxsteps`/`epochs`, `bsize`, `accumsteps`.
- `colbert.train.distillation_alpha`, `temperature` (KD).
- `colbert.train.gradient_checkpointing`, `fp16`/`bf16`, GradCache (memory-bounded large effective batch).

- inherited: `seed`, `segment.cold_threshold` (cold/warm recall split, §6).

## 10. Definition of Done & review checklist
- [ ] `ColBERTChannel` implements F2 `RetrievalChannel`; exact signature
      `batch_text_to_item_retrieval(queries, topk, batch_context, user_ids)`.
- [ ] Index built over A1-**enriched** docs (`id_to_metadata(enriched=True)`), from F1 `index_to_id`;
      size assert + pid→track_id round-trip pass.
- [ ] Every returned id `canonical_track_id`-normalized and ⊆ `Catalog.track_ids`.
- [ ] Fed a **focused** `per_channel` query (recent turn + goal + cultural); `query_maxlen`/`doc_maxlen`
      caps honored (doc trimmed expansion-first); cap-hit counters logged.
- [ ] Content-only (no `batch_context`/`user_ids` use); no second history channel introduced (R5 owns history).
- [ ] Pretrained Phase-A measured first; Phase-B fine-tune only after the gate; model/source/revision in `config/<exp>.yaml`.
- [ ] `recall@{50,100,200,500}` overall + cold/warm + unique recall logged; replace-R4 ablation logged.
- [ ] Determinism tests green; brute-force vs PLAID parity (if PLAID) within recall tolerance.
- [ ] Code review approved; thin adapter over PyLate/ColBERT; no in-pickle id zip left.

## 11. Build order & dependencies
**Parked — built in §12 "advanced levers" (plan Day 8–10), after A1 (enriched docs) and R1 (focused
`Query`), once the P0 recall ceiling shows lexical+dense fusion is short of recall@K ≥ 0.90.** Depends
on: F1 (id space, `id_to_metadata(enriched=True)`), F2 (contracts/config), F3 (`recall_at_k`), A1,
R1 (`Query.per_channel`), P0 (`topk_internal`, query/doc length caps, cold/warm split, model
short-list). **Relates to:** R4 (replaces its query-dense role behind the §6 ablation), R5 (the single
history channel — must NOT be duplicated), R7 (fusion consumes this channel). Off the critical path
until the recall gate forces it; the baseline submission ships without it.
