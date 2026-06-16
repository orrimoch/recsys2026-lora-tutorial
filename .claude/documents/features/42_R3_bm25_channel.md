# R3 — BM25 Channel (enriched docs)

> Phase-1 retrieval channel. A sparse lexical retriever over the **enriched** track docs (A1),
> implementing the F2 `RetrievalChannel` contract. The cheap, deterministic, no-GPU recall workhorse
> on the critical path to a first valid submission (`F1+F2+F3 → P0 → R1+R3 → R7 → L1 → responder → D1`,
> `000_INDEX.md`). Strong for explicit artist / genre / title / album intent — the lexical anchor RRF
> fuses with the semantic channels (R4/R5/R6). Grounded in PRISTINE `bm25.py` (reuse) + plan §7.2.1,
> §7.3.1. See `000_INDEX.md` for the catalogue and `11_F2_interfaces_contracts_config.md` for contracts.

## 1. Purpose
Build and serve a cached BM25 index over the canonical `all_tracks` catalog rendered as **enriched
doc text** (from A1 via `Catalog.id_to_metadata(track_id, enriched=True)`, `corpus_types`-driven),
and answer the F2 `RetrievalChannel.batch_text_to_item_retrieval` call with the top-`topk_internal`
**canonical** `track_id`s per query — the lexical input list R7 fuses. One responsibility: rank the
catalog by lexical match to the constructed query; it does **not** fuse, rerank, or build doc text.

## 2. Interface / contract
Lives in `mcrs/retrieval/bm25_channel.py`. Implements the F2 `RetrievalChannel` Protocol
(`11_F2_…` §2) verbatim — same signature shape as the salvage RRF sub-retriever, so R7 calls it like
any channel. Registered under a unique `label` (default `"bm25"`).

```python
class BM25Channel:                       # implements F2 RetrievalChannel
    label: str = "bm25"                  # unique registry key; R7 reads it as a channel_spec label

    def __init__(self, catalog: "Catalog", cfg: "BM25Config"): ...
    # cfg: corpus_types, enriched(bool), k1, b, cache_dir, label, lowercase, stopwords

    # F2 surface — the ONLY method R7/D1 calls. Returns canonical track_ids, absent ⇒ not retrieved.
    def batch_text_to_item_retrieval(
        self, queries: list[str], topk: int,
        batch_context: list[dict] | None = None, user_ids: list[str] | None = None,
    ) -> list[list[str]]: ...            # len == len(queries); each inner list ≤ topk, canonical ids

    def text_to_item_retrieval(self, query: str, topk: int) -> list[str]: ...  # single-query convenience

    def build_index(self) -> None: ...   # build + persist; idempotent (skip-if-exists unless force)
```

**Inputs.** `queries` = the constructed causal query strings (R1 builds the F2 `Query`; R7 passes
`Query.text`, or `Query.per_channel["bm25"]` if a per-channel override is configured — R3 itself never
re-derives the query from raw turns, per the Phase-1 query-flow contract in `000_INDEX.md`).
`batch_context`/`user_ids` are accepted for signature parity but **unused** by BM25 (lexical, no
personalization) — documented as a no-op, not silently dropped.

**Outputs.** `list[list[str]]`, outer aligned to `queries`, each inner list ordered by decreasing BM25
score, ≤ `topk`, every id passed through F1 `canonical_track_id` (§4.4). This is exactly the
per-channel `list[list[track_id]]` R7 consumes (`46_R7_…` §2 wiring; R7 then attaches `channel_ranks`/
`channel_scores` for label `"bm25"`).

**Wiring.** `R1/R2 → Query` → `R3.batch_text_to_item_retrieval(query_texts, topk_internal)` →
`R7.fuse(...)` → `Candidate` pool → K1/K2. R3 reads doc text from A1-enriched `Catalog`; it sits
strictly between F1/A1 (docs + id space) upstream and R7 downstream.

## 3. Dependencies
- **Modules:** F1 (`Catalog` — `id_to_metadata(enriched=True)`, `track_ids`, `index_to_id`,
  `canonical_track_id`), F2 (`RetrievalChannel` Protocol, config schema), **A1** (the enriched corpus
  that backs `id_to_metadata(enriched=True)`; R3 degrades to the raw `corpus_types` doc if A1 is
  absent), F3 (`recall_at_k` for the gate), P0 (decided `topk_internal`, cold/warm split, which
  `corpus_types` to include), R1 (`Query`), R7 (consumer).
- **Libs:** `bm25s` (the pristine backend), `numpy`, stdlib `json`/`os`. **No GPU, no model, no
  external API.** CPU-only, fast.
- **Data:** none directly on disk — the corpus is read through `Catalog` (F1 owns the `all_tracks`
  load + canonical id space). The only artifact R3 writes is its own index cache.
- **Config:** `retrieval.channels[]` entry with `type: bm25` (`label`, `weight`, `topk_internal`,
  `query_key`) + a `bm25` block (`corpus_types`, `enriched`, `k1`, `b`, `cache_dir`).

## 4. Design & logic

### 4.1 Corpus = A1-enriched docs through the F1 accessor (no doc text built here)
The index is built over **one doc per canonical `track_id`**, rendered by
`Catalog.id_to_metadata(track_id, enriched=True)` — the A1→R3 enriched-doc contract in
`000_INDEX.md` ("BM25/dense docs come from here — channels never build doc text themselves").
A1 layers its enrichment (doc2query expansions, inferred mood/genre blurbs for sparse tracks, plan
§12) **on top of** the base `corpus_types`. The base assumed-doc text is the established
`format_catalog_track_text` rendering (salvage `track_text.py`):
`"track_id: <id>, track_name: <vals>, artist_name: <vals>, album_name: <vals>, release_date: <vals>"`
with list fields `", ".join(...)`-ed and lowercased — the baseline `corpus_types`
`[track_name, artist_name, album_name, release_date]` (plan §7.2.1 adds `tags`). `enriched=True` is
the contract knob: when A1 has run, the same accessor returns that base text **plus** the appended
expansions; when A1 has not run, it returns the base text only and R3 still works (graceful
degradation — flagged in a startup log, not a crash). R3 does **not** reach into raw metadata rows
or call `format_catalog_track_text` itself; it consumes only the accessor output (one source of doc
truth, kills train/serve doc-format skew — the documented `bge_m3_ft` failure mode in `track_text.py`).

### 4.2 Index build, ordering, and id-canonicalization at the channel boundary (plan §7.3 req #1)
- Iterate `Catalog.index_to_id` (the F1 **stable** row order) so the BM25 corpus row index `i` maps to
  a fixed `track_id`. Build a private `self.doc_ids: list[str]` of those **already-canonical** ids
  (they come from `Catalog.track_ids`, the canonical universe) so retrieval maps `bm25_doc_id → track_id`
  with no per-call lookup. Build cost is one pass over ~47,071 docs (F1 §3) — seconds, CPU.
- Tokenize each doc with `bm25s.tokenize` (lowercased; configurable stopword set — default the
  `bm25s` English list, but **recorded in config** so train==serve), `BM25().index(...)`, persist via
  `retriever.save(...)` + a `doc_ids.json` sidecar (port of pristine `track_ids.json`).
- **Canonicalization at the boundary (the load-bearing F1/F2 invariant):** every id R3 emits is
  passed through F1 `canonical_track_id` before return. In practice the doc ids are already canonical
  (sourced from `Catalog.track_ids`), so this is a cheap idempotent pass + a guard — but R3 **must**
  apply it explicitly at its output so the channel boundary is self-certifying (plan §7.3 req #1; F2
  §4 "any channel deriving ids canonicalizes before returning"; R7 then re-asserts `⊆ catalog`). R3
  never leans on salvage `strip_track_id_prefix` (a doc-text helper, not an id normalizer — F1 §4.1).

### 4.3 Pull depth = `topk_internal`, P0-sized, ≥ fusion_K — NOT the salvage default 60 (plan §7.3.1)
R3 is **always** pulled to `topk_internal` before fusion, never the pristine `RRF_MODEL`/`BM25_MODEL`
default of 60 — "a gold ranked 61–500 in a channel is dropped before fusion ever sees it" (plan
§7.3.1, the silent-recall-ceiling trap). The value is **set in P0** from the per-channel
recall-at-depth probe, sized so `topk_internal ≥ fusion_K` (target **300–500**), and locked in
`retrieval.channels[<bm25>].topk_internal`. R7 passes that value as the `topk` argument; R3 caps each
inner list at it. The invariant `topk_internal ≥ fusion_K ≥ cross_encoder_K` (R7 §4 / plan §7.3.1) is
R7's to enforce, but R3 asserts the `topk` it receives is `> 60` and warns if it equals the legacy
default (defends against a config that forgot to size it).

### 4.4 Query handling, batching, determinism
- One `bm25s.tokenize([q.lower() for q in queries])` + one batched `retriever.retrieve(..., k=topk)`
  call for the whole batch (port of pristine `batch_text_to_item_retrieval`) — vectorized, no
  per-query Python loop over the index.
- **Determinism:** BM25 scoring is a pure function of (index, query) → identical ranks every run.
  Tie-breaking on equal scores must be **stable** (break ties by ascending `Catalog.id_to_index`, not
  by `bm25s` insertion accident) so the fused order downstream is reproducible (R7 determinism test +
  F1 §6 determinism gate). `seed` does not affect BM25 ranking but is recorded with the run.
- **Empty / degenerate query** (R1 produced `""` or all-stopword text → no tokens): return `[]` for
  that row (a VISIBLE warning, never a crash, never a silent full-catalog dump). RRF tolerates an
  empty channel output (contributes 0), so this fails safe.

### 4.5 Cold vs warm
BM25 is a **content/intent** channel — equally live for cold and warm users (cold users have a query
but no history/CF; plan §10 leans on BM25 + dense for cold). R3 therefore applies **no** segment
gating itself; any cold/warm re-weighting is R7's `cold_weight`/`warm_weight` knob (R7 §4), not R3's.
The eval gate (§6) still reports recall split by `TurnContext.segment` so P0/R7 can see where BM25's
unique recall lands.

### 4.6 No-leak / causal constraints
The corpus is the static `all_tracks` catalog (no per-turn, no gold, no future data) — structurally
leak-free. The only causal surface is the **query**, which R1 already built causally (≤t utterances);
R3 consumes the finished `Query` string and adds no turn data of its own. R3 must **not** be built
over `test_tracks` or any held split — `all_tracks` only (F1 §8 guard).

## 5. Reuse
- **BM25 engine — REUSE (pristine, do not rewrite):** `music-crs-baselines/mcrs/retrieval_modules/bm25.py`
  (`BM25_MODEL`) — its `bm25s` index/build/save/load + batched `batch_text_to_item_retrieval` are
  correct and already match the F2 channel signature shape. **Changes on adoption:** (1) source the
  corpus from `Catalog.id_to_metadata(enriched=True)` over `Catalog.index_to_id` instead of the
  internal `_load_corpus()`/`_stringify_metadata()` (one doc-text source — A1, not a private
  formatter); (2) emit ids through F1 `canonical_track_id`; (3) accept `topk_internal` from config and
  drop the implicit `60`; (4) accept the unused `batch_context`/`user_ids` kwargs for F2 parity;
  (5) skip-if-exists build with a `--force` override (avoid the in-place-overwrite footgun from the
  ColBERT incident in memory). **Reuse + thin adapter.**
- **Doc-format helpers — PORTED INTO A1, referenced here:** salvage `track_text.py`
  (`format_catalog_track_text`) and `bge_m3_format.py` (`format_track_text`, the enriched 5-field
  pipe form) are the canonical renderers. They belong to **A1 / F1 `id_to_metadata`** (the single doc
  source), not R3 — R3 only *consumes* their output via the accessor. Cited so the doc-text contract
  is unambiguous; R3 copies no formatter code.
- **Config shape:** clone the `bm25`/`corpus_types`/`cache_dir` keys from
  `music-crs-baselines/config/llama1b_bm25_devset.yaml` (verified in F1 §5), typed by the F2 loader.

## 6. Eval & acceptance gate
**Own metric (F3 `recall_at_k`, on dev):** **recall@{50, 100, 200, 500}** of BM25 **alone**,
reported **overall + cold + warm**. The number is the channel's standalone lexical reach; it is logged
to `reports/experiments.md` and feeds P0's `topk_internal` sizing + R7's keep/drop decision. There is
no fixed pass threshold for a single channel (R7 owns the fused recall@20 ≥ 0.75 / recall@200 ≥ 0.90
DoD) — instead R3's gate is twofold:
1. **Reproduces the baseline lexical floor:** BM25 on the **base** `corpus_types` (`enriched=False`)
   reproduces the pristine `llama1b_bm25` baseline recall on the same dev split (parity check — proves
   the port is faithful before A1 enrichment is credited).
2. **Unique-recall contribution (the R7 keep/drop signal, plan §7.3 req #2):** the count/fraction of
   dev golds BM25 surfaces in its `topk_internal` that **no other kept channel** surfaces. A channel
   with ~0 unique recall is dropped or zero-weighted in R7 (drop noise, don't rescue it). BM25 is
   expected to be a strong **explicit-intent** contributor (artist/title/genre queries) and to clear
   this comfortably; if A1 enrichment (`enriched=True`) does **not** raise recall and/or unique recall
   over `enriched=False`, that is logged as an A1 finding (enrichment didn't help BM25), not an R3 bug.

Measured via F3's harness over the dev gold set; cold/warm split by `TurnContext.segment`.

## 7. Tests
- **F2 conformance:** a `BM25Channel` instance satisfies the `RetrievalChannel` structural check;
  `batch_text_to_item_retrieval` returns `len(queries)` lists, each ≤ `topk`.
- **Id-canonicalization at the boundary (plan §7.3 req #1):** every returned id ∈ `Catalog.track_ids`;
  a doc whose raw id needs canonicalization still emits the canonical form; F3's `⊆ catalog` assert
  passes on R3 output.
- **Doc source = A1 accessor:** the indexed corpus equals
  `[Catalog.id_to_metadata(tid, enriched=True) for tid in Catalog.index_to_id]` (a monkeypatched
  accessor changes the index — proves R3 builds no doc text itself); `enriched=False` falls back to
  the base `corpus_types` text without crashing.
- **Pull depth (the §7.3.1 trap):** a fixture gold ranked at BM25 position 250 appears in the output
  iff `topk` ≥ 250; a `topk == 60` config raises the legacy-default warning.
- **Ranking correctness:** an exact artist/title query ranks the matching track at/near rank 1; a
  multi-value (list) metadata row is matchable on any of its values (list-join coverage).
- **Determinism:** two builds over the same catalog → byte-identical `doc_ids.json` order; repeated
  retrieval → identical ranks; equal-score ties broken by ascending `id_to_index` (stable).
- **Edge cases:** empty/all-stopword query → `[]` + warning (no crash, no full-catalog dump);
  `batch_context`/`user_ids` passed → ignored, output unchanged.
- **Cache:** build is skip-if-exists; `--force` rebuilds; a corpus/`corpus_types` change invalidates
  the cache key (different `corpus_name` dir) so a stale index is never silently served.

## 8. Failure modes & guards
- **Shallow pull caps recall** (the §7.3.1 trap) → `topk_internal` from P0, `≥ fusion_K`; assert
  `topk > 60` and warn on the legacy default.
- **Doc-format skew train↔serve** (the `bge_m3_ft` failure) → single doc source = `id_to_metadata`
  (A1); R3 builds no text; a test pins the indexed corpus to the accessor output.
- **Stale cache after corpus/`corpus_types`/A1 change serves an old index** → cache key includes
  `corpus_name` (joined `corpus_types`) + an `enriched` flag + an A1 corpus-version tag; skip-if-exists
  only on an exact key match; `--force` override.
- **Non-canonical / mixed id emission** → `canonical_track_id` at the output boundary + F3 `⊆ catalog`
  assert (plan §7.3 req #1); never use the doc-text `strip_track_id_prefix` as an id normalizer.
- **Empty/degenerate query silently dumps or crashes** → return `[]` + VISIBLE log; RRF treats absent
  as 0 (fails safe).
- **Indexing `test_tracks` / leak split** → build over `all_tracks` only (F1 §8); assert the catalog
  split at init.
- **`bm25s` tokenizer/version drift changes ranks** → pin the `bm25s` version + stopword/lowercase
  config in `config/<exp>.yaml`; the determinism test catches a silent tokenizer change.
- **No unique recall (pure noise after fusion)** → §6 unique-recall report; R7 drops/zero-weights it
  (plan §7.3 req #2). For BM25 this is a safeguard, not the expected outcome.

## 9. Config knobs (values default; types validated by the F2 loader)
- `retrieval.channels[<bm25>].type` = `bm25`; `.label` (default `"bm25"`); `.weight` (R7 RRF weight,
  from P0 sweep); `.topk_internal` (**P0-sized, ≥ fusion_K, target 300–500 — NOT 60**); `.query_key`
  (optional; reads `Query.per_channel[label]`, else `Query.text`).
- `retrieval.bm25.corpus_types` (default `["track_name","artist_name","album_name","release_date"]`;
  plan §7.2.1 adds `"tag_list"`); `.enriched` (bool, default `true` — consume A1 docs; `false` =
  base-doc parity/ablation).
- `retrieval.bm25.k1`, `.b` (BM25 params; defaults the `bm25s` library defaults, recorded for parity).
- `retrieval.bm25.lowercase` (default `true`), `.stopwords` (default the pinned `bm25s` English list).
- `retrieval.bm25.cache_dir` (default `./cache`), `.force_rebuild` (default `false`).

## 10. Definition of Done & review checklist
- [ ] `BM25Channel` implements F2 `RetrievalChannel`; `batch_text_to_item_retrieval` returns canonical
      ids only, ≤ `topk`, aligned to `queries`; `batch_context`/`user_ids` accepted as no-ops.
- [ ] Corpus built **only** from `Catalog.id_to_metadata(enriched=True)` over `Catalog.index_to_id`;
      R3 builds no doc text; graceful `enriched=False` fallback.
- [ ] Pulled to `topk_internal` (P0, ≥ fusion_K); legacy-60 default removed + guarded.
- [ ] `canonical_track_id` applied at the output boundary; F3 `⊆ catalog` passes (plan §7.3 req #1).
- [ ] recall@{50,100,200,500} (overall + cold/warm) + unique-recall logged to `reports/experiments.md`;
      base-`corpus_types` baseline-parity check passes; A1 enrichment lift (or null result) recorded.
- [ ] All §7 tests green (conformance, id-space, doc-source, pull-depth, determinism, cache, edges);
      `bm25s` + tokenizer config pinned in `config/<exp>.yaml` (train==serve).
- [ ] Skip-if-exists build with `--force`; cache key invalidates on corpus/`corpus_types`/A1 change.
- [ ] Code review approved; no `Any` in public signatures; CPU-only, deterministic, read-only catalog.

## 11. Build order & dependencies
**Built right after F1+F2+F3 and P0, in parallel with R1** (R3 is on the day-3–4 first-submission
critical path: `F1+F2+F3 → P0 → R1+R3 → R7 → L1 → trivial responder → D1`, plan §17 / `000_INDEX.md`).
Depends on: **F1** (`Catalog`, `id_to_metadata(enriched)`, `canonical_track_id`, id space), **F2**
(`RetrievalChannel`, config), **A1** (enriched corpus; R3 runs degraded without it), **R1** (`Query`),
**P0** (`topk_internal`, cold/warm split, `corpus_types` to include), **F3** (`recall_at_k`).
**Blocks:** **R7** (fuses R3's `list[list[track_id]]`) and therefore K1/K2/L1/S1/D1 and the first
valid Blind-A submission. It is the minimum lexical channel needed for R7 to produce a non-trivial
fused pool.
