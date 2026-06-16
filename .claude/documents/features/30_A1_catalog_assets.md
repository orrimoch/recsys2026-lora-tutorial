# A1 — Catalog Enrichment / doc2query + Catalog Embedding (offline, cached)

> Assets module. The **one-time, offline, fully-cached** pipeline that (a) enriches the sparse
> 47,071-track catalog with doc2query/LLM blurbs + inferred mood/genre tags (Gemini-lite,
> resumable) and (b) embeds the catalog for the dense channel. Outputs an **enriched corpus keyed by
> canonical `track_id`** — read by R3 (BM25) and R4 (dense) **only** through the F1 surface
> `Catalog.id_to_metadata(track_id, enriched=True)` — plus **cached embedding matrices** mirrored to
> the HF Hub. Pure offline batch job: it never sees a conversation, a turn, or a gold. See
> `000_INDEX.md` for the catalogue and the Phase-1 shared contracts. Plan §12, §15, §6.2.

## 1. Purpose
Close the **conversational↔metadata vocabulary gap** (P0's smoking gun: thin docs — name/artist/
album/few tags — bury new-artist golds under intent queries like *"high-energy hip-hop for
driving"*). For each of the 47,071 catalog tracks, an LLM writes a short descriptive blurb +
example listener-requests (doc2query) + inferred mood/genre tags **from track metadata only**, which
A1 **appends** to the original doc (expansion, not replacement — exact name/artist/album match
survives), then re-embeds the enriched corpus with a configurable encoder. A1 produces two cached,
content-hashed artifacts — the **enriched doc parquet** (R3/R4 read it via F1) and the **dense
embedding matrix** (R4/R5/R6 read it via F1 `TrackEmbeddings`) — and nothing else. Its gate is
**enrichment coverage % + a measured recall@K lift of enriched-vs-raw docs** (via the F3/P0 probe):
**keep enrichment only if it lifts recall.**

## 2. Interface / contract
A1 is a **pair of offline scripts + their cached outputs**, not an importable serve-time module. It
runs once (resumable), writes artifacts under `paths.cache_root`, and mirrors them to the Hub; the
serve path reads those artifacts exclusively through **F1** — A1 exports no Python surface that R3/R4
call at request time.

**A1 scripts (ported from salvage — see §5):**
```python
# mcrs/assets/enrich_catalog.py  (CLI; resumable; Gemini-lite, thinking OFF)
#   reads:  Catalog.metadata(track_id) for every track_id in Catalog.track_ids  (F1, raw row)
#   writes: <cache>/enriched_docs/<recipe_hash>/docs.parquet
#           columns: track_id (canonical str), enriched_text (str), tags_inferred (list[str]),
#                    doc_text (str = raw meta_text + " | " + enriched_text + tags), content_hash (str)
def main(argv=None) -> int: ...
def build_enrich_prompt(meta: dict, n_requests: int) -> tuple[str, str]: ...   # pure, unit-tested
def clean_enrichment(raw: str, max_chars: int) -> str: ...                      # pure, unit-tested
def enriched_document(meta: dict, enrichment: str, tags: list[str]) -> str: ... # pure, unit-tested
def select_pending(rows, done_ids) -> list: ...                                # pure (resume)

# mcrs/assets/embed_catalog.py  (CLI; one batched encode pass)
#   reads:  the enriched docs parquet (--doc-source) OR Catalog.id_to_metadata(tid, enriched=...)
#   writes: <cache>/dense_local/<safe_model>/<label>/track_embeddings.pkl
#           {"track_ids": list[str], "track_mat": np.ndarray (47071, dim, float32, L2-normalized),
#            "doc_source_hash": str, "model_revision": str}
def build_doc_text(metadata: dict, fields: list[str]) -> str: ...               # pure, unit-tested
def main() -> int: ...
```

**The enriched-doc accessor (A1 → R3/R4) — the pinned contract:** A1 writes the enriched corpus; the
text is read **only** through the F1 `Catalog` surface, with enrichment layered by A1's parquet:
```python
Catalog.id_to_metadata(track_id: str, enriched: bool = False) -> str
#   enriched=False -> raw corpus-typed doc (the F1 baseline, ", ".join of List[str] fields)
#   enriched=True  -> the A1 doc_text for track_id IF present in the loaded enriched parquet,
#                     ELSE fall back to the raw doc (graceful degradation; never raises, never empty).
```
R3/R4 pass `enriched=<config bool>`; they **never** read A1's parquet directly and **never** build
doc text themselves (Phase-1 shared contract: "BM25/dense docs come from here"). F1 owns loading the
parquet (path/hash from config) and merging it onto the raw rows; A1 only **produces** the parquet.
The embedding matrix is read symmetrically via `TrackEmbeddings.matrix(modality)` with the A1
`label`/modality registered in config — row order is `Catalog.id_to_index` (Phase-1 embedding
accessor).

**Up/downstream wiring:** P0 (→ which fields are sparse enough to enrich, the recall-ceiling baseline
to beat) → **A1** → R3 (BM25 over enriched docs), R4 (dense over enriched docs / A1 embedding
matrix), and the A1 embedding matrix is also available to R5/R6. F3/P0 runs the keep/drop ablation.

## 3. Dependencies
- **Data (read-only, via F1):** `Catalog.metadata` / `Catalog.track_ids` over `all_tracks`
  (verified **47,071** rows; List[str] fields `track_name`, `artist_name`, `album_name`, `tag_list`,
  `artist_id`, `album_id`, `ISRC`; scalar `popularity`/`release_date`/`duration`). **No conversation,
  user, turn, or gold data is read** — A1 is catalog-only by construction (§4.6).
- **Modules:** F1 (`Catalog` raw rows + canonical id space; F1 owns the `enriched=` accessor + parquet
  load), F2 (config schema for `assets.*`, `data.corpus_types`, `model_revisions{}`), F3/P0 (the
  recall probe that scores the gate).
- **Models / external APIs:**
  - **Gemini-lite** for enrichment (default `gemini-2.5-flash-lite`, **thinking_budget=0** — descriptive,
    not reasoning, so no thinking tokens; cheaper, faster, no output-budget truncation). Optional
    local HF instruct fallback (`--model Qwen/Qwen2.5-7B-Instruct`) for an API-free run. Routed via
    the ported `GeminiClient` (plan §6.2 responder/enrichment client). Competition rules permit
    external APIs (memory: rules check — Gemini ALLOWED).
  - **Embedding encoder** (sentence-transformers, default per P0 §7.2; e.g. BGE-large-en-v1.5 /
    Qwen3-Embedding-0.6B). GPU one-time; CPU/MPS fallback for smoke.
- **Config:** `assets.enrich.*`, `assets.embed.*`, `data.corpus_types`, `paths.cache_root`,
  `paths.hub_repo`, `model_revisions.{enrich_llm, embed_encoder}`, `seed`.
- **Libs:** `datasets`, `pandas`, `numpy`, `sentence-transformers`, `torch`, the Gemini SDK; stdlib
  `hashlib`/`concurrent.futures`. No serve-time deps (artifacts are plain parquet/pickle).

## 4. Design & logic

### 4.1 Enrichment = document expansion (doc2query), not replacement
For each track, build a prompt **only from its metadata** (artist/title/album/tags/year/popularity)
and ask Gemini-lite for: (1) a 2-sentence description (genre, era, mood, energy/instrumentation, 2
similar artists), (2) exactly `n_requests` (default 4) short example listener-requests phrased like a
person talking to a music app (the doc2query signal — injects real user vocabulary into the
indexable doc), and (3) a few inferred mood/genre tags. The cleaned enrichment is **appended** to the
verbatim `meta_text` (`enriched_document(meta, enrichment, tags)`), so exact name/artist/album lexical
match is preserved and BM25/dense only *gain* the expanded vocabulary. Output is flattened to one
indexable line (drop markdown/bullets, collapse whitespace, cap `max_chars` so the embedding stays
within the encoder seq limit).

**Hallucination guard (prompt, ported verbatim from salvage):** the system prompt forces grounding
strictly in the provided metadata + tags + widely-known facts; if the model doesn't recognize the
track it must **infer style from tags and invent no specific facts** (no fake dates/collaborators/
labels/charts). P0 §5 (metadata quality) tells A1 which fields are sparse enough to warrant
enrichment — A1 can enrich the **full** catalog or only the **sparse subset** (`assets.enrich.only_sparse`)
if full-catalog cost is too high (plan §15 free-tier-first).

### 4.2 Embedding the enriched corpus
One batched `model.encode(...)` pass over all 47,071 enriched `doc_text`s (sentence-transformers
sorts-by-length + batches internally → a single vectorized pass, no per-track loop), L2-normalized,
stored **float32** (compute dtype may be fp16 on CUDA to fit a large model; storage dtype fixed).
The pickle stores `{"track_ids", "track_mat", "doc_source_hash", "model_revision"}`; F1 stacks it into
the `(47071, dim)` matrix **row-ordered by `Catalog.id_to_index`** (A1 stores ids alongside so F1
re-indexes, never trusting file row order). `--doc-source` = the A1 enriched parquet's `doc_text`
column; with no `--doc-source` it falls back to `build_doc_text`/`id_to_metadata` (raw), which is
exactly how the **raw-vs-enriched A/B** is produced (same encoder, two doc sources).

### 4.3 Caching by content hash (plan §15 — "never recompute")
- **Recipe hash** = sha256 of `{prompt_template_version, n_requests, max_chars, llm_model+revision,
  only_sparse, corpus_types}`. The enriched parquet lives at `enriched_docs/<recipe_hash>/docs.parquet`,
  so changing any knob writes a new artifact and never silently mixes recipes. Each row also carries a
  per-track `content_hash` (sha256 of its source `meta_text`) so a metadata refresh re-enriches only
  changed tracks.
- **Embedding cache** keyed on `<safe_model>/<label>/`, with `doc_source_hash` (the recipe hash of its
  input) + `model_revision` stored **inside** the pickle. F1/R4 assert the embedding's `doc_source_hash`
  matches the configured enriched parquet — a mismatch is a hard error (no train/serve doc skew).
- **Resumable enrichment:** re-running reads existing rows, skips `track_id`s already done
  (`select_pending`), checkpoints every `CHECKPOINT` (2000) tracks; an interrupt never re-pays, failed
  (empty) enrichments are **not** persisted so a re-run fills them. A limited/SMOKE re-run over a full
  parquet is **non-destructive** (carry-forward prior `doc_text` for tracks not in this run's subset).
- **Hub mirror (plan §15 — "cache embeddings to HF Hub"):** after a full run, push both artifacts to
  `paths.hub_repo` (a private dataset repo) tagged by hash, so Colab/Blind runs `hf download` instead
  of recomputing. The hash in the path is the cache key; the run records it in the submission config.

### 4.4 Cost control (plan §15)
One-time over 47k tracks: start on the **free tier**, batch (`ThreadPoolExecutor`, default 24
concurrent), thinking OFF, short outputs (`max_output_tokens` default 320). A **dry-run token count**
(`--limit N`) estimates cost/1k before scaling; a hard `assets.enrich.budget_calls` cap aborts past
budget. If full-catalog is too costly, `only_sparse=True` enriches just the P0-flagged sparse subset.
Embedding GPU cost is minutes (one pass); both artifacts are cached + Hub-mirrored so re-runs are
free.

### 4.5 Edge cases
- API failure on a track → empty enrichment, **not cached**, retried next run; that track's
  `id_to_metadata(enriched=True)` falls back to its raw doc (no gap in the corpus).
- List[str] metadata fields (`track_name` etc. are **lists**) → `meta_text`/`build_doc_text` `", ".join`
  them and `_first(...)` for prompt fields (matching salvage); tests cover multi-value rows.
- A track absent from the enriched parquet (sparse-only run, or a new catalog row) → F1 falls back to
  raw doc; coverage % reflects this.
- Prompt-injection-safe: track names/tags are sanitized before templating (plan §6.2 guardrail) — they
  are catalog data, not user input, but the same hygiene applies.

### 4.6 Causal / no-leak constraint (the load-bearing safety property)
**A1 reads track metadata ONLY — never conversations, turns, user histories, or golds.** Enrichment
is a property of the *item*, identical for every (session, turn) at train and serve, so it **cannot**
leak future-turn or gold information into retrieval. This is the structural reason A1 lives in the
offline "Assets" phase: its output is causal-invariant. Tests assert A1's input loader touches no
conversation/gold data path, and that two runs produce row-identical artifacts given a fixed seed +
pinned LLM/encoder revision.

## 5. Reuse
- **Enrichment:** **port** `salvage/scripts/enrich_catalog.py` (doc2query: description + N
  listener-requests, resumable, checkpointed, non-destructive limited re-run, `build_records`
  carry-forward) — its pure helpers (`meta_text`, `build_enrich_prompt`, `clean_enrichment`,
  `enriched_document`, `select_pending`) are already unit-test-shaped; **extend** to also emit
  `tags_inferred` and the recipe/`content_hash` columns, and to canonicalize `track_id` via F1.
  `salvage/scripts/enrich_track_docs.py` (single rich paragraph + HF-local fallback) is the
  **alternate recipe** — keep its `_hf_generate` path for an API-free run; pick the recipe by P0/gate.
- **Embedding:** **reuse/port** `salvage/scripts/embed_catalog.py` nearly verbatim — it already has
  `--doc-source` (enriched parquet), `--doc-col`, fp16-on-CUDA, single batched encode, the
  `{track_ids, track_mat}` pickle, and `id_to_metadata`/`build_doc_text` recipes. **Extend** to store
  `doc_source_hash` + `model_revision` in the pickle and to push to the Hub.
- **LLM client:** reuse the ported `GeminiClient` (`thinking_budget=0`, retries, batched threads).
- **What's new:** the `enriched=` layering lives in **F1's** `id_to_metadata` (A1 only produces the
  parquet); the content-hash cache keys, Hub mirror, `tags_inferred`, and the recall-lift gate harness
  call are new. Move scripts into `mcrs/assets/` (off `salvage/scripts/`) and align ids to the F1
  canonical normalizer.

## 6. Eval & acceptance gate
**Two numbers; both measured via the F3 recall primitives on the P0 dev probe; enrichment is kept
only if it lifts recall.**
1. **Coverage %** = fraction of the 47,071 tracks with a non-empty `enriched_text` (and, for
   sparse-only mode, of the targeted subset). **Gate: ≥ 95%** of targeted tracks enriched (the
   remaining ≤5% gracefully fall back to raw docs); log the count + API-failure tally.
2. **Recall@K lift (raw vs enriched), the decisive gate** — using the **identical** P0/F3 probe
   (same dev `TurnContext`s, same encoder, same recall@{50,100,200,500} function), build two BM25 +
   two dense indices, one over `id_to_metadata(enriched=False)` and one over `enriched=True`, and
   compare per-channel recall@K (overall + cold/warm split). **Keep enrichment iff Δrecall@100 > 0**
   (a real, non-noise lift) on at least the dense channel (the weak link enrichment targets); a flat
   or negative lift means **drop the lever** (it adds cost/latency for nothing — plan §6.2/§12 ablation
   rule). Report the per-channel + fused recall delta and the new-artist-gold rescue count.

Measured in the F3 harness / P0 recall-ceiling notebook; numbers logged to `reports/experiments.md`
+ memory. (Note: memory records prior doc-enriched attempts were Blind-nDCG-flat under a strong
ColBERT/CLAP stack — A1's gate is therefore **recall@K on the probe**, the cleanest signal, and the
keep-decision is honest A/B, not a foregone conclusion.)

## 7. Tests
- **Unit (pure helpers):** `meta_text` joins List[str] fields + appends tags/year; `build_enrich_prompt`
  degrades cleanly on missing fields; `clean_enrichment` strips markdown/bullets + caps length;
  `enriched_document` = base + " | " + enrichment (and == base when enrichment empty); `select_pending`
  excludes done ids; `build_doc_text` skips empty/absent fields (no `field: ` artefacts).
- **Caching/determinism:** identical knobs → identical `recipe_hash` and row-identical parquet; changing
  a knob changes the hash/path; embedding pickle stores the matching `doc_source_hash`; two embed runs
  → byte-identical matrix given fixed seed + revision; resume skips done ids; a SMOKE re-run does not
  truncate a full parquet (carry-forward).
- **No-leak (the safety gate):** A1's loaders read **no** conversation/user/gold path (assert the
  import/IO surface); `enriched_text` is a function of metadata only; `id_to_metadata(enriched=True)`
  is identical across all turns of a session.
- **Wiring/integration:** F1 `Catalog.id_to_metadata(tid, enriched=True)` returns the A1 `doc_text`
  for a covered track and the **raw** doc for an uncovered one (never raises/empty); the A1 embedding
  matrix loaded by `TrackEmbeddings.matrix(label)` is row-aligned to `id_to_index`
  (`matrix[id_to_index[tid]] == vector(tid)`); a `doc_source_hash` mismatch raises.
- **Gate harness (small fixture):** the raw-vs-enriched recall A/B runs end-to-end on a tiny catalog +
  dev fixture and produces the Δrecall table F3/P0 expects.

## 8. Failure modes & guards
- **Hallucinated facts in blurbs** (fake dates/collaborators) → polluted docs, wrong matches. Guard:
  the strict "infer-from-tags-only, invent-nothing" system prompt; tags-grounded; spot-check sample.
- **Enrichment *replaces* instead of *appends*** → loses exact name/artist match, recall regresses.
  Guard: `enriched_document` always prefixes verbatim `meta_text`; unit test asserts base is a prefix.
- **Train/serve doc skew** (R3/R4 index a different recipe than configured) → silent recall loss.
  Guard: recipe `content_hash` in the path + `doc_source_hash` stored in the embedding pickle + F1/R4
  assert-on-mismatch.
- **Cost/quota blowup** on the 47k batch. Guard: free-tier first, `--limit` dry-run token estimate,
  `budget_calls` hard cap, thinking OFF, short outputs, `only_sparse` fallback (plan §15/§16).
- **Embedding/metadata row misalignment** → personalization points at the wrong track. Guard: store
  ids in the pickle; F1 re-indexes by `id_to_index`, never file order (mirrors F1 §6.4).
- **List-typed fields treated as scalars** → `KeyError`/garbled docs. Guard: `_first`/`", ".join`;
  multi-value tests.
- **Enrichment keeps a flat/negative lift** "because we built it" → wasted latency. Guard: §6 gate is
  hard — drop the lever if Δrecall@100 ≤ 0 (memory: prior doc-enriched was nDCG-flat; gate on recall).
- **Cache corruption / partial write** → unreadable parquet. Guard: atomic `tmp`+`os.replace`;
  checkpointed flushes; re-runnable.

## 9. Config knobs (defaults; types validated by the F2 loader)
- `assets.enrich.enabled` (bool, default `true`) — run/skip the enrichment pass.
- `assets.enrich.llm_model` (default `gemini-2.5-flash-lite`), `assets.enrich.thinking_budget` (default `0`).
- `assets.enrich.n_requests` (int, default `4`), `assets.enrich.max_chars` (int, default `1500`),
  `assets.enrich.max_output_tokens` (int, default `320`).
- `assets.enrich.batch_size` (int, default `24` concurrent calls), `assets.enrich.budget_calls` (int hard cap).
- `assets.enrich.only_sparse` (bool, default `false`) + `assets.enrich.sparse_predicate` (P0-defined: e.g.
  `tag_list` empty or `< k` tags) — enrich only the sparse subset to cap cost.
- `assets.enrich.prompt_template_version` (str) — bumps the recipe hash on a prompt change.
- `assets.embed.encoder` (default per P0 §7.2), `assets.embed.label` (modality label R4 registers),
  `assets.embed.fields` (raw fallback), `assets.embed.batch_size`, `assets.embed.max_seq_len`,
  `assets.embed.dtype` (default `auto`), `assets.embed.normalize` (default `true`).
- `data.corpus_types` — the raw doc fields (shared with F1/R3); `model_revisions.{enrich_llm, embed_encoder}`
  (pinned for determinism); `paths.cache_root`, `paths.hub_repo`; `seed`.
- Consumer flags (read by R3/R4, **declared here for the contract**): `retrieval.channels[].extra.enriched`
  (bool) — whether the channel reads `id_to_metadata(enriched=True)` / the A1 embedding `label`.

## 10. Definition of Done & review checklist
- [ ] `enrich_catalog.py` + `embed_catalog.py` ported into `mcrs/assets/`, ids canonicalized via F1,
      pure helpers + content-hash caching + Hub mirror in place.
- [ ] Full-catalog (or sparse-subset) enrichment run; **coverage ≥ 95%** of targeted tracks logged.
- [ ] Enriched parquet + embedding pickle written, content-hashed, mirrored to `paths.hub_repo`;
      embedding pickle stores `doc_source_hash` + `model_revision`.
- [ ] **Recall A/B run** (raw vs enriched) on the F3/P0 probe; Δrecall@{50,100,200,500} (overall +
      cold/warm) logged to `reports/experiments.md`; **keep/drop decision recorded** (keep iff
      Δrecall@100 > 0 on dense).
- [ ] F1 `id_to_metadata(tid, enriched=True)` returns A1 docs for covered tracks, raw for uncovered
      (verified, never raises); embedding matrix row-aligned to `id_to_index`; hash-mismatch raises.
- [ ] No-leak test green: A1 reads catalog metadata only (no conversation/user/gold path); enrichment
      identical across a session's turns.
- [ ] Unit + caching/determinism + wiring tests green; resume + non-destructive SMOKE re-run verified.
- [ ] Cost: free-tier-first, dry-run token estimate logged, `budget_calls` cap honored.
- [ ] Code review approved; scripts deterministic given pinned revisions; no `Any` in public helpers.

## 11. Build order & dependencies
**Built in the Assets phase, after F1+F2+F3+P0** (`000_INDEX.md` graph: `F1 → A1 → R3, R4`). A1 needs
F1's `Catalog` (raw rows + canonical id space + the `enriched=` accessor it layers onto), F2's config
schema, and **P0's findings** (which fields are sparse → enrichment target; the raw recall-ceiling
baseline to beat). **Blocks:** R3 (BM25 over enriched docs) and R4 (dense over the enriched
corpus / A1 embedding matrix); the A1 embedding matrix is also consumable by R5/R6. A1 is **off the
critical path to the first valid submission** (`F1+F2+F3 → P0 → R1+R3 → R7 → L1 → responder → D1` uses
raw docs first); it layers in as a recall lever once the spine runs — and only stays if §6's gate
passes. As an offline cached asset, it runs **once** (Colab/free-tier), and every later run reads the
Hub-mirrored artifacts.
