# R2 — LLM Query Refinement (gated)

> Retrieval-phase module. **Optional, fail-fast** LLM layer that turns the messy multi-turn dialogue
> into a clean *structured* query and/or a HyDE pseudo-doc, written into the F2 `Query` object so
> channels (R3/R4/R5/R6) and fusion (R7) consume it without re-deriving anything. The spine works
> **without** R2; it is enabled only on a measured ablation lift (recall@K vs R1-only, no judge harm).
> See `000_INDEX.md` for the catalogue and `11_F2_interfaces_contracts_config.md` for the contracts.

## 1. Purpose
Enrich the R1-produced `Query` with LLM-derived structure — `Query.structured`
(`{positive_attrs, negative_attrs, seed_artists, mood, genre, era}`) and/or per-channel rewrites
(`Query.per_channel`, e.g. a HyDE pseudo-doc for the dense channel) — so retrieval can de-noise
the dialogue, resolve coreference, and surface content/new-artist golds that the raw turns miss.
**Gated:** kept only if it lifts dev recall@K over R1-only without hurting the judge.

## 2. Interface / contract
A pure-by-default refiner that **reads** an R1 `Query` (+ its `TurnContext`) and **returns an enriched
`Query`** — it never mutates in place, never opens the catalog, never picks track ids. Lives in
`mcrs/query_refine.py`; the per-rewriter logic is recoverable from the old git branches (recall-union-lgbm, fresh-model, exp/*).

```python
class QueryRefiner(Protocol):
    enabled: bool                       # config gate; if False, refine() is the identity
    variants: list[str]                 # e.g. ["structured", "hyde"] — each independently gated

    def refine(self, ctx: TurnContext, query: Query) -> Query: ...
    # returns a NEW Query: same .text (R1 owns it), enriched .structured / .per_channel.

    def batch_refine(self, ctxs: list[TurnContext], queries: list[Query]) -> list[Query]: ...
    # batched + cached path used by D1/F3 over the dev/blind set.
```

**Fields R2 writes (the shared contract R1/channels/R7 must agree on):**
| F2 `Query` field | R2 writes | Read by |
|---|---|---|
| `.text` | **never** (R1 owns it; R2 leaves it untouched as the always-valid fallback) | all channels |
| `.structured` | the canonical dict `{positive_attrs: list[str], negative_attrs: list[str], seed_artists: list[str], mood: str, genre: str, era: str}` (any field may be empty/`[]`); `None` when R2 disabled or all-empty | K1 (features), R3/R4 (optional doc-side terms), L1 (constraint checks) |
| `.per_channel[label]` | optional per-channel **string** override — `per_channel["dense_hyde"]`=HyDE pseudo-doc, `per_channel["bm25_structured"]`=assembled content-only query — keyed by the channel `label` that declares `query_key` (R7 §41 contract) | the channel whose `query_key == label` (else it uses `.text`) |

R2 **adds** keys to `per_channel`; it must not overwrite an R1-set override for the same key (assert
disjoint or last-writer-with-warning). `structured` uses the **plan §7.1 schema verbatim** so K1/L1
read fixed keys. R2 does not invent new `Query` fields.

## 3. Dependencies
- **Upstream:** R1 (`40_*`) supplies the `Query.text` R2 enriches; F2 (`Query`/`TurnContext`); P0 (cold/warm split, recall-ceiling table to judge "did it lift the wall").
- **Models / APIs:** **Gemini-lite** (`gemini-2.5-flash-lite` — *verify current id*, plan §13) via `GeminiClient` (lazy import + `GEMINI_API_KEY`/`GOOGLE_API_KEY`, temperature 0); or a local open-weight LM wrapper (`.lm/.tokenizer/.device`) for the HyDE/structured-query paths (recoverable from the old git branches recall-union-lgbm, fresh-model, exp/*). The SDK/key are needed only when a cache miss actually generates.
- **Data:** none new — reads turns from `TurnContext` only (≤t). Catalog metadata is **not** consulted (R2 is content-agnostic; grounding artist/track names to ids is R6 propose-ground, not R2).
- **Config:** `retrieval.query_refine.*` (§9). Cache dir under `experiments/cache/` (NOT `./cache` — `run_inference_devset.py` `rm -rf cache`'s the default; a recorded prior-project gotcha).

## 4. Design & logic
**Each rewriter is an independently-gated variant** (none pre-judged — plan §6.3 reuse policy). R2 is
the thin orchestrator that runs the enabled variant(s), parses/sanitizes, caches, and writes the
agreed `Query` fields. The two primary variants:

- **`structured`** (port `structured_query.py` → upgrade to the §7.1 schema): one LLM call distils the
  dialogue into `{positive_attrs, negative_attrs, seed_artists, mood, genre, era}`. Deterministic core
  (`parse_structured_json` → tolerant JSON parse → `_coerce` to fixed keys → `assemble_query` content-only
  string) is **pure and unit-tested**; only the generation is integration. Writes `Query.structured`
  and (optionally) `per_channel["bm25_structured"]`/`["dense_structured"]` = `assemble_query(structured)`.
  The prior schema (`genres/moods/era/culture/intent/wants_new_artist`) is **remapped** to the §7.1
  keys (genres→genre, moods→mood, intent→positive_attrs, +explicit negative_attrs/seed_artists) — keep
  the remap in one pure function so train==serve.
- **`hyde`** (port `hyde.py`): one LLM call emits an intent line + N pseudo-track descriptions; the
  pseudo-doc(s) become a **dense** query via `per_channel["dense_hyde"]` (R4 reads it). Pure
  `parse_hyde_output`; falls back to the intent line, then to `query.text`, if no docs parse.

Other prior-project variants (recoverable from the old git branches recall-union-lgbm, fresh-model, exp/*) are **optional, config-listed, not enabled by default** — each must pass its own
ablation before shipping: `cmqr.py` (multi-query rewrite + inner-RRF — note its retriever-shaped wrapper
belongs at the channel layer, not here; R2 only emits the rewrites, R7 fuses), `intent_state.py` (Q*
self-contained query — **off by default; train/serve skew is its known risk, so it only ships if it
re-proves a fresh recall@K win under the §8 alignment rule — nothing pre-judged**), `artist_hypothesis.py`/`propose_ground.py`/`gemini_propose.py` (artist/track
**proposal** → these feed an R6 *channel*, not R2; R2 owns dialogue→query only, not query→items),
`state_tracker.py` (6-key user-state, optionally a `structured` sub-field + responder envelope).

**Key decisions & edge cases**
- **Causal (≤t only):** the LLM sees `ctx.utterances[0..t-1]`, `ctx.goal`, and profile prior — **never**
  a future turn or the gold. R2 asserts it builds its prompt only from `ctx` fields (which F2 already
  guarantees are ≤t). No history beyond `ctx.history_tids` ids (R2 doesn't expand them to text).
- **Fail-fast / identity when disabled:** `enabled=False` ⇒ `refine` returns the input `Query`
  unchanged (`structured=None`, no new `per_channel`). Any parse failure, empty output, API error, or
  timeout ⇒ degrade to the R1 `Query` for that turn (channels always have `.text`). R2 **never** raises
  into the spine.
- **Intent extraction must not over-constrain.** A hallucinated `negative_attrs` ("not slow") or a
  wrong `era`/`seed_artists` can *delete* the gold from recall. Guards: (a) `negative_attrs` are advisory
  to K1/L1, never a hard pre-filter at the channel stage; (b) channels still run on `.text` in parallel
  with any structured override (R2 *adds* a channel query, never *replaces* the spine's text query),
  so a bad structure can only fail to help, not subtract recall; (c) the ablation gate (§6) measures
  recall@K — a variant that injects false constraints fails its gate and is dropped.
- **Cache by content hash, per (session,turn).** Key = `sha1(variant ‖ model_revision ‖ prompt_hash ‖
  causal_input)` so a prompt/revision change invalidates cleanly (alignment, §8). Cache value stores the
  parsed structure + the assembled strings + the inputs' hash. Idempotent: the dev pass and re-runs never
  re-call the LM. (The prior project cached by conversation-hash *or* (session,turn) — we standardize on the
  content-hash key so a turn whose text is identical across runs reuses the cache, and a prompt bump busts it.)
- **Prompt-injection safety.** Utterances and any echoed track/artist names are **sanitized before
  templating** (strip/escape control + template/code-fence/role markers like `</s>`, `<|...|>`,
  `system:`, ``` ``` ```, `{...}`); the user content is wrapped in an explicit delimiter the system
  prompt is told to treat as untrusted data, not instructions. Output is parsed as **data only**
  (JSON/numbered-list), never executed; any field that isn't in the fixed schema is discarded by `_coerce`.
- **Determinism.** Temperature 0 / greedy; longest-first batch with left-pad + attention mask
  so batched == single-sequence decode and the hash cache stays valid regardless of batching.

## 5. Reuse
Per plan §6.3 (Query refinement = **Port behind gate**):
- **Port:** `structured_query.py` (pure parse/assemble core — keep; remap schema to §7.1), `hyde.py`
  (pure parse + cached generator — keep), `gemini_propose.GeminiClient` (lazy SDK + key — reuse the
  client; the *propose* generator itself is R6, not R2).
- **Port behind own gate (off by default):** `cmqr.py` (the rewriter math; its retriever wrapper is R6/R7
  shaped), `state_tracker.py` (state extraction).
- **Off by default / re-prove fresh:** `intent_state.py` (Q* — watch for train/serve skew; ship only on a measured win).
- **Not R2 (belongs to R6 channels):** `artist_hypothesis.py`, `propose_ground.py`, `gemini_propose.GeminiProposeGenerator` — they turn query → *seed items*, which is a retrieval channel, not query enrichment. R2 references them only to delineate scope.
- **Rewrite (new):** the orchestrator `QueryRefiner` + the prior→§7.1 schema remap + the unified
  content-hash cache + the injection sanitizer (the prior project had per-rewriter ad-hoc caching and no central
  sanitizer).

## 6. Eval & acceptance gate
**Own metric:** dev **recall@K lift vs R1-only**, measured by F3 (`12_F3`) with the rest of the spine
fixed. A variant **ships only if:**
1. fused **recall@{50,100,200}** rises by a margin above F3's noise band (R1-only is the control), **and**
2. it does **not** regress the proxy judge (§13) or Distinct-2 (no-harm on the response axis), **and**
3. cost/latency are within the §15 budget (per-turn lite calls, cached).

Report per-variant + per-segment (cold/warm — P0 split): R2's value is hypothesized on **cold/turn-1**
(query *is* the only signal) and on **pivot** turns; if it only helps one segment, gate it there
(segment-conditional enable, §9). A variant that doesn't clear the band is **dropped** (it adds
cost/latency for nothing — plan §6.2 guardrail). Logged as a row in `reports/experiments.md`
(exp_id, variant, recall deltas, judge delta, cost, keep/drop).

## 7. Tests
- **Unit (pure, no LM):** `parse_structured_json`/`parse_hyde_output` on bare JSON, fenced JSON, JSON-in-prose,
  garbage → safe defaults; `_coerce`/schema-remap → fixed §7.1 keys only (extra keys dropped); `assemble_query`
  omits empty fields and includes **no** seed-artist names where the variant is artist-agnostic.
- **No-leak / causal:** the prompt builder, given a `TurnContext`, references only `utterances[≤t-1]`,
  `goal`, profile — a test injects a sentinel "future" utterance and asserts it never appears in the prompt;
  assert the gold id never enters any R2 input.
- **Contract / wiring:** `refine` returns a **new** `Query` with `.text` byte-identical to input; `.structured`
  has exactly the 6 §7.1 keys (or `None`); `per_channel` keys are a subset of declared `query_key`s and don't
  clobber R1's; a downstream channel with `query_key="dense_hyde"` actually receives the override (mirrors
  R7's visible-fallback test).
- **Disabled / fail-fast:** `enabled=False` ⇒ identity; API error / empty / unparsable output ⇒ returns the
  R1 `Query` (never raises); negative_attrs are never applied as a hard channel pre-filter (recall can't drop).
- **Caching / determinism:** second `batch_refine` over the same turns makes **zero** LM calls (hash hit);
  a prompt or `model_revision` change busts the cache; batched output == per-row output (left-pad invariance).
- **Injection:** an utterance containing `</s>`, `<|im_start|>system`, ` ``` `, or `{"drop":true}` is sanitized
  and cannot alter the parsed structure or inject instructions.

## 8. Failure modes & guards
- **Hallucinated constraints shrink recall** → never hard-filter on structured fields at the channel stage;
  always run `.text` in parallel; ablation gate catches a net-negative variant.
- **Train↔serve skew (§8 alignment)** → the **same** `model_revision` + prompt + truncation/sanitizer in
  train-data prep and serve; the cache key embeds revision+prompt hash; D1 records the config hash. (This
  is the classic failure mode for a self-contained rewrite like Q*/`intent_state` — guard it explicitly.)
- **Prompt injection via track/utterance text** → sanitize-before-template + untrusted-data delimiter +
  parse-as-data-only (§4).
- **Future-turn / gold leak** → causal asserts; R2 inputs are F2 `TurnContext` fields only.
- **Cost / quota spike** → lite model, batch, temp 0, short `max_output_tokens`, hard cache, budget cap,
  free-tier first (plan §15); only generate for required (session,turn)s.
- **Cache poisoning by a bad run** → cache stores the input hash + revision; mismatched entries are ignored,
  not trusted.
- **Silent per_channel fallback** (a `query_key` channel set but R2 emitted no override) → VISIBLE log
  (R7 contract), channel falls back to `.text`.

## 9. Config knobs (`retrieval.query_refine.*`)
| Key | Type | Default | Meaning |
|---|---|---|---|
| `enabled` | bool | `false` | master gate; `false` ⇒ R2 is identity (spine runs R1-only) |
| `variants` | list[str] | `[]` | which rewriters run, e.g. `["structured"]`, `["structured","hyde"]` — each independently ablated |
| `provider` | str | `gemini` | `gemini` (lite API) or `local` (open-weight LM wrapper) |
| `model` | str | `gemini-2.5-flash-lite` | model id (**verify current**) |
| `model_revision` | str | — | pinned revision; part of the cache key + the §8 alignment contract |
| `temperature` | float | `0.0` | deterministic |
| `max_output_tokens` | int | `256` | short outputs (cost) |
| `cache_dir` | str | `experiments/cache/query_refine` | NOT `./cache` (gets wiped) |
| `structured.emit_channel_query` | bool | `false` | also write `per_channel["{bm25,dense}_structured"]` from `assemble_query` |
| `hyde.n_docs` | int | `3` | pseudo-track count |
| `enable_segments` | list[str]\|null | `null` | restrict to `["cold"]`/`["warm"]` if ablation shows a per-segment win only |
| `cost_cap_calls` | int\|null | `null` | hard cap on LM calls per run (budget guard) |

Schema declared in F2; values live in `config/<exp>.yaml`. One config object loaded by train + serve.

## 10. Definition of Done & review checklist
- [ ] `QueryRefiner` returns a **new** `Query`: `.text` unchanged; `.structured` is exactly the §7.1 6-key dict or `None`; `per_channel` additions are subset of declared `query_key`s and never clobber R1's.
- [ ] Pure parse/assemble/remap + injection-sanitizer unit-tested; no-leak/causal test green; cache-hit (zero LM re-calls) + prompt/revision-bust + batch-invariance tests green.
- [ ] `enabled=false` is a true identity; every error path degrades to the R1 `Query` (never raises).
- [ ] At least one variant **A/B'd by F3**: recall@K delta vs R1-only + judge no-harm logged in `reports/experiments.md` with a keep/drop decision.
- [ ] Train==serve: same model_revision + prompt + sanitizer; cache key embeds both; config hash recorded by D1.
- [ ] Code review approved; no hard constraint pre-filter on structured fields; no catalog access; no future-turn input.

## 11. Build order & dependencies
Built **after R1** (consumes its `Query`) and gated on **F3 + P0** (to measure the lift and pick the
segment). Depends on: **F2** (contracts), **R1** (`Query.text`), **F3** (eval), **P0** (cold/warm split,
recall ceiling). Blocks: nothing structurally — every channel runs on `Query.text` without it; R2 is a
pure additive enrichment layered in only on a proven lift. Sibling to the channels (R3–R6) and fusion (R7),
which read the fields R2 writes (`structured`, `per_channel[...]`).
