# R1 — Query Construction (causal, rule-based)

> Phase-1 entry module. Turns the causal `TurnContext` (utterances `1..t` only) into the F2 `Query`
> every retrieval channel consumes. Free, fast, deterministic, rule-based — no model, no API
> (the LLM refinement of the query is the separate, gated R2). This is the **single point** where the
> dialogue → query rendering happens; channels never re-derive a query from raw turns. Grounded in
> plan §7.1 (query construction), §8 (context-length/truncation), §5 step 3 (conversation anatomy);
> contracts from `11_F2_interfaces_contracts_config.md`; `query_key` wiring from `46_R7_rrf_fusion.md`.
> See `000_INDEX.md`.

## 1. Purpose
Construct, per session-turn `t`, the causal F2 `Query` from a `TurnContext`: a recency-weighted
concatenation of utterances `1..t` with the conversation `goal` appended, explicit entities
(artist/genre/era/mood) surfaced via metadata-vocabulary match, the §8 truncation/context-cap policy
applied (cap value supplied by P0; latest utterance never truncated), plus optional `per_channel`
query overrides (e.g. a compact query for a phrase-level / token-budgeted channel) wired to R7's
`query_key`. It does **not** call any LLM (that is R2, which layers `Query.structured` on top).

## 2. Interface / contract
Pure functions + a thin builder; no heavy deps. Lives in `mcrs/retrieval/query_construction.py`.
Consumes the F2 `TurnContext`; emits the F2 `Query` (does **not** redefine either — F2 owns the types).

```python
# mcrs/retrieval/query_construction.py

class QueryBuilder:
    """Causal, rule-based TurnContext -> F2 Query. Deterministic, model-free, no API."""
    def __init__(self, cfg: "QueryConfig", catalog: "Catalog" | None = None): ...
    #   catalog (F1) is OPTIONAL: only needed when entity extraction (vocab match) is enabled.

    def build(self, ctx: TurnContext) -> Query: ...                 # one turn
    def build_batch(self, ctxs: list[TurnContext]) -> list[Query]:  # vectorized over a split
        return [self.build(c) for c in ctxs]

# ----- pure helpers (unit-testable in isolation; no I/O) -----
def render_main_query(
    utterances: list[str], goal: str | None, cfg: "QueryConfig",
) -> str: ...
    # recency-weighted concat of utterances 1..t, goal appended, §8 cap applied.

def extract_entities(text: str, vocab: "EntityVocab") -> dict[str, list[str]]: ...
    # {"artist": [...], "genre": [...], "era": [...], "mood": [...]} by exact/normalized
    # surface-form match against catalog-derived vocabularies. Returns {} when no catalog.

def render_per_channel(
    utterances: list[str], goal: str | None, ctx: TurnContext, cfg: "QueryConfig",
) -> dict[str, str]: ...
    # channel_label -> override query (e.g. {"colbert": <compact>}), only for configured keys.
```

**Output object (F2 `Query`, unchanged):**
- `Query.text` — the shared main query (recency-weighted utterances + appended goal, §8-capped).
- `Query.per_channel` — `{channel_label: override_text}` for channels declaring a `query_key` (R7).
- `Query.structured` — left **`None`** here; filled later by R2 (gated). R1 never sets it.

**Wiring contract (the spine, per F2 §wiring + INDEX "Query flow"):**
`Conversations.turns` (F1) → `TurnContext` → **R1 `QueryBuilder.build` → `Query`** → (optional) R2
fills `.structured` → each `RetrievalChannel` (R3/R4/R5/R6) reads `Query.text`, or
`Query.per_channel[label]` when its R7 spec sets `query_key` → R7 fuses. A channel **never**
re-renders from raw `ctx.utterances`; it consumes the `Query` R1 produced. This guarantees train and
serve use byte-identical query text per channel (the salvage "single source / byte-identical" rule).

## 3. Dependencies
- **Modules:** F2 (`TurnContext`, `Query`, `QueryConfig` slice of the config schema). F1 *optionally*
  (`Catalog.metadata`/vocab + `canonical_track_id`) — only when entity extraction is on; R1 reads
  catalog **text vocab**, never returns track_ids (it builds queries, not candidates).
- **P0 inputs (decided, not by R1):** `context.context_cap` (token cap), `context.recency_window`
  (turns kept verbatim / weighting span), and the tokenizer used to measure length (must match the
  dense encoder's, §8). R1 *applies* these; the values come from P0's conversation-anatomy probe.
- **Libs:** stdlib + the encoder tokenizer (e.g. `transformers` `AutoTokenizer`) for the token-aware
  cap; `numpy` optional for the weight vector. **No model inference, no GPU, no network.** Pure/CPU.
- **Config:** the `query.*` block (§9).

## 4. Design & logic

### 4.1 Causal inputs only (the non-negotiable, plan §11 / F2 §4)
- R1 reads **only** `ctx.utterances` (turns `1..t`, F1 guarantees `len == turn_number`), `ctx.goal`,
  and `ctx.user_profile` (lean demographics). It **never** touches the gold (structurally absent from
  `TurnContext`), any future turn, or the `thought` field (F1 already excludes `thought` from
  `utterances`; R1 adds a defensive guard, §8 below). No history *track text* enters `Query.text` —
  the history-kNN/CF channels consume `history_tids` directly; rendering ids as text is the prior
  project's raw-UUID bug (see §8). R1 asserts each utterance is a human-readable string, not an id.

### 4.2 Main query — recency-weighted concatenation (plan §7.1)
- **Recency weighting.** Most-recent turns carry the active intent; older turns are context. R1
  builds the main text by ordering utterances oldest→newest and emphasizing recency via the
  configured `query.recency` strategy:
  - `window` (default): keep the last `recency_window` user turns **verbatim**; drop/compress older
    ones (the §8 policy — see 4.4). Simple, deterministic, matches salvage `last_user*`/`raw` family.
  - `repeat`: light recency boost by duplicating the latest user turn once at the front of the text
    (a free, lexical-only "boost" for BM25/dense without a learned weight) — gated, default off.
  - `weighted`: emit a parallel per-token weight vector only if a downstream channel can consume it;
    default off (most encoders/BM25 can't, so it's a no-op — don't ship dead complexity).
- **User-turn focus.** Like salvage `build_user_dialog`, the default renders **user**-role turn
  `content` (assistant/system text and music-turn ids are retrieval noise). A `query.include_roles`
  knob can re-include assistant turns if a P0/ablation shows lift; default = user turns only.
- **Goal appended (plan §7.1, §5 step 3).** When `ctx.goal` is non-empty, append a labelled
  `goal: <listener_goal>` line (mirrors salvage `raw_with_goal`'s `f"{base}\ngoal: {gt}" if gt else
  base`). Empty/whitespace/None goal degrades **byte-identically** to the goal-less query (no stray
  `goal:` token) — a parity requirement so train==serve and turn-1 cold rows aren't perturbed.
- **Turn-1 (cold-turn) handling.** A turn-1 ctx may have a single short utterance and often no goal.
  R1 must still emit a non-empty `Query.text` (the lone user utterance, possibly + goal). Empty
  utterances list → empty-but-valid query (logged), never a crash.

### 4.3 Explicit-entity extraction (plan §7.1, §3 insight 4)
- **Goal.** Surface artist/genre/era/mood terms the user *said*, so BM25/dense docs that literally
  contain those terms are favored — closing the conversational↔metadata vocabulary gap on the
  explicit-intent fraction P0 measures.
- **Vocabularies** (built once from F1 `Catalog`, cached): `artist` (artist_name surface forms),
  `genre`/`mood` (from `tag_list`), `era` (decade/year buckets from `release_date` plus a small
  hand-list like "80s/90s/2000s"). Matching is **rule-based**: exact / NFC-normalized / lowercased
  surface-form match (optionally whole-word, longest-match-first to avoid substring false hits like
  "Era" inside "Erasure"). No fuzzy/semantic match here (that's dense retrieval's job; fuzzy match
  injects false positives that displace golds).
- **Where the entities go.** R1 keeps extraction **lightweight and optional** (`query.extract_entities`,
  default per P0's explicit-metadata fraction):
  - They can be lightly **appended** to `Query.text` (a labelled `entities: artist=…; genre=…` tail)
    to bias the lexical channels — gated, only if the ablation lifts recall.
  - They are the natural seed for **R2** (`Query.structured`) and for K1 features (artist/genre/era
    match) — R1 may stash them on `Query.per_channel` under a reserved non-channel key OR simply
    leave them for R2; default = compute + log coverage, append only if gated on. **R1 never mutates
    `Query.structured`** (R2 owns that field) to avoid two writers racing on one slot.
- Extraction is **causal** (operates only on the rendered ≤t text) and **deterministic** (sorted,
  dedup'd term lists).

### 4.4 §8 truncation / context-cap policy (the alignment rule)
- **The cap is P0's number, applied here.** `context.context_cap` (token budget under the encoder
  tokenizer) and `context.recency_window` come from P0's cumulative-length distribution (§5.3) sized
  so the latest utterance + goal fit at ~p95. R1 **applies** the policy; it does not pick the value.
- **Never truncate the latest utterance (plan §8).** Truncation order is **oldest-first**: keep the
  goal line + the most-recent user turn verbatim; drop/compress older turns until the token budget
  fits. If even the latest utterance alone exceeds the cap, truncate **that** utterance from its
  **left** (keep its tail — the active ask), and log a WARN (a P0-flagged rarity). The goal line and
  the labelled `culture:` tail (if used) are placed **before** the long-utterance body so a rare
  right/left-truncation drops query filler, never the durable intent (salvage `compact_colbert`
  rationale).
- **Token-aware, not char-aware.** Measure with the **same tokenizer** the dense encoder uses (§8
  alignment); a char heuristic silently over/under-cuts. BM25 docs are short and unaffected, but R1
  emits one capped text reused across channels — sized to the tightest consumer (the 512-cap encoder).
- **Train==serve.** The cap, window, tokenizer id, and recency strategy live in `config/*.yaml`; the
  same config object loads in the train-data builder and at serve (F2 §config-drift guard), so the
  query is byte-identical in both (the salvage `build_colbert_train_data` parity discipline).

### 4.5 `per_channel` overrides (R7 `query_key` wiring)
- A channel whose R7 spec sets `query_key` reads its query from `Query.per_channel[query_key]`
  instead of `Query.text` (R7 §4 "Per-channel queries", salvage `resolve_sub_queries`). R1 populates
  `per_channel` **only** for the keys configured in `query.per_channel_specs` — it does not invent
  keys. Channels with no `query_key` use the shared `Query.text`.
- **Canonical use-case — a compact / phrase-level query** (e.g. ColBERT, a token-budgeted late-
  interaction channel): goal + culture labelled and placed FIRST, then the bare most-recent user turn
  (salvage `compact_colbert`), built under a **separate, tighter** `compact_cap`. This is the
  recurring prior-project lesson: the full dialogue truncates to noise past 512 tokens, so phrase-
  level channels get a humble high-signal slice.
- **Visibility guard (matches R7 §4 / §8):** if a channel declares a `query_key` but R1 produced no
  override for it, that's a config error — R1 logs a VISIBLE warning and R7's wiring then falls back
  to `Query.text` (never silent). A test asserts every configured `per_channel_spec` yields a key.
- **Causality holds per override.** Every override is built from the same ≤t inputs; no override may
  pull a field the main query couldn't (no future turns, no gold, no ids-as-text).

### 4.6 Determinism
- Pure functions of (`ctx`, `cfg`, cached vocab). Fixed iteration order, sorted entity lists,
  stable join separators. Two builds of the same ctx → byte-identical `Query`. Vocab built once with
  a fixed seed/order from F1. No randomness, no clock, no network.

## 5. Reuse
- **Port + adapt** salvage `crs_baseline.build_retrieval_query` (the mode zoo: `raw`,
  `raw_with_goal`, `last_user`, `last_user_with_goal`, `compact_colbert`, `bge_m3_structured`) and
  `sasrec_model.build_user_dialog` / `build_sasrec_context` (user-turns-only + goal-append, with the
  byte-identical goal-less degradation). **Mine for the rendering recipes, not the outcomes** — the
  prior approach plateaued and carried a raw-UUID-in-context bug; keep the *formats* (especially the
  goal/culture-FIRST compact slice and the `f"{base}\ngoal: {gt}" if gt else base` parity rule),
  drop the SASRec/state-tracker coupling and the leaky `thought`/id paths.
- **Rebuild** the public surface as F2-typed pure helpers (`render_main_query`, `extract_entities`,
  `render_per_channel`, `QueryBuilder`) — salvage had no central F2 `Query` builder, threaded modes
  through call sites, and mixed query construction with the responder/SASRec. R1 is the one clean
  source.
- **Do not reuse** salvage `intent_state`/`state_tracker` outputs as inputs (that's R2's LLM path and
  was a recorded train/serve-skew hazard); R1 stays model-free.

## 6. Eval & acceptance gate
R1 is an **input module** — it has no standalone recall number; its gate is downstream
**non-regression + causality**:
1. **Downstream-recall non-regression (the gate).** With R1's query vs a **raw-concat baseline**
   (newline-join all utterances, no recency/goal/cap policy — salvage `raw`), measured through the
   **R3+R4 channels → R7 fusion** on dev: fused recall@{20,100,200} **≥ baseline** (do **not** make
   recall worse), reported overall + cold/warm + turn-1. Any R1 lever (recency window, goal-append,
   entity-append, compact override) that does **not** hold-or-lift fused recall is dropped/defaulted-
   off (plan §7.3 req #2 — don't inject query noise). Measured in F3's harness against P0's cached
   recall-ceiling baseline.
2. **Causality / no-leak (hard pass).** The no-leak tests (§7) must all pass — a turn-`t` `Query`
   contains no turn-`>t` content, no gold, no `thought`, no raw track_id.
3. **Parity (hard pass).** Same config → byte-identical query in the train-data builder and at serve.

## 7. Tests
- **Unit (pure helpers):**
  - `render_main_query`: goal appended iff present; empty/None goal byte-identical to goal-less;
    recency `window` keeps the last-N verbatim; latest utterance never dropped.
  - `extract_entities`: exact/normalized surface-form match finds known artist/genre/era/mood;
    longest-match wins; no false substring hits; returns `{}` with no catalog.
  - `render_per_channel`: emits an override only for configured keys; compact slice puts goal/culture
    FIRST then the bare last user turn.
- **§8 cap:** with a tiny cap + a long oldest turn, oldest is dropped first and the latest utterance
  survives; with a single over-cap utterance, it's left-truncated (tail kept) + a WARN logged; cap is
  measured with the configured tokenizer (token count ≤ cap), not chars.
- **No-leak / causal (the gate's hard half):** build over a turn-`t` ctx whose `>t` turns and gold
  are known — assert none of that text appears in `Query.text` or any `per_channel` value; assert a
  `thought`-bearing or raw-UUID-bearing utterance is rejected/sanitized (the prior-project bug guard).
- **Wiring:** the `Query` R1 emits is consumed unchanged by an R3/R4 channel stub (`Query.text`) and
  by a `query_key` channel stub (`Query.per_channel[key]`); a `query_key` with no override triggers a
  visible-log path and falls back to `Query.text`.
- **Downstream non-regression (integration):** R1 vs raw-concat baseline through R3+R4→R7 on a dev
  fixture: fused recall@{20,100,200} non-regression (the §6 gate), overall + cold/warm + turn-1.
- **Determinism:** two builds of the same ctx → byte-identical `Query`; vocab build is stable.

## 8. Failure modes & guards
- **Raw track_id / UUID rendered into the query (the prior-project bug, plan §16, MEMORY).** A music-
  turn id or a stray id token in an utterance becomes meaningless dense/BM25 noise and silently caps
  recall. **Guard:** R1 renders **user-turn human text only**; it never reads `history_tids` as text;
  a sanitizer rejects/strips tokens matching the catalog-id shape (and the no-leak test asserts no
  raw id reaches the query). Spot-check 5 rendered queries before trusting any recall number (P0 hazard).
- **`thought` / future-turn leak.** F1 already excludes `thought`; R1 adds a defensive assert (any
  field outside `utterances[≤t]`/`goal`/lean-profile is unreachable) so a future F1 change can't
  silently leak. No-leak test enforces it.
- **Truncating the latest utterance (active intent loss, plan §8).** Guard: oldest-first truncation;
  latest utterance kept verbatim; only-if-it-alone-overflows left-truncate-with-WARN; goal/culture
  placed first so truncation never eats durable intent.
- **Goal-append perturbs the no-goal/turn-1 path.** Guard: byte-identical degradation when goal is
  empty (`… if gt else base`); parity test on a goal-less ctx.
- **Char-vs-token cap mismatch (train/serve skew, §8).** Guard: token-aware cap with the encoder's
  tokenizer id pinned in config; loaded by both train and serve.
- **Entity extraction false positives** (fuzzy/substring) displace golds. Guard: exact/normalized,
  whole-word, longest-match-first; off-by-default unless P0/ablation earns it; recall-non-regression
  gate catches a net-negative.
- **Silent per-channel-query fallback.** Guard: VISIBLE log + test (mirrors R7 §4) when a `query_key`
  has no override.
- **Empty utterances / malformed ctx.** Guard: emit a valid empty-but-logged query; never crash.

## 9. Config knobs (the `query.*` block; types validated by the F2 loader)
- `query.recency` — `"window" | "repeat" | "weighted"` (default `"window"`).
- `query.recency_window` (int) — turns kept verbatim / weighting span. **Value from P0** (§5.3
  cumulative-length probe); R1 applies it.
- `query.context_cap` (int, tokens) — the §8 token budget. **Value from P0**; R1 applies it.
- `query.tokenizer` (str) — HF tokenizer id used to measure the cap; **must match the dense encoder**
  (§8 alignment). Default = the P0/encoder choice (e.g. `BAAI/bge-large-en-v1.5`).
- `query.include_roles` — roles whose `content` is rendered (default `["user"]`).
- `query.append_goal` (bool, default `true`); `query.goal_label` (default `"goal"`).
- `query.extract_entities` (bool) — default per P0's explicit-metadata fraction; `query.entity_fields`
  (default `["artist","genre","era","mood"]`); `query.append_entities` (bool, default `false`, gated).
- `query.per_channel_specs` — `[{query_key, template, compact_cap?, include_culture?}]`; e.g. the
  compact ColBERT spec. R1 builds `Query.per_channel[query_key]` from each.
- `seed` (int) — deterministic vocab/order (R1 is otherwise deterministic without it).

## 10. Definition of Done & review checklist
- [ ] `QueryBuilder` + the three pure helpers implemented; emit the **F2** `Query` (no schema
      re-invention); `Query.structured` left `None` (R2 owns it).
- [ ] Recency-weighted concat + goal-append (byte-identical goal-less degradation) + §8 token-aware
      cap (latest utterance never truncated) implemented and tested.
- [ ] Entity extraction is exact/normalized, causal, deterministic, default-gated; vocab built once
      from F1, cached.
- [ ] `per_channel` overrides built only for configured `query_key`s (incl. the compact slice);
      visible-log guard on a missing override; consumed unchanged by an R7 `query_key` channel.
- [ ] No-leak/causal tests green (no future turn, gold, `thought`, or raw id reaches the query);
      raw-UUID guard in place; 5 rendered queries spot-checked.
- [ ] Downstream-recall non-regression vs raw-concat baseline (R3+R4→R7, dev, overall+cold/warm+turn-1)
      met and logged to `reports/experiments.md`; every gated lever justified by a recall number.
- [ ] Cap/window/tokenizer/recency locked in `config/<exp>.yaml`; train==serve byte-identical (parity
      test); context_cap + recency_window recorded as **P0 inputs** (R1 didn't decide them).
- [ ] Code review approved; helpers pure (no I/O), no `Any` in public signatures, deterministic.

## 11. Build order & dependencies
**Built first in Phase 1**, right after the foundation + P0. Depends on: F2 (`TurnContext`/`Query`/
config types), F1 (catalog vocab + `canonical_track_id`, only when entity extraction is on), and
**P0** (the `context_cap` + `recency_window` numbers and the tokenizer/explicit-metadata facts).
**Blocks:** R2 (refines R1's `Query` → `.structured`), and R3/R4/R5/R6 (every channel consumes the
`Query`), hence R7 fusion. On the critical path
`F1+F2+F3 → P0 → R1+R3 → R7 → L1 → responder → D1` (plan §17, `000_INDEX.md`). R1 can be built in
parallel with A1/R3 once P0's context cap lands.
