# Tier-1 #3.1b — LLM structured-query extraction channel

Date: 2026-06-07
Status: approved (design)

## Context

Cold-firable recall (Tier 1). The dense channels embed the raw dialogue, which is
noisy and artist-biased on pivot turns ("play something NEW like X"). An LLM can
distil each turn into a clean, content-only query (genres/moods/era/culture/intent
+ a wants_new_artist pivot flag), which embeds closer to the gold's content and
de-emphasizes the seed artist — surfacing new-artist golds the raw-query dense
channel misses. Mirrors the existing HyDE channel pattern.

## Decisions (approved)

- Backend: local Qwen2.5-7B-Instruct via LLAMA_MODEL (same as HyDE) — self-
  contained at serve, no API dependency on the Blind run.
- Integration: new opt-in union channel `structured_query` (HyDE-style), gated
  use_structured_query (default off). Composes with 3.1a/3.2.

## Module layout (new, all opt-in)

- `mcrs/query_rewriters/structured_query.py`
  - `parse_structured_json(text) -> dict` — tolerant JSON parse; missing/malformed
    -> safe defaults; validated fields only.
  - `assemble_query(fields) -> str` — content-only query string; omits empty
    fields; NO artist names (artist-agnostic by construction).
  - `StructuredQueryExtractor(lm, prompt_path, cache_dir, batch_size)` —
    `extract_batch(queries, session_ids, turn_numbers)`; per-(session,turn) JSON
    cache (cmqr/HyDE idiom); temp 0.
- `mcrs/retrieval_modules/structured_query_channel.py`
  - `StructuredQueryRetriever(extractor, inner_dense, ...)` — synthetic query ->
    inner dense retrieve -> ranked list. Standard retriever interface
    (batch_text_to_item_retrieval / text_to_item_retrieval).
- `mcrs/system_prompts/structured_query.txt` — extraction prompt (strict JSON).
- `_wrrf_union_v1_specs`: `if ec.get("use_structured_query")` -> append
  {type: "structured_query", weight: w_structured_query default 0.5}.
- `load_retrieval_module` branch `structured_query`: build LLAMA_MODEL +
  StructuredQueryExtractor + inner dense (dense_metadata_qwen3, like HyDE) ->
  StructuredQueryRetriever.

## JSON schema (LLM output)

```
{"genres": [str], "moods": [str], "era": str, "culture": str,
 "intent": str, "wants_new_artist": bool}
```

## Leakage guard

The channel sees only the retrieval-query string, which already excludes
`thought` and `goal_progress_assessments` (not in session_memory). The prompt
must never reference those fields. Tested.

## Regression safety (must not break Tier 0 / 3.1a / 3.2 / 3.4)

- All opt-in, default off -> config 194 + _wrrf_union_v1_specs base list + all
  prior files bit-identical.
- Full test suite after change: prior tests green; only the 4 known pre-existing
  unrelated failures remain.
- Tier-end QA re-runs prior-tier guard tests.

## Tests (TDD, local, no LLM)

- parse_structured_json: valid JSON; JSON embedded in prose; malformed -> defaults;
  type coercion (genres always list, wants_new_artist always bool).
- assemble_query: omits empty fields; no artist names; content-only ordering.
- StructuredQueryRetriever: with a fake extractor + fake inner, returns the inner's
  ranking for the synthetic query; batch alignment preserved.
- union gating: use_structured_query off by default; appends one channel at
  w_structured_query when on.
- prompt file excludes 'thought'/'goal_progress'.

## Validation gate (Colab, no retrain)

Run the extractor over dev turns (cache), build the structured_query channel, and
report recall@100 turn-1 vs baseline (and fused into union+SASRec). Proceed to
offline-cache 122k train turns + feature rebuild + retrain ONLY if turn-1 recall
lifts beyond noise.

## Non-goals

No offline 122k cache run here (Colab). No retrain. No serve config change.
wants_new_artist is informational in v1 (assembled query is already artist-free);
weighting/pivot-routing is a later increment.
