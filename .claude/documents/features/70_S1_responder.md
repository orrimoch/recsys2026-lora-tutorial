# S1 — Responder (LLM, grounded)

> Phase-4. Generates the `predicted_response` — the text the Gemini judge scores on Personalization +
> Explanation Quality, plus Distinct-2. **Decoupled from retrieval scoring**: it reads only the final
> top tracks + context, so it develops in parallel and never affects nDCG. See `000_INDEX.md`.

## 1. Purpose
Produce a grounded, personalized, varied explanation of the recommendations — maximizing the LLM-as-Judge dimension and Distinct-2 without overfitting the (secret) judge prompt.

## 2. Interface / contract
Lives in `mcrs/lm/responder.py`. Implements F2 `Responder`.

```python
class Responder:                            # implements F2 Responder
    def __init__(self, model_revision: str, cfg: "ResponderConfig"): ...
    def respond(self, ctx: TurnContext, top_tracks: list[dict]) -> str: ...   # grounded response; may be ""→ but aim non-empty
```

**Wiring:** input = `TurnContext` (profile, goal, last utterance + dialogue gist) + the final top tracks (names/artists/tags, from L1's ids resolved via F1 `Catalog.metadata`); output = `SubmissionRow.predicted_response` (set by D1). Cached per (session, turn).

## 3. Dependencies
F2 (`Responder`, `TurnContext`), F1 (`Catalog.metadata` to render track names/artists/tags; `UserProfile`), F3 (Distinct-2 + proxy-judge harness). Model: a hosted lite LLM (Gemini-lite, e.g. `gemini-2.5-flash-lite` — **verify current id**) via API key, OR an open-weight LoRA-fine-tuned responder (its own §6.1 notebook → Hub adapter). Salvage: `gemini_responder.py`, `precheck_prediction.py` (response checks). No retrieval coupling.

## 4. Design & logic
- **Grounding inputs:** user profile (age/gender/country, preferred_musical_culture), conversation goal, last utterance + brief dialogue gist, and the **actual** top recommended tracks (names/artists/tags). Grounding in real retrieved tracks + user specifics is what Personalization rewards.
- **Prompt design:** system prompt = music-curator persona + domain knowledge; explicit instructions (justify picks via the user's stated mood/genre/history; be specific; one coherent paragraph); **few-shot** examples spanning cold/warm + different goals; light **CoT** ("first note the intent, then map each pick"). Keep responses **varied** (anti-template — reference concrete track/artist/mood details, not reusable filler) to help Distinct-2.
- **Secret-judge caution:** the judge prompt is not published — optimize *genuine* quality; iterate against a clearly-labeled **proxy judge** (our own LLM call with a reasonable rubric) + a small human eyeball check; never overfit a guessed rubric.
- **Model choice:** default = hosted Gemini-lite (strong out-of-box, large context, cheap, no GPU); an open-weight LoRA responder is a gated alternative for full control/reproducibility (the open-weight-for-what-we-ship principle) — adopt only if it matches/beats the proxy-judge score.
- **Injection safety:** sanitize track names/utterances before templating (they're untrusted text). Length cap; non-empty guarantee (fallback template if the model returns empty/degenerate).

## 5. Reuse
Port `salvage/scripts/gemini_responder.py` (prompt + batched/cached Gemini calls) → adapt to the F2 `Responder` interface + S1 grounding inputs. **Port + adapt.** LoRA responder path = a new §6.1 notebook if pursued.

## 6. Eval & acceptance gate
Via the F3 proxy-judge + Distinct-2: **proxy Personalization/Explanation up vs. a plain baseline** and **Distinct-2 ≥ 0.2558** (the baseline floor); every response non-empty & within length. The official LLM-judge score is the real target but unavailable offline — the proxy is explicitly labeled and not over-trusted.

## 7. Tests
- Prompt-builder is **deterministic** (fixed inputs/seed) & **injection-safe** (a track name with prompt-like text is sanitized, doesn't alter instructions).
- Response non-empty & within length; empty-model-output → fallback template fires.
- Distinct-2 computed via F3 official function; proxy-judge harness runs on a fixture.
- Grounding: the response references at least the top track/artist + a user-specific detail (a check, not a hard gate).
- Train==serve: pinned `model_revision` + prompt; cache key embeds revision+prompt hash.

## 8. Failure modes & guards
- **Templated/boilerplate responses** tanking Distinct-2 → anti-template prompt + variation check.
- **Overfitting a guessed judge rubric** → optimize genuine quality; proxy clearly labeled; human spot-check.
- **Prompt injection** via track names/utterances → sanitize before templating.
- **Cost/quota spike** → only generate required turns, lite model, short outputs, batch, cache, budget cap, free tier first (§15).
- **Train/serve prompt skew** → pin revision + prompt; D1 records the hash.
- **Empty/degenerate output** → fallback template guarantees a valid non-empty string.

## 9. Config knobs
`responder.backend` (`gemini`|`open_lora`), `responder.model_revision`, `responder.max_tokens`, `responder.temperature`, `responder.few_shot_set`, `responder.top_n_for_prompt` (how many tracks to ground on), `responder.cache`, `responder.proxy_judge.{enabled,model,rubric}`, `responder.budget_cap`. Defaults/types from F2 loader.

## 10. Definition of Done & review checklist
- [ ] Implements F2 `Responder`; grounded in real top tracks + profile + goal.
- [ ] Deterministic, injection-safe prompt builder; non-empty + length guarantees; fallback covered.
- [ ] Proxy Personalization/Explanation up vs baseline; Distinct-2 ≥ 0.2558 (via F3).
- [ ] Caching + cost cap in place; pinned revision/prompt (train==serve); D1 records the hash.
- [ ] Code review approved; decoupled from retrieval (no nDCG impact).

## 11. Build order & dependencies
**Built in parallel with retrieval/rerank** (decoupled — needs only the final top tracks + `TurnContext`). Depends on: F1 (`Catalog.metadata`), F2 (`Responder`), F3 (proxy-judge + Distinct-2). A trivial templated responder unblocks the first end-to-end submission (plan §17 day 3–4); the full LLM responder is the §13 quality lever. **Blocks:** D1 (fills `predicted_response`).
