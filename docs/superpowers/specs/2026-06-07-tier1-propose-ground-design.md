# Tier-1 #3.5 — RAG propose-then-ground recall channel

Date: 2026-06-07
Status: approved (design)

## Context

The only proposed lever that sources candidates from OUTSIDE the collaborative/
content graph: an LLM proposes real artist+title suggestions that fit the intent
(explicitly including NEW artists in the same style), then each proposal is
grounded to a real catalog track by dense nearest-neighbor. The LLM's parametric
knowledge of artist similarity / genre scenes is signal no channel in the stack
has — directly attacking the new-artist wall. Distinct from the closed SID work
(that decoded learned tokens with no world knowledge); here we decode real
artist/title strings and ground by embedding NN.

## Decisions (approved)

- Backend: local Qwen2.5-7B-Instruct via LLAMA_MODEL (same as HyDE / 3.1b).
- Integration: new opt-in union channel `propose_ground` (HyDE-style:
  generate -> per-proposal dense NN -> RRF), default off.

## Architecture (mirrors HyDE / structured_query)

- `mcrs/query_rewriters/propose_ground.py`
  - `parse_proposals(text) -> list[str]` — pure; parse numbered or JSON list of
    "Artist - Title" proposals; tolerant of prose/fences; [] on failure.
  - `ProposeGenerator(lm, prompt_path, cache_dir, n_proposals, batch_size)` —
    `generate_batch(queries) -> [{"proposals": [...]}]`; hash-cached; batched
    greedy decode (HydeGenerator idiom).
- `mcrs/retrieval_modules/propose_ground_channel.py`
  - `ProposeGroundRetriever(generator, inner_dense, topk_per_proposal=100, rrf_k=60)`
    — generate proposals -> inner dense NN per proposal -> RRF per query -> top-K.
    Standard retriever interface.
- `mcrs/system_prompts/propose_tracks.txt` — strict list of real artist+title
  suggestions matching the intent, including new artists in the same style; no
  prose. Leakage-guarded (never references thought / goal_progress).
- `_wrrf_union_v1_specs` gate `use_propose_ground` (+ w_propose_ground default 0.5,
  pg_model, n_proposals); `load_retrieval_module` branch builds LLAMA_MODEL +
  ProposeGenerator + inner dense -> ProposeGroundRetriever.

## Leakage guard

Channel sees only the retrieval-query string (excludes thought/goal_progress).
Prompt must never reference those. Tested. Watch the LLM training cutoff: it must
propose by STYLE, not recognize specific golds — style proposals don't leak gold
ids.

## Regression safety (must not break Tier 0 / prior Tier-1)

- All opt-in/new files; default off -> config 194 + base union + prior files
  bit-identical. Full-suite gate after. Tier-end QA re-runs prior-tier guards.

## Tests (TDD, no LLM)

- parse_proposals: numbered list; JSON list; prose-wrapped; malformed -> [];
  strips numbering/fences.
- ProposeGroundRetriever: fake generator + fake inner -> RRF-fused grounded ids;
  batch span alignment preserved.
- union gating: propose_ground off by default; appended at w_propose_ground.
- prompt excludes thought / goal_progress.

## Validation gate (Colab, no retrain)

nb74 cell: build the channel (loads Qwen2.5-7B), report recall@100 turn-1 +
new-artist-subset vs baseline. Kill criterion: if < ~20% of proposals ground to
in-catalog tracks (above a cosine threshold) the channel is too sparse to matter.
Proceed to fold into the combined retrain only if turn-1 / new-artist recall lifts.

## Non-goals

No retrain here. wants_new_artist-style routing and an explicit grounding cosine
threshold are later refinements (v1 relies on dense NN + RRF, like HyDE).
