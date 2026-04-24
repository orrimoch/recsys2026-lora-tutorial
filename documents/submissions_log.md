# CodaBench Submissions Log — RecSys 2026 Blind A

One row per submission. Sorted oldest → newest. Scores are what CodaBench displayed on the leaderboard after scoring.

| Submission ID | Date (local) | Tid / approach | nDCG@20 | Cat Div | Lex Div | LLM (1–5) | Composite | Rank | Notes |
|---|---|---|---|---|---|---|---|---|---|
| **v1** | 2026-04-19 16:51 | `naive_bm25_blindset_A` — BM25 only, templated response `"You might enjoy X by Y based on what you just told me."` | 0.19 | 0.03 | 0.46 | 1.3 | **0.17** | 6 | Retrieval was already competitive (top-3 nDCG@20). LLM judge dragged score — templated one-liner. |
| **v2** | 2026-04-19 17:44 | `qwen1.5b_bm25_blindset_A` — BM25 + **Qwen2.5-1.5B-Instruct** (MPS float32, greedy, max_new=96). Stock `response_generation.txt` prompt, top-1 track fed as fake assistant turn. User profile + conversation goal appended. | 0.19 | 0.03 | 0.72 | 1.8 | **0.23** | 5 | Lexical diversity +0.27. LLM score only +0.5 because: (a) prompt tells model to apologize on mismatches → "I'm sorry" responses; (b) chat template pre-pends track as assistant message → "I'm glad you enjoyed X" hallucinations; (c) 9/80 responses cut off at 96 tokens. |
| **v3** | 2026-04-19 18:07 | `qwen3b_bm25_blindset_A` — BM25 + **Qwen2.5-3B-Instruct** (MPS float32, greedy, max_new=192). Custom `response_generation_v2.txt` (no apology, no hallucinated history). **Top-3 tracks in system prompt** with metadata. Chat template = [system, user] only (no fake assistant turn). User profile + conversation_goal injected. | 0.19 | 0.03 | 0.66 | 2.7 | **0.29** | **4** | LLM score +0.9 vs v2; retrieval unchanged. Local diagnostics: 0 apologies, 0 hallucinations, 0 truncations (v2 had all three). |
| **v4** | 2026-04-19 19:xx _(in flight)_ | `qwen3b_bm25tags_blindset_A` — **BM25 corpus includes `tag_list`** (5 fields, cache reused). **LLM query expansion** stage-1 rewrites user_query into dense keyword query (genres/moods/eras/synonyms) with `no_repeat_ngram_size=3, repetition_penalty=1.2`. **Few-shot exemplars** (3 examples) added to response prompt (`response_generation_v3.txt`). Retrieval input = expanded_query + original_query. | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | Smoke results: expansions are useful ("danceable upbeat disco funk latin" etc.). Full run ~20 min. |

## Schema notes
- **Composite** = `0.50·nDCG@20 + 0.10·CatDiv + 0.10·LexDiv + 0.30·(LLM-1)/4`
- **Rank** = position on the CodaBench Blind A leaderboard at submission time (rank can shift as others submit).
- Missing-from-log (never submitted) runs belong in `experiments/ledger.tsv`, not here.

## Files per submission

### v1 — naive BM25
- **Config:** `music-crs-baselines/config/naive_bm25_blindset_A.yaml`
- **Inference script:** `music-crs-baselines/run_inference_blindset_retrieval_only.py`
- **Output JSON:** `music-crs-baselines/exp/inference/blindset_A/naive_bm25_blindset_A.json`
- **Packaged zip:** overwritten by v2 (JSON still recoverable).
- **Runtime:** < 1 min (CPU, BM25 only).

### v2 — BM25 + Qwen 1.5B
- **Config:** `music-crs-baselines/config/qwen1.5b_bm25_blindset_A.yaml`
- **Inference script:** `music-crs-baselines/run_inference_blindset_full.py`
- **Output JSON:** `music-crs-baselines/exp/inference/blindset_A/qwen1.5b_bm25_blindset_A.json`
- **Packaged zip:** overwritten by v3 (JSON still recoverable).
- **Runtime:** ~5 min on M4 MPS (float32, greedy, batch=2).
- **Prompt files used:** stock `mcrs/system_prompts/{roleplay,response_generation,personalization}.txt`.

### v3 — BM25 + Qwen 3B + v2 prompt + top-3 context
- **Config:** `music-crs-baselines/config/qwen3b_bm25_blindset_A.yaml`
- **Inference script:** `music-crs-baselines/run_inference_blindset_full_v2.py`
- **Output JSON:** `music-crs-baselines/exp/inference/blindset_A/qwen3b_bm25_blindset_A.json`
- **Packaged zip:** overwritten by v4 (JSON still recoverable).
- **Prompt files used:** `mcrs/system_prompts/roleplay.txt` + `response_generation_v2.txt`.
- **Key code change:** `custom_batch_generate()` — [system, user] chat template only, no fake assistant turn carrying the recommendation.
- **Runtime:** ~15 min on M4 MPS (float32, greedy, batch=1).

### v4 — BM25+tags + query expansion + few-shot response
- **Config:** `music-crs-baselines/config/qwen3b_bm25tags_blindset_A.yaml` (5-field BM25 corpus including `tag_list`, cache hit).
- **Inference script:** `music-crs-baselines/run_inference_blindset_full_v3.py`.
- **Output JSON:** `music-crs-baselines/exp/inference/blindset_A/qwen3b_bm25tags_blindset_A.json`.
- **Expansions debug file:** `music-crs-baselines/exp/inference/blindset_A/qwen3b_bm25tags_blindset_A.expansions.json` (per-row user_query → expanded_query mapping).
- **Packaged zip:** `music-crs-baselines/exp/inference/blindset_A/prediction.zip` (after packaging).
- **Prompt files used:** `mcrs/system_prompts/{roleplay.txt, query_expansion.txt, response_generation_v3.txt}`.
- **Key code changes:** (1) two-stage LLM — query expansion before retrieval, then response generation. (2) `no_repeat_ngram_size=3` + `repetition_penalty=1.2` on query-expansion pass to prevent degenerate repetition. (3) BM25 input concatenates expanded query + raw user query so literal artist/song names are preserved.
- **Runtime:** ~20 min on M4 MPS (float32, greedy, batch=1, 2 LLM passes per row).

## Next ideas (from earlier analysis, unshipped)

| Idea | Expected nDCG@20 gain | Expected LLM gain | Risk |
|---|---|---|---|
| BM25 corpus + `tag_list` field (cache already built as `track_name_artist_name_album_name_release_date_tag_list`) | +0.01–0.03 | neutral | low |
| RRF(BM25, dense) — configs 007/009/010 exist | +0.02–0.05 | neutral | medium (dense embeddings only cover 25% of tracks per data_exploration.md — need BM25 fallback) |
| Cross-encoder rerank of top-40 → top-20 (BGE-reranker-base) | +0.005–0.02 | neutral | low (slow on M4, ~30 min for 80 rows) |
| Feed user_profile fields parsed (age, country, goal) as separate prompt sections | — | +0.2–0.4 | none |
| Try Qwen2.5-7B-Instruct (Colab GPU) | — | +0.3–0.7 | model size — CPU/MPS too slow, needs Colab |
| Query expansion: let LLM rewrite the user_query into a richer BM25 query before retrieval | +0.01–0.03 | +0.1–0.3 | medium (doubles LM runtime) |

## How to add a new entry

1. Run the new inference variant.
2. Package as `prediction.zip` (single `prediction.json` at root).
3. Upload to CodaBench and wait for the scorer.
4. Once scored, add a row to the table above with the new tid, numbers, rank, and notes.
5. Add a "Files per submission" entry pointing to the config + script + output JSON.
6. If the approach opens a new branch of ideas, append to "Next ideas".
