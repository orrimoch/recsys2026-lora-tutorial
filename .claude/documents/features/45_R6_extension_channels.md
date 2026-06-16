# R6 — Extension Channels (CLAP / related-artist / propose-ground / SASRec) [gated]

> Phase-1 retrieval module. A bundle of **orthogonal, wall-cracker** candidate generators that may reach
> golds the core channels (R3 BM25, R4 dense-text, R5 content-kNN/CF/same-artist) structurally miss —
> by **acoustics** (CLAP), **cross-session artist co-occurrence** (related-artist), **LLM-proposed seeds**
> (propose-ground), **LLM pseudo-docs/structured queries** (HyDE / structured-query), and **sequential
> session state** (SASRec). **Every channel here is GATED**: it is wired into R7 only on a *measured
> unique-recall lift* on dev (plan §7.3 req #2) — none is pre-judged, none is assumed. Each implements the
> F2 `RetrievalChannel` contract and emits canonical ids only; R7 fuses, R6 never fuses. SASRec is the one
> channel here that needs a **trained artifact** (a phase notebook per §6.1). See `000_INDEX.md`.

## 1. Purpose
Provide the gated, orthogonal extension retrievers that attack the **recall wall** P0/R7 measure — the
golds that BM25/dense/content-kNN/CF/same-artist all miss (prior work: ~96% of those are *new-artist*
golds the session channels cannot reach). Each R6 channel adds candidates by a signal the core channels
lack; the module's whole job is to **earn or be dropped** on per-channel unique-recall, not to ship every
clever idea. Out of scope (these are **R5**): content-kNN-from-history, CF (`user_emb·track_emb`), and
same-artist — R6 is strictly the *extension* set beyond those.

## 2. Interface / contract
Lives in `mcrs/retrieval_modules/` (new package, ports from salvage). Every R6 retriever is a F2
`RetrievalChannel`: it consumes the shared `Query` (or a `per_channel` override) and returns **canonical**
`track_id`s over `all_tracks`; it does **not** fuse, score-calibrate, or know about other channels.

```python
# F2 RetrievalChannel surface (verbatim shape — salvage already conforms):
class <Channel>Retriever:           # implements F2 RetrievalChannel
    label: str                      # unique fusion label, e.g. "clap_text", "related_artist", "sasrec"
    def batch_text_to_item_retrieval(
        self, queries: list[str], topk: int,
        batch_context: list[dict] | None = None, user_ids: list[str] | None = None,
    ) -> list[list[str]]: ...       # canonical track_ids; absent = not retrieved; [] for a no-signal turn
    def text_to_item_retrieval(self, query, topk, user_id=None) -> list[str]: ...  # single-row convenience
```

**`batch_context` payload (R1 → channel, per row).** R6 channels need more than the query string;
R1 supplies a per-row dict so they stay causal and parity-aligned:
- `played_tids` / `history_tids` — canonical played/listening ids **≤ t** (CLAP-audio recall, related-artist, SASRec).
- `user_dialog` — the **user-turns-only** dialogue text (SASRec was trained on this; missing → loud warn + raw-query fallback, train/serve skew).
- `session_id`, `turn_number` — cache keys for the LLM-backed channels (propose-ground, HyDE, structured-query).
- `segment` — `"cold"|"warm"` (F1) for the cold/warm gate split.

**Channels in this module (each gated; one row per channel in §6 keep/drop table):**

| Label | Signal / mechanism | Cold-firable? | Needs trained/LLM artifact |
|---|---|---|---|
| `clap_audio` (a.k.a. clap_recall) | mean-pool **played** tracks' CLAP audio → catalog audio-NN ("sounds like what they played") | **No** (needs history) | pure-compute (precomputed `audio-laion_clap`) |
| `clap_text` | encode **query text** in the CLAP joint space → audio-NN ("sounds like what they asked for") | **Yes** | CLAP text tower (`laion/larger_clap_music`) — model, no training |
| `related_artist` | cross-session **artist co-occurrence** (train) → new co-occurring artists' top tracks | No (needs played artists) | pure-compute (co-occurrence map, built once) |
| `propose_ground` | **LLM proposes** "Artist – Title" seeds → ground each to catalog by dense-NN → RRF | Yes | **LLM** (Gemini/Qwen) + dense inner retriever |
| `hyde` | **LLM writes pseudo-track docs** → dense-NN in the catalog's own space → RRF | Yes | **LLM** + dense inner retriever |
| `structured_query` | **LLM extracts a content-only synthetic query** → dense-NN | Yes | **LLM** + dense inner retriever |
| `sasrec` | dialog + played history → **SASRec session state** → score frozen item-repr matrix | partial (cold-turn = dialog-only state) | **TRAINED** SASRec checkpoint (phase notebook, §6.1) |

**Wiring:** `R1/R2 → Query (+ batch_context) → each R6 channel → list[list[track_id]] → R7` fuses with the
channel's configured `weight`/`topk_internal`/`query_key`. The same channel objects later expose features
to **K1** (e.g. SASRec score / rank, CLAP cosine) — built once, used in both stages, train==serve.

## 3. Dependencies
- **Modules:** F1 (`canonical_track_id`/`canonical_track_ids`, `Catalog`, `TrackEmbeddings.matrix("audio-laion_clap")`, `Conversations`/`Users`), F2 (`RetrievalChannel`, `Query`, `TurnContext`, config), R1 (`Query` + `batch_context`), R4 (the **dense inner retriever** that propose-ground/HyDE/structured-query ground into), F3 (`recall_at_k`, unique-recall), P0 (cold/warm threshold, channel-keep priors, `topk_internal`/fusion-K), R7 (fuses).
- **Data/assets:** precomputed `audio-laion_clap` track embeddings (F1, modality matrix); the **artist co-occurrence map** built once from the **train** split + cached (`cache/related_artist/artist_cooc.pkl`); catalog `artist_name`/`popularity` (related-artist track ranking).
- **Models:** CLAP text tower `laion/larger_clap_music` (`ClapTextModelWithProjection`, transformers — *no training*, GPU optional); a **SASRec checkpoint** (trained here, on Hub); the LLM for propose-ground/HyDE/structured-query (Gemini-lite or open Qwen — competition-permitted, §6.2); the SASRec context encoder `BAAI/bge-base-en-v1.5` + frozen item modalities (`metadata-qwen3_embedding_0.6b`, `audio-laion_clap`).
- **External APIs:** Gemini-lite (LLM-backed channels) — cost-capped + cached (§4.6, §15). HyDE/structured-query/propose-ground can also run an open Qwen3 locally.
- **Libs:** `numpy`, `torch` (SASRec, CLAP), `transformers`/`sentence-transformers`, `datasets` (co-occurrence build). GBDT-free.

## 4. Design & logic

### 4.1 Common contract & invariants (all R6 channels)
- **Canonical ids at the boundary.** Every list returned passes through `canonical_track_id` *before* return; F3/R7 assert `⊆ catalog`. A channel that derives ids (related-artist from artist→tids, propose-ground from grounded NN) canonicalizes on the way out (plan §7.3 req #1). Channels must **not** lean on salvage `strip_track_id_prefix` (a doc-text helper, not an id normalizer).
- **No-signal turn → `[]`.** A row with no usable signal (cold turn for CLAP-audio/related-artist/SASRec-history) returns an empty list, contributing **0** to fusion — never a degenerate or padded list. This is what makes a channel safely cold/warm-gated by P0.
- **Pull depth.** Each channel pulls to its `topk_internal` (P0-sized, ≥ fusion-K, target 300–500) so a gold it only surfaces at rank ~300 still reaches fusion (§7.3.1 trap). For the composed LLM channels, `topk_per_doc/proposal ≈ 100` per seed then RRF up to `topk`.
- **Determinism.** Fixed model revisions + fixed tie-break (sort by score then id); LLM channels are deterministic *through the cache* (same prompt revision ⇒ same cached output).

### 4.2 CLAP audio (`clap_audio`) and CLAP text (`clap_text`) — the acoustic levers
- **`clap_audio`:** mean-pool the L2-normed `audio-laion_clap` vectors of the session's **played** tracks → one session query → cosine-NN over the catalog audio matrix, drop played, take `topk`. Reaches "more tracks that *sound* like what they've been playing." **Warm-only** (no played CLAP vector ⇒ `[]`). Prior probe: rescued ~269 union-missed *new-artist* golds, +~0.035 recall@100 ceiling — a wall lever, but **must re-earn** its gate here.
- **`clap_text`:** encode the **query text** into the CLAP joint space via `ClapTextModelWithProjection.text_embeds` (the projected text embedding in the audio space — *not* `get_text_features`, whose dtype varies across transformers versions) → cosine-NN over the same `audio-laion_clap` matrix. **Cold-firable** (reads query, not history) — fires on turn-1/Blind where `clap_audio` is silent. Reaches new-artist wall golds by *acoustics*, a signal the lexical/metadata/late-interaction text channels structurally lack.
- **Caution (prior finding):** `clap_text` was previously flagged as **RRF noise** in some fusion sweeps — exactly why R6 ships it *only* on a measured unique-recall lift (§6), and why R7's keep-list / weight sweep is the arbiter. CLAP-as-rerank-*feature* (`clap_session_sim`) was rejected for an in-sample leak; the **recall channel adds candidates** (not reordering an existing pool) so it is not affected by that leak — keep the two uses distinct.
- Query/document share the L2-normed joint space so dot product = cosine; both towers/matrices loaded once (module-level cache).

### 4.3 Related-artist (`related_artist`) — cross-session co-occurrence
- **Build once (train, cached):** for each train session, de-dup its played artists, then `cooc[a][b] += 1` for every ordered pair (one count per shared session, replays don't inflate). Pickle to `cache/related_artist/artist_cooc.pkl`; serve loads it. Built from **train only** (no dev/blind) — no leak.
- **Serve:** from the session's seen artists, expand to **co-occurring NEW artists** (not already in the session), summed by count; emit those artists' most-popular tracks (minus played), to `topk`. This reaches *new* artists — the wall same-artist (session artists only) and SASRec structurally cannot. Prior probe: ~29% of union-missed dev golds reachable at top-100 co-occurring artists (an order of magnitude over other levers) — still **gated** here.
- Cold turn (no played artists) ⇒ `[]`.

### 4.4 Propose-ground / HyDE / structured-query — LLM-as-retrieval (§6.2)
All three are **composed**: an LLM generator/extractor + a **dense inner retriever** (R4 over a Qwen3 field) so LLM output is grounded in the catalog's *own* embedding space.
- **`propose_ground`:** LLM proposes real "Artist – Title" tracks; each proposal is grounded to catalog tracks by dense-NN (`topk_per_proposal ≈ 100`); per-query proposal lists are **RRF-fused** (`rrf_k≈60`) → `topk`. Cold-firable.
- **`hyde`:** LLM writes pseudo-track **descriptions** (HyDE docs) for the dialogue; each doc → dense-NN in the catalog space → RRF-fused. Falls back to the intent query if no docs. Cold-firable.
- **`structured_query`:** LLM extracts a **content-only synthetic query** (`{positive/negative attrs, mood, genre, era, seed_artists}` flattened to text) → single dense-NN pull. Cold-firable. (R2 may already produce `Query.structured`; this channel is the *retrieval* use of it — reuse R2's extractor + cache, don't double-call the LLM.)
- These bridge the **conversational↔metadata vocabulary gap** the core text channels miss; they are the per-turn LLM levers of §6.2.

### 4.5 SASRec (`sasrec`) — sequential session state (the trained channel)
- **Model:** content-fused, dialog-conditioned SASRec. Input per row = (a) `user_dialog` (user-turns-only) encoded by `BAAI/bge-base-en-v1.5`, (b) the played-history item-feature sequence (frozen `metadata-qwen3` ⊕ `audio-laion_clap`, last `max_len≈50`, unknown ids dropped) → `model.encode(ctx_emb, feats, lengths)` → session state → `model.score(state, item_repr)` over the **frozen catalog item-repr matrix** → top-`topk`. CF-bpr was dropped from item feats (structurally inert for the new-artist majority).
- **Train/serve parity (hard):** trained on **user-turns-only** dialog; callers MUST pass `batch_context['user_dialog']`. Missing ⇒ loud `warnings.warn` + raw-query fallback (works but skews). This was a real prior bug — guard with a test.
- **Trained artifact (§6.1):** `scripts/train_sasrec.py` → its own phase notebook. Train + per-epoch val on the **train** split with a **session-disjoint** val slice (deterministic via `SHA1(session_id)`); model selection on `val_recall@100`; the **test/dev** split used **once** at the end for a standalone number, never for selection. **LoRA-free** (SASRec is not a foundation model — only the frozen bge context encoder + frozen item modalities, the SASRec head trains from scratch); checkpoint = `dict(state_dict, model_kwargs, item_feats, track_ids)` pushed to the **HF Hub** by revision (§6.1 saving discipline). Full-softmax next-item CE.
- **Dual use:** the same checkpoint later powers a **K1 rerank feature** (SASRec in-pool score/rank). Built once here, consumed in both stages (train==serve). `train_sasrec.py` exposes **OOF cross-fit** (`--oof-fold`/`--oof-num-folds`) so the K1 SASRec feature can be built **leak-free** on train (a candidate's SASRec score comes from a fold the row didn't train on).
- **Cost note:** SASRec/CLAP-text generation wants GPU; serve-time scoring is CPU-viable (frozen item-repr matrix, `batch_size≈256`). The LLM channels need GPU/API only at *generation* time — cached outputs serve from disk with no model loaded.

### 4.6 No-leak, caching & cost
- **No-leak:** every input is `≤ t` — `played_tids`/`history_tids` exclude the turn-`t` gold and any future turn (F1 guarantees); the co-occurrence map is **train-only**; SASRec val is session-disjoint; LLM channels see only utterances `≤ t` and never the gold (§6.2 guardrail). The gold is used *only* to score recall (F3), never an input.
- **Caching (the LLM channels):** propose-ground/HyDE/structured-query are **cached per `(session_id, turn_number, prompt_revision)`** content hash — re-runs are free and deterministic. CLAP text-tower outputs and the SASRec item-repr matrix are loaded once (module-level cache). The artist co-occurrence map is a one-time train pass cached to disk.
- **Cost (§15):** LLM channels are **cost-capped + dry-run-counted** before any full-dev pass; per-turn ≈ 8k dev turns × (1 small generation) — lite model, short outputs, batched. Prompt-injection-safe (sanitize track names/utterances before templating). Train==serve LLM revision + prompt + truncation. A channel that doesn't move unique recall is **dropped** (it only adds latency/cost otherwise, §6.2).

## 5. Reuse
Port **behind the gate** from `salvage/mcrs/retrieval_modules/` (plan §6.3 asset map — "Port behind gate"); each already matches the F2 `batch_text_to_item_retrieval` shape, so porting is canonicalization + F2 typing + wiring, not reshaping.

| Channel | Salvage source | Call |
|---|---|---|
| `clap_audio` | `clap_recall.py` (+ `clap_similarity.py` `clap_session_query`/`load_clap_lookup`) | **Port** — swap `load_clap_lookup` for F1 `TrackEmbeddings.matrix("audio-laion_clap")`; canonicalize output. |
| `clap_text` | `clap_text.py` (`ClapTextRetriever`, `_load_clap_text_encoder`, `cosine_topk`) | **Port** — keep the module-level encoder cache + `ClapTextModelWithProjection`; F1 audio matrix. |
| `related_artist` | `related_artist.py` (`RelatedArtistRetriever`) | **Port** — keep the cached co-occurrence build; route catalog access through F1 `Catalog`; canonicalize. |
| `propose_ground` | `propose_ground_channel.py` (+ `query_rewriters/propose_ground.py` generator) + `hyde_qwen3.py` `rrf_fuse` | **Port behind gate** — inner = R4 dense; add `(session,turn)` cache + cost cap. |
| `hyde` | `hyde_qwen3.py` (`HydeQwen3Retriever`, `rrf_fuse`) + `query_rewriters/hyde.py` | **Port behind gate** — same. |
| `structured_query` | `structured_query_channel.py` + `query_rewriters/structured_query.py` | **Port behind gate** — reuse R2's extractor/cache. |
| `sasrec` | `sasrec_seq.py` (`SasrecRetriever`), `sasrec_model.py` (`SasrecModel`, `build_user_dialog`, `prior_turns`, `next_item_loss`, `apply_item_feats_mode`) | **Port behind gate.** |
| SASRec training | `scripts/train_sasrec.py` | **Port** into a §6.1 phase notebook (session-disjoint val, Hub save). |
| RRF helper | `hyde_qwen3.rrf_fuse` (intra-channel seed fusion) | **Reuse** (distinct from R7's cross-channel RRF). |

**Do not port** the rejected feature variants (`clap_similarity` *reranker* feature, `sequential_rerank`) — R6 is recall channels only; rerank features are K1.

## 6. Eval & acceptance gate
**Gate (per-channel, on dev via F3 — plan §7.3 req #2):** a channel is **kept** (wired into R7 with non-zero weight) **iff** it adds **unique recall** the kept core channels (R3/R4/R5) don't already cover — measured as the channel's per-channel **unique-recall contribution** (golds it hits that no other kept channel hits) and the **fused recall@{20,200} lift** when it is added. Reported **overall + cold + warm + turn-1**. A channel with **no unique recall is dropped or zero-weighted** (it injects false top ranks that displace golds — RRF is not free to add to).

**Procedure:** reuse P0's cached per-turn id-lists + F3 `recall_at_k`; add each R6 channel to the kept set and re-fuse (R7 `fuse_per_sub`, cheap) → keep iff Δ fused recall ≥ a logged threshold *and* unique-recall > 0 on the segment it targets (e.g. `clap_text`/HyDE earn their place on **cold/turn-1**; `clap_audio`/related-artist/SASRec on **warm**). **No channel is pre-judged** — every one runs a clean current ablation; prior probe numbers are leads, not results.

**SASRec sub-gate (it has a trained artifact):** the SASRec checkpoint passes only if its **standalone** val/dev recall is non-trivial **and** adding it lifts fused recall — selection on session-disjoint `val_recall@100`, never on dev/test. Logged to `reports/experiments.md` + the SASRec phase notebook's eval cell.

## 7. Tests
- **Contract / canonicality:** each channel's output ⊆ canonical catalog id set; a non-canonical/prefixed-id fixture fails; dedup within a row.
- **No-signal → `[]`:** cold turn (no played/history/artist) ⇒ empty list for `clap_audio`/`related_artist`/`sasrec`(history-empty), contributing 0 to fusion.
- **No-leak / causal:** inputs use only `≤ t` (played/history exclude gold + future); co-occurrence map built from train only (assert no dev/blind session id in the build); LLM prompt contains no gold/future turn.
- **CLAP:** `clap_text` cosine matches a hand fixture (L2-normed dot = cosine); `clap_audio` session-pool query = mean of played vectors; played dropped from output.
- **Related-artist:** co-occurrence count = #shared sessions (de-dup per session); seen artists excluded from expansion; popularity-sorted track emission.
- **LLM channels:** generator/extractor mocked → grounding + intra-channel `rrf_fuse` deterministic; **cache hit** returns identical ids with no second LLM call; cost-cap aborts before exceeding budget.
- **SASRec:** group/parity — `user_dialog` missing ⇒ warning fired (train/serve skew guard); session-disjoint val split is deterministic via `SHA1(session_id)`; `_histories_to_feats` last-`max_len`/unknown-id-drop correct; checkpoint round-trips (`state_dict, model_kwargs, item_feats, track_ids`).
- **Determinism:** fixed revision + seed ⇒ identical ranked lists; LLM channels deterministic through the cache.
- **Wiring:** each channel satisfies the F2 `RetrievalChannel` structural check and R7 fuses its output unchanged.

## 8. Failure modes & guards
- **Noise channel injects false top ranks** (the central R6 risk) → the §6 keep/drop gate + R7 recall-non-regression; `clap_text` flagged as prior RRF noise → ship *only* on a measured lift; default weight 0 until earned.
- **Train/serve skew (SASRec):** raw-query fallback when `user_dialog` missing → loud warn + a test; R1 must populate `batch_context['user_dialog']` from `build_user_dialog(prior_turns)`.
- **Non-canonical ids** (related-artist artist→tid, propose-ground grounded NN) → canonicalize at boundary + ⊆-catalog assert.
- **CLAP text-tower API drift** → pin `ClapTextModelWithProjection.text_embeds` (not `get_text_features`); record the model revision.
- **LLM cost/quota spike** → per-turn cache + hard budget cap + dry-run token count + free-tier first (§15); a channel that doesn't earn its gate is dropped to stop paying for it.
- **Prompt injection** via track names/utterances → sanitize before templating (§6.2).
- **Co-occurrence map staleness / leak** → built from train only, cached with the data revision; rebuild on data refresh.
- **Cold-segment collapse** → report turn-1 recall separately; cold/warm gate so a warm-only channel isn't forced onto cold turns.
- **SASRec selection leak** → model selection on session-disjoint val recall@100 only; dev/test touched once at the end.

## 9. Config knobs (under `retrieval.channels[]` + `retrieval.extension.*`; types validated by the F2 loader)
- Per channel (in `retrieval.channels[]`, the R7 spec): `{label, type, weight, topk_internal, query_key, cold_weight, warm_weight, enabled}` — `enabled:false` / `weight:0` keeps a channel built-but-dropped.
- `retrieval.extension.clap.text_model` (default `laion/larger_clap_music`), `clap.audio_modality` (`audio-laion_clap`), `clap.normalize` (true).
- `retrieval.extension.related_artist.cooc_cache` (path), `.cooc_min_count`, `.top_artists`.
- `retrieval.extension.propose_ground.{llm_model, prompt_revision, topk_per_proposal=100, rrf_k=60, cache_dir, max_calls, batch_size}`; same block for `hyde.{topk_per_doc, rrf_k, ...}` and `structured_query.{topk_internal}`.
- `retrieval.extension.sasrec.{checkpoint_revision, ctx_model=BAAI/bge-base-en-v1.5, item_feats_mode, max_len=50, batch_size=256}`.
- Shared LLM: `llm.model`, `llm.revision`, `llm.prompt_revision`, `llm.cost_cap`, `llm.cache_dir`, `seed`, `segment.cold_threshold` (read, not set — P0 owns it).

## 10. Definition of Done & review checklist
- [ ] Each channel implements F2 `RetrievalChannel`, returns canonical ids (⊆ catalog assert), `[]` on no-signal.
- [ ] Per-channel **unique-recall** + fused-lift measured on dev (overall/cold/warm/turn-1) via F3; keep/drop decision logged in `reports/experiments.md` with the threshold that justified it. **No channel kept without a measured lift.**
- [ ] SASRec checkpoint trained in its **own phase notebook** (§6.1): session-disjoint val, selection on val_recall@100, Hub revision recorded; standalone dev number logged.
- [ ] LLM channels (propose-ground/HyDE/structured-query) cached per `(session,turn,prompt_rev)`, cost-capped, dry-run-counted, prompt-injection-safe; train==serve LLM revision/prompt/truncation.
- [ ] No-leak tests green (≤t inputs, train-only co-occurrence, no gold/future in LLM prompt, SASRec selection on val only).
- [ ] All §7 tests green incl. SASRec `user_dialog`-missing warn guard and CLAP-tower API pin.
- [ ] Kept channels' weights/keep-list locked in `config/<exp>.yaml` (train==serve); dropped channels left in code, `enabled:false`.
- [ ] Code review approved; no `Any` in public signatures; salvage ports cleaned (no rejected rerank-feature variants pulled in).

## 11. Build order & dependencies
**Built after the core channels and P0**, in parallel as each earns its gate. Depends on: F1 (ids, catalog, audio matrix, conversations), F2 (contracts), R1 (`Query` + `batch_context`), R4 (the dense inner retriever for the LLM-grounded channels), F3 (recall/unique-recall), P0 (cold/warm threshold + channel-keep priors + `topk_internal`). The **SASRec** channel additionally requires its training phase notebook to produce a Hub checkpoint **before** it can be wired/gated. **Blocks:** R7 (only kept R6 channels enter the fusion keep-list) — and R6 is itself the **gap-router target**: if R7's fused recall@200 < 0.90, the shortfall routes here (and to A1) for an orthogonal wall-cracker channel (R7 §6). SASRec also later feeds **K1** as a rerank feature. Off the strict critical path to the first submission, but the primary lever for clearing the recall wall toward the 0.55 target.
