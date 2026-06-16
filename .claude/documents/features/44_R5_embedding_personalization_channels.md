# R5 — Embedding Personalization Channels (content-kNN from history + CF + same-artist)

> Phase-1 retrieval module. The **warm-user personalization channels**: three independent
> `RetrievalChannel`s that turn the user's listening history (and provided CF user vector) into a
> ranked candidate list, complementing the conversation-driven channels (R3 BM25 / R4 dense-text).
> Each captures "more of what they've been playing"; each **degrades to empty for cold users**, so
> the gating is the load-bearing design decision here. Grounds in plan §7.2.3 (content-kNN from
> history), §7.2.4 (CF), §10 (cold vs warm), and ports salvage `session_history.py` / `session_cf.py`
> / `cf_bpr.py` / `same_artist.py` behind an ablation gate. See `000_INDEX.md` for the catalogue.

## 1. Purpose
Generate personalized candidate tracks from the user's session/listening history and their provided
CF embedding — three F2 `RetrievalChannel`s, each emitting canonical `track_id`s over `all_tracks`,
pulled to `topk_internal`: **(a) content-kNN** (recency-weighted pool of the user's history track
embeddings → nearest catalog tracks), **(b) CF** (`user_emb · track_emb` top-N from the cf-bpr user
vectors), **(c) same-artist** (catalog tracks by artists in the user's history). They feed R7 fusion
alongside R3/R4. The module owns one job — **warm personalization candidate generation that fails
cleanly to empty on cold** — and is gated on a warm-segment recall@K lift with no cold regression.

## 2. Interface / contract
Lives in `mcrs/retrieval/personalization.py` (one module, three classes). Each implements the F2
`RetrievalChannel` Protocol (`11_F2`): same `batch_text_to_item_retrieval` shape as every other
channel, returns **canonical** ids only (F1 `canonical_track_id`), registered by a unique `label`,
fuses nothing. All three read history via the **shared** `_history_tids(ctx)` helper (ported from
salvage `session_history.played_tids_from_context`) so train and serve recover identical history.

```python
class ContentKNNChannel:                 # implements F2 RetrievalChannel — plan §7.2.3
    label = "content_knn"
    def __init__(self, catalog: "Catalog", track_embs: "TrackEmbeddings", cfg: "ChannelConfig"): ...
    # query text is UNUSED for scoring (history-driven); kept for interface parity.
    def batch_text_to_item_retrieval(
        self, queries: list[str], topk: int,
        batch_context: list[dict] | None = None, user_ids: list[str] | None = None,
    ) -> list[list[str]]: ...

class CFChannel:                          # implements F2 RetrievalChannel — plan §7.2.4
    label = "cf"
    def __init__(self, catalog: "Catalog", track_embs: "TrackEmbeddings",
                 user_embs: "UserEmbeddings", cfg: "ChannelConfig"): ...
    # query text UNUSED; ranking is user_emb · track_emb. Cold/missing user → [].
    def batch_text_to_item_retrieval(self, queries, topk, batch_context=None, user_ids=None) -> list[list[str]]: ...

class SameArtistChannel:                  # implements F2 RetrievalChannel — plan §7.2.3
    label = "same_artist"
    def __init__(self, catalog: "Catalog", cfg: "ChannelConfig"): ...
    # candidates = catalog tracks by artists already in history, minus played, ranked by
    # (artist session-count desc, popularity desc). Empty history → [].
    def batch_text_to_item_retrieval(self, queries, topk, batch_context=None, user_ids=None) -> list[list[str]]: ...
```

**Inputs.** `batch_context[i]` is the per-turn context dict for query `i` — it MUST carry
`history_tids` (the causal listening history up to turn `t`, canonical ids; F1 `TurnContext.history_tids`
flows in here) and `segment` (`"cold"`/`"warm"`). `user_ids[i]` is the CF lookup key. The channels
read history from `batch_context` (matching the salvage sub-retriever contract) rather than from the
typed `TurnContext` so the shape is identical to R3/R4 and R7 can call all channels uniformly.

**Outputs.** `list[list[str]]`, one ranked canonical-id list per query, length ≤ `topk`
(= `topk_internal` from config), already de-duplicated and **excluding the user's own history tracks**
(the played set is removed — fusion/L1 decide replays per the L1 history A/B; R5 never re-injects a
played track as a "discovery"). A cold or history-empty query returns `[]`.

**Wiring (the spine).** R1/R2 build the `Query`; F1 builds `TurnContext` carrying causal
`history_tids` + `segment`; D1/R7 assemble `batch_context` and `user_ids` from those and call each
channel → ranked ids → **R7 fuses** (segment-aware weights, `46_R7`) → K1/K2. R5 produces no
`Candidate`s itself (R7 does that); it only ranks.

## 3. Dependencies
- **Modules:** F2 (`RetrievalChannel`, `Query`, `TurnContext`, `ChannelConfig`), F1
  (`Catalog` for `metadata`/`popularity`/`id_to_index`/`track_ids`; `TrackEmbeddings.matrix(modality)`
  for content-kNN + CF; `UserEmbeddings.vector(user_id)` for CF; `canonical_track_id`; `segment_for`),
  R1 (the `Query` — unused for scoring but consumed for parity), R7 (fuses the outputs; sets
  `topk_internal`/weights), F3 (`recall_at_k`, segmented `by_segment`, unique-recall diagnostic), P0
  (sets `cold_threshold`, `topk_internal`, recency knobs, channel keep/weights).
- **Embedding modalities (F1 `TrackEmbeddings`):** content-kNN uses a **content** modality
  (default `metadata-qwen3_embedding_0.6b`; CLAP/SigLIP are R6 territory and out of scope here — a
  P0 ablation may switch the content-kNN modality but the default stays a text/content one). CF uses
  the `cf-bpr` track matrix **and** `UserEmbeddings` (cf-bpr user vectors). Both are L2-normalized
  by F1 (or normalized here once; never per-call).
- **Libs:** `numpy` only. **No GPU, no model, no external API** — these are matmul/index-lookup
  channels over precomputed embeddings (matches salvage `cf_bpr` / `session_cf` "numpy-only").
- **Config:** `retrieval.channels[]` entries for `content_knn`/`cf`/`same_artist`
  (`label`, `type`, `weight`, `topk_internal`, `cold_weight`, `warm_weight`, `extra`), and `extra`
  knobs in §9. `segment.cold_threshold` (shared with F1/P0).

## 4. Design & logic

### 4.1 Shared history recovery (causal, single source)
All three channels recover the played track ids through **one** helper `_history_tids(ctx)` — ported
from salvage `played_tids_from_context` — which prefers `ctx["history_tids"]` (explicit canonical ids
populated by F1/D1) and intersects with `catalog.track_ids`. **No-leak invariant:** `history_tids` is
sliced ≤ turn `t` by F1 and the turn-`t` gold is held in `Conversations.gold(...)` (F1 §4.6), so R5
structurally cannot see a future turn or the gold. R5 additionally **asserts the gold is not in its
input history** is enforced upstream (F1), and R5's own no-leak test re-checks it (§7). The played set
is also **removed from the output** so the channel never "rediscovers" a track the user just played.

### 4.2 (a) Content-kNN from history (plan §7.2.3) — warm
1. Recover `played = _history_tids(ctx)`; if empty → `[]` (cold path).
2. Map each played id → its embedding row via `catalog.id_to_index` on
   `track_embs.matrix(content_modality)` (rows L2-normalized).
3. **Recency-weighted pool:** weight more-recent history tracks higher. Default weight for the
   `j`-th-from-most-recent track is `decay ** j` (`decay` in config, default `0.9`; `decay = 1.0` ⇒
   plain mean — the salvage `session_cf` centroid behavior). Pool vector `p = Σ_j w_j · e_j`,
   L2-normalize (skip query if `||p|| < 1e-9`).
4. **Score** all catalog rows by `M · p` (one matmul, plan "brute-force cosine = one matmul"),
   `argsort` desc, drop the played set, take `topk`. Deterministic tie-break by `id_to_index`.
5. **Edge cases:** all-played-missing-embedding (none map) → `[]`; single history track → kNN of
   that one track (still useful "more like the last play"). Optional `recency_window` caps the pool to
   the most-recent `W` tracks (long histories) — default `None` (use all).

### 4.3 (b) CF channel (plan §7.2.4) — warm, hard-cold gate
1. `u = user_embs.vector(user_id)`; if `user_id is None` **or** `u is None` (cold / not in the cf-bpr
   table) → `[]`. This is the **hard cold signal** — a user absent from the provided cf-bpr user
   embeddings has no CF vector and the channel contributes nothing (so fusion falls back to
   conversation channels). Never fabricate a vector.
2. Score `cf_track_mat · u` (cf-bpr 128-d, L2-normalized both sides), top-`topk` via
   `argpartition`+sort (salvage `cf_bpr` batched path: stack warm users, one matmul). Drop the
   played set, take `topk`. Query text unused.
3. **Batch:** collect warm rows, single `(W, T)` matmul, scatter back — cold rows stay `[]` (verbatim
   salvage batching, so cold users cost ~0).

### 4.4 (c) Same-artist channel (plan §7.2.3) — warm
1. Build once at init from `catalog.metadata`: `tid → artist` (lower/trimmed) and
   `artist → [tids sorted by popularity desc]` (salvage `_build_index`).
2. Per query: recover `played`; if empty → `[]`. Count artists in the played set
   (`Counter(artist for played tid)`); for each artist in `most_common()` order, emit its catalog
   tracks (popularity-desc) minus the played set, until `topk`. Deterministic.
3. Rationale (plan): a large fraction of gold next-tracks are by an artist already in the session —
   this is the cheapest, strongest *warm* recall channel and the one most orthogonal to embedding cosine.

### 4.5 Cold gating (plan §10 — the central design call)
- **Mechanism = empty list + segment-aware fusion weight, not a crash.** For cold users (empty
  `history_tids` and/or missing cf-bpr vector → `UserEmbeddings.vector()==None`), all three channels
  return `[]`. An empty list contributes **0** to weighted-RRF (`46_R7` math), so cold users fall back
  cleanly to R3/R4 + popularity priors — *zero regression by construction* (salvage `cf_bpr` docstring:
  "Zero regression for cold, net positive for warm").
- **`cold_weight` ≤ `warm_weight` in R7** (default `cold_weight = 0.0` for all three R5 channels): even
  if a cold user has a *short* degenerate history (1–2 tracks below the `cold_threshold`), the channel
  is down-weighted to avoid injecting low-confidence noise into the cold top-ranks (plan §10
  "down-weight/disable … avoid degenerate-embedding noise"). Segment routing keys off
  `ctx["segment"]` (= F1 `segment_for(history_tids, cold_threshold)`); enable per-segment weights only
  on a P0/ablation win, default both = `weight` for non-personalization channels.
- **`is_cold` feature feed-through:** R5 does not own the GBDT, but its per-channel rank/score is the
  signal K1 turns into the `is_cold` regime feature (plan §10 "GBDT learns the regime switch itself").
  R5's contract guarantees `[]` for cold so that feature is well-defined (a cold user has 0 R5 hits).
- **`cold_threshold` is set in P0** (history-length distribution, F1 §4.5), not here — R5 only *reads*
  `ctx["segment"]`.

### 4.6 Determinism & purity
Pure functions of (history, embeddings, catalog) + fixed `seed`; no RNG, no I/O at call time (embeddings/
indices built once at init). Argsort ties broken by `id_to_index` (stable). Two identical calls → byte-
identical output. No state mutated across calls.

## 5. Reuse
Port behind gate (plan §6.3: "Content-kNN / history §7.2.3" + "CF retriever §7.2.4" = *Port behind gate*):
- **`salvage/mcrs/retrieval_modules/session_history.py`** → `_history_tids` helper. **Keep verbatim**
  (the `history_tids`-first recovery is exactly the F1 contract). `session_match_features` is a *reranker*
  feature (K1), out of scope here.
- **`salvage/mcrs/retrieval_modules/session_cf.py`** (`SessionCFRetriever`, centroid recall@100 ~0.24)
  → **content-kNN** logic. **Port + extend:** add recency weighting (salvage is a plain mean) and read
  embeddings through F1 `TrackEmbeddings.matrix(content_modality)` instead of donating from `CF_BPR`
  (salvage coupled it to cf-bpr; R5 decouples so content-kNN uses a *content* modality, CF uses cf-bpr).
- **`salvage/mcrs/retrieval_modules/cf_bpr.py`** (`CF_BPR`, batched warm matmul, cold→`[]`) → **CFChannel**.
  **Port + adapt:** drop the embedded HF-dataset loaders/pickle caches (F1 `TrackEmbeddings`/`UserEmbeddings`
  own loading now); keep the batched `(W,T)` matmul + cold-skip + L2-norm exactly.
- **`salvage/mcrs/retrieval_modules/same_artist.py`** (`SameArtistRetriever`) → **SameArtistChannel**.
  **Port + adapt:** read catalog via F1 `Catalog.metadata` (was `MusicCatalogDB.metadata_dict`); keep
  the (artist-session-count, popularity) ranking verbatim.
- All four salvage channels already match the F2 `batch_text_to_item_retrieval` shape — **no reshaping**,
  only canonicalization at the boundary (F1) and F1-typed deps.

## 6. Eval & acceptance gate
**Gate (warm segment is the point):** measured in F3 with each channel run standalone over dev, then
fused (R7) with R3/R4:
1. **Warm recall@{50,100,200,500} lift:** on the **warm** segment (`by_segment["warm"]`), fused
   recall@K with R5 channels ON > fused recall@K with them OFF (R3/R4 only), at every K — strictly,
   and by a margin > F3 noise. This is the channel's reason to exist.
2. **Per-channel unique-recall (plan §7.3 req #2):** each of `content_knn`/`cf`/`same_artist` must add
   **unique** recall (golds it surfaces that R3/R4 + the other two miss) on warm, else it is dropped or
   zero-weighted in R7. Report the unique-recall table per channel.
3. **Cold non-regression (report, don't hide):** on the **cold** segment, fused recall@K with R5 ON
   must be **≥** R5 OFF at every K. Expectation is **~0 lift on cold** (channels return `[]`) — that is
   the *correct, honest* result and is reported as such, not concealed. Any cold *regression* fails the
   gate (it would mean a degenerate-history user got noise) → fix via `cold_weight`/threshold.
4. **Overall recall@K** reported for completeness (warm-weighted blend of the two segments).
Numbers + the weight/decay sweep that justified the chosen config are logged to `reports/experiments.md`
and memory. Final fusion gate (recall@20 ≥ 0.75, recall@200 ≥ 0.90) lives in R7, not here.

## 7. Tests
- **Unit — content-kNN:** known history of 2 tracks with crafted embeddings → the nearest catalog row
  ranks first; recency `decay < 1` shifts ranking toward the most-recent track; `decay = 1.0` reproduces
  the salvage plain-centroid output (byte-identical).
- **Unit — CF:** crafted user vector + track matrix → expected top-K; `user_id is None` → `[]`;
  `user_id` absent from `UserEmbeddings` (`vector()==None`) → `[]`; batched call matches per-query calls.
- **Unit — same-artist:** history by artists {A,B}, A played twice → A's tracks rank before B's; played
  tracks excluded; topk respected; deterministic tie-break.
- **No-leak / causal:** turn-`t` gold (from `Conversations.gold`) is **never** in any channel output;
  channel input history contains no turn-`>t` id and not the gold; played tracks excluded from output.
- **Id-space (plan §7.3 req #1):** every output id `∈ Catalog.track_ids` (canonical); a non-catalog
  history id is dropped, not emitted.
- **Cold gating:** empty `history_tids` → all three `[]`; missing cf-bpr vector → CF `[]`; a `[]` channel
  contributes 0 to R7 fusion (integration assert); cold-segment fused recall ≥ R5-OFF (non-regression).
- **Wiring/integration:** R7 fuses R3+R4+R5 with segment-aware weights; warm fused recall@K rises and
  cold is flat (the §6 gate, on a fixture + dev).
- **Determinism:** fixed seed + same inputs → byte-identical lists across two runs; embedding row order
  matches `Catalog.id_to_index` (spot-check `matrix[id_to_index[tid]] == vector(tid)`).

## 8. Failure modes & guards
- **Cold user → degenerate/missing CF vector** treated as a real signal, not an error → `vector()==None`
  ⇒ `[]`; `cold_weight=0`; never crash, never fabricate (F1 §8 + plan §10).
- **Short degenerate history injects noise** (1–2 tracks below `cold_threshold`) → segment-aware
  down-weight + cold non-regression gate catches it.
- **Embedding/metadata row misalignment** → scores point at the wrong track → always index via
  `Catalog.id_to_index`; never trust embedding-file row order (F1 §8 guard); determinism test asserts.
- **Non-canonical id from history/catalog** → fusion double-counts/misses → `canonical_track_id` at the
  boundary + `⊆ catalog` assert (plan §7.3 req #1).
- **Replay re-injection** (channel re-emits a played track as a "discovery") → played-set removal +
  test; replay handling is the L1 history A/B's job, not R5's.
- **Wrong modality for content-kNN** (accidentally cf-bpr or an R6 audio modality) → modality is an
  explicit config knob, default `metadata-qwen3`; init asserts the modality exists in `TrackEmbeddings`.
- **Noise channel with no unique recall** (plan §7.3 req #2) → unique-recall gate (§6.2) drops/zero-weights
  it; never ship a channel that only displaces golds.
- **Pull depth too shallow** → `topk_internal ≥ fusion_K` enforced by R7; R5 simply honors the requested
  `topk`.

## 9. Config knobs (defaults; types validated by the F2 loader)
Per-channel under `retrieval.channels[]` (label = `content_knn` / `cf` / `same_artist`):
- `type` — `"content_knn" | "cf" | "same_artist"` (factory dispatch).
- `weight` (float) — base RRF weight (from P0 sweep).
- `topk_internal` (int) — pull depth (≥ fusion_K; from P0).
- `cold_weight` / `warm_weight` (float, optional) — segment-aware RRF weights; default `cold_weight=0.0`,
  `warm_weight=weight` for all three R5 channels (plan §10). Used by R7 only if `fusion_strategy=segmented`.
- `extra.content_modality` (str, `content_knn` only) — `TrackEmbeddings` modality; default
  `"metadata-qwen3_embedding_0.6b"`.
- `extra.recency_decay` (float, `content_knn`) — per-step recency decay; default `0.9` (`1.0` ⇒ plain mean).
- `extra.recency_window` (int|null, `content_knn`) — cap pool to most-recent W tracks; default `null`.
- `extra.normalize` (bool) — L2-normalize at init if F1 didn't; default `true`.
Shared: `segment.cold_threshold` (read for routing; **value set in P0**), `seed`.

## 10. Definition of Done & review checklist
- [ ] `ContentKNNChannel`, `CFChannel`, `SameArtistChannel` implement F2 `RetrievalChannel`; one shared
      `_history_tids` helper; return **canonical** ids only; emit no `Candidate`s (R7's job).
- [ ] Cold path proven: empty history / `UserEmbeddings.vector()==None` → `[]`; contributes 0 to R7;
      `cold_weight` wired; `is_cold` well-defined (cold ⇒ 0 R5 hits).
- [ ] Warm recall@{50,100,200,500} lift demonstrated on `by_segment["warm"]` vs R3/R4-only; **cold
      reported as ~0 lift and non-regressing** (not hidden); per-channel unique-recall table logged.
- [ ] No-leak (gold/future-turn never in output, played excluded), id-space ⊆ catalog, and determinism
      tests green; embeddings indexed via `id_to_index`.
- [ ] Salvage ported (session_history/session_cf/cf_bpr/same_artist) with F1-typed deps; content-kNN
      decoupled from cf-bpr (content modality); CF keeps batched warm matmul + cold-skip.
- [ ] Chosen weights/decay locked in `config/<exp>.yaml` (train==serve); sweep logged to
      `reports/experiments.md`; code review approved; no `Any` in public signatures.

## 11. Build order & dependencies
**Built after F1/F2/F3 + P0 (which sets `cold_threshold`, `topk_internal`, content modality, weights)
and after R1 (the `Query`).** Runs **in parallel with R3/R4/R6** — all are independent
`RetrievalChannel`s. **Depends on:** F1 (`Catalog`, `TrackEmbeddings`, `UserEmbeddings`,
`canonical_track_id`, `segment_for`), F2 (`RetrievalChannel`, `TurnContext`, config), F3 (segmented
recall + unique-recall), P0 (sizing/threshold), R1 (`Query`). **Blocks:** R7 (fuses these channel
outputs with segment-aware weights) and therefore K1/K2 and the first end-to-end submission
(`… → R5 → R7 → L1 → responder → D1`). Not on the *minimum* critical path (R3+R7 suffice for a first
valid submission) but is the primary **warm-segment** recall lever layered on next.
