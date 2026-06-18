# F1 — Data Access & Catalog/User DB

> Foundation module. The **single source of truth for loading and exposing** every on-disk dataset
> (catalog metadata, track embeddings, user CF embeddings, user profiles, conversations) and the
> **canonical `track_id` id-space** that every retrieval channel fuses over. Builds the causal
> `TurnContext` / `UserProfile` objects defined in `11_F2_interfaces_contracts_config.md` — it does
> **not** redefine those types, it constructs them. See `000_INDEX.md` for the catalogue.

## 1. Purpose
Load the five TalkPlayData-Challenge datasets, expose catalog metadata + the canonical `track_id`
universe (`all_tracks`), the pre-computed track & user embeddings, and the conversation/turn data;
provide the **one** id-normalization function used everywhere (plan §7.3 req #1); construct causal
`TurnContext`/`UserProfile` objects per F2; and apply the cold/warm `segment` label. Its gate is
**integrity**, not a model metric.

## 2. Interface / contract
Lives in `mcrs/data/` (new package): `mcrs/data/catalog.py`, `mcrs/data/users.py`,
`mcrs/data/embeddings.py`, `mcrs/data/conversations.py`, `mcrs/data/ids.py`, `mcrs/data/segment.py`.
Returns the F2 record types; never invents new schemas. Public surface (signatures, not full bodies):

```python
# mcrs/data/ids.py — the ONE id normalizer; imported by every channel (plan §7.3 req #1)
def canonical_track_id(raw: str) -> str: ...        # strip prefixes/whitespace, NFC, exact-match to catalog
def canonical_track_ids(raw: list[str]) -> list[str]: ...

# mcrs/data/catalog.py
class Catalog:
    """all_tracks metadata + the frozen canonical id set."""
    def __init__(self, dataset_name: str, split_types: list[str], corpus_types: list[str]): ...
    track_ids: frozenset[str]                        # the canonical universe (~47,071 — verify in P0)
    id_to_index: dict[str, int]                      # stable row order for the embedding matrix
    index_to_id: list[str]
    def metadata(self, track_id: str) -> dict: ...   # raw row (track_name, artist_name, ... lists)
    def id_to_metadata(self, track_id: str, enriched: bool = False) -> str: ...
        # corpus-typed doc text (BM25/dense doc). enriched=True layers A1's enriched corpus
        # (graceful fallback to the raw doc for tracks A1 didn't cover); F1 loads/merges A1's parquet.
    def __contains__(self, track_id: str) -> bool: ...
    def __len__(self) -> int: ...

# mcrs/data/embeddings.py
class TrackEmbeddings:
    """One float32 matrix per modality, row-aligned to Catalog.id_to_index."""
    def matrix(self, modality: str) -> "np.ndarray": ...   # (n_tracks, dim), L2-norm optional
    def vector(self, track_id: str, modality: str) -> "np.ndarray": ...
    modalities: list[str]   # audio-laion_clap, image-siglip2, cf-bpr,
                            # attributes/lyrics/metadata-qwen3_embedding_0.6b

class UserEmbeddings:
    def vector(self, user_id: str) -> "np.ndarray | None": ...  # cf-bpr; None ⇒ cold/missing
    def __contains__(self, user_id: str) -> bool: ...

# mcrs/data/users.py
class Users:
    def profile(self, user_id: str) -> "UserProfile": ...       # F2 UserProfile (frozen)
    def __contains__(self, user_id: str) -> bool: ...

# mcrs/data/conversations.py
class Conversations:
    """Iterates sessions and emits causal TurnContext objects (F2)."""
    def turns(self, split: str) -> "Iterator[TurnContext]": ...      # one per session×turn, causal
    def gold(self, session_id: str, turn_number: int) -> str | None: ...  # held SEPARATE from ctx

# mcrs/data/segment.py
def segment_for(history_tids: list[str], cold_threshold: int) -> str: ...  # "cold" | "warm"
```

`Conversations.turns(...)` is the constructor of `TurnContext`: it fills
`utterances` with turns `1..t` only, attaches the F2 `UserProfile`, the `history_tids` available
**up to** turn `t`, the `goal` (`conversation_goal.listener_goal`), and the `segment` label. The
gold track for turn `t` is returned **only** by `Conversations.gold(...)` — never placed on the
`TurnContext` (causal-by-construction, per F2 §4).

## 3. Dependencies
- **Data (on disk under `data/`, gitignored; re-fetch via `download_data.py`):**
  - `data/TalkPlayData-Challenge-Track-Metadata/{all_tracks,test_tracks}` — catalog rows (verified `num_examples`: all_tracks **47,071**, test_tracks 7,405).
  - `data/TalkPlayData-Challenge-Track-Embeddings/{all_tracks,test_tracks}` — 6 modalities per track (verified field names below).
  - `data/TalkPlayData-Challenge-User-Embeddings/{train,test_warm,test_cold}` — `cf-bpr` user vectors (train 8,591 / warm 371 / cold 129 rows).
  - `data/TalkPlayData-Challenge-User-Metadata/all_users` — 8,772 user rows.
  - `data/TalkPlayData-Challenge-Dataset/{train,test}` — conversations (train 15,199 / dev-test 1,000 sessions); `data/TalkPlayData-Challenge-Blind-A/test` — 80 sessions.
- **Modules:** F2 (`mcrs/contracts.py` types: `TurnContext`, `UserProfile`; `mcrs/config.py` for `paths.*`, `segment.cold_threshold`).
- **Libs:** `datasets` (arrow loader), `numpy`. No GPU, no models, no external APIs.
- **Config:** `paths.*`, `data.dataset_names{}`, `data.corpus_types[]`, `segment.cold_threshold`, `seed`.

## 4. Design & logic

### 4.1 The id space (plan §7.3 req #1 — the load-bearing decision)
- `Catalog.track_ids` (the `all_tracks` split) is **the** universe. Every retrieval channel emits ids
  passed through `canonical_track_id` before fusion; F3 asserts each channel output `⊆ track_ids`.
- `canonical_track_id` is the **single** normalizer (matches F2 §4 "one id space"). It must NOT be
  confused with the prior `strip_track_id_prefix` doc-**text** helper,
  not an id normalizer (plan §7.3 req #1 explicitly warns against leaning on it). Initial body:
  strip surrounding whitespace + any `track_id: ` prefix, NFC-normalize, then require exact membership
  in `track_ids` (raise/log on miss). The exact set of transforms is fixed in **P0 EDA** by inspecting
  raw ids in conversations vs. catalog (do they already match byte-for-byte? — likely yes; verify).
- `id_to_index` gives a **stable** row order so `TrackEmbeddings.matrix()` rows align to ids; this is
  what makes "brute-force cosine over the full catalog = one matmul" (plan §insight 2) correct.

### 4.2 Catalog metadata
- Follow the prior `MusicCatalogDB` pattern: build `{track_id: row}` once, share across call sites
  (a process-wide `_SHARED_METADATA` cache served exactly this — see Reuse).
- `metadata()` returns the raw row. Fields (verified from `dataset_info.json`): `track_id` (str),
  and **List[str]** fields `ISRC`, `track_name`, `artist_name`, `album_name`, `tag_list`, `artist_id`,
  `album_id`; plus `popularity` (float64), `release_date` (str), `duration` (int64).
  **Note:** name/artist/album/tags are *lists*, not scalars — `id_to_metadata` must `", ".join(...)`
  (matching the prior `format_catalog_track_text`). A1 (enrichment) and R3/R4 (BM25/dense docs) consume
  `id_to_metadata`. **Enriched layering (A1 wiring):** `id_to_metadata(tid, enriched=True)` returns
  A1's enriched `doc_text` for covered tracks and falls back to the raw doc otherwise; F1 loads/merges
  A1's enriched-corpus parquet (keyed by canonical `track_id`, path/hash from config) — A1 produces it,
  F1 owns the merge + accessor, R3/R4 read text only through this surface.
  `corpus_types` is config-driven (baseline default: `track_name, artist_name,
  album_name, release_date`).
- **`all_tracks` vs `test_tracks`:** F1 loads `all_tracks` as the retrieval universe. The
  size-discrepancy question (plan §2: site says ">1M", evaluator/baseline says ~50k; on-disk
  `all_tracks` = **47,071**) is recorded as a **P0 input, not an F1 decision** — F1 simply exposes
  whatever the loaded split contains and asserts its count. If P0 confirms a true ~1M catalog, F1's
  brute-force matrix path is swapped for an ANN index downstream (R4); F1's contract is unchanged.

### 4.3 Track & user embeddings
- Track-embedding modalities (verified field names): `audio-laion_clap`, `image-siglip2`, `cf-bpr`,
  `attributes-qwen3_embedding_0.6b`, `lyrics-qwen3_embedding_0.6b`, `metadata-qwen3_embedding_0.6b`.
  Each is `List[float64]`; F1 stacks each modality into one **float32** `(n_tracks, dim)` matrix,
  row-ordered by `index_to_id`. **Per-modality dims are not in `dataset_info.json` (variable-length
  List type) — read them once and record (verify in P0 EDA).**
- User embeddings: only `cf-bpr` per user. `UserEmbeddings.vector` returns `None` for users absent
  from the embedding table → a hard cold signal feeding `segment` and the `is_cold` GBDT feature
  (plan §10). The split layout (`train` / `test_warm` / `test_cold`) is itself a warm/cold signal —
  load all and key by `user_id`; record which split each user came from (verify in P0).
- Embeddings are loaded **lazily per modality** (the track-embeddings arrow is ~1.9 GB on disk); a
  channel that only needs `metadata-qwen3` never materializes CLAP/SigLIP.

### 4.4 Conversations → causal `TurnContext`
- A session row carries: `session_id` (`{user_id}__{date}`), `user_id`, `session_date`,
  embedded `user_profile` (age, age_group, country_code, country_name, gender, preferred_language,
  **preferred_musical_culture**, user_id, user_split), `conversation_goal`
  (`category`, `listener_goal`, `specificity`), `conversations` (List of {`content`, `role`,
  `thought`, `turn_number`}), and `goal_progress_assessments`.
- For each turn `t` (1..N, avg ~8), emit a `TurnContext` with `utterances` = the conversation
  contents for turns `1..t` **only** (assert `len(utterances) == turn_number`, per F2), `goal` =
  `conversation_goal.listener_goal` (if present), `history_tids` = the user's listening history
  available up to `t`. **How turn-`t` gold and the available history are derived from the row is a
  P0-confirmed rule** (inspect `music-crs-evaluator/make_ground_truth.py` — plan §5.2): gold = the
  track played/selected at turn `t`; history-up-to-`t` excludes the turn-`t` gold and any future
  turns (verify exact field & whether gold may also appear in history — drives the L1 history A/B).
- **`thought` is a leak surface:** the conversation includes a model `thought` field; F1 must **never**
  put `thought` (or any future-turn content) into `TurnContext.utterances`. P0 leakage audit confirms
  what `thought` contains; default = exclude it from causal inputs.

### 4.5 `UserProfile` (F2) + segment
- `Users.profile` builds the F2 `UserProfile` (`user_id`, `age`, `gender`, `country`, `history_tids`)
  from User-Metadata. F2's `UserProfile` is intentionally lean; the richer session-embedded
  `user_profile` (preferred_musical_culture, preferred_language, user_split) is exposed on the raw
  row for channels/feature-builders that want it, without bloating the F2 contract.
- `segment_for(history_tids, cold_threshold)` → `"cold"`/`"warm"` by `len(history_tids) <= threshold`
  (and/or missing CF vector). **F1 only *applies* the threshold; the value is decided in P0** from the
  history-length distribution (plan §5.4, §10) and lives in `config.segment.cold_threshold`.

### 4.6 No-leak / causal constraints
- Gold is reachable only via `Conversations.gold(...)`, structurally separate from `TurnContext`.
- `history_tids` and `utterances` are sliced to `≤ t`; constructors assert no future turn leaks.
- All loaders are read-only and deterministic given `seed` (stable row order, sorted id sets).

## 5. Reuse
- **Catalog:** the prior `MusicCatalogDB` pattern (recoverable from the old git branches —
  recall-union-lgbm, stage-b-cross-encoder, fresh-model, exp/*) had a process-wide `_SHARED_METADATA`
  cache and a `format_catalog_track_text` single-source doc rendering; **rebuild the surrounding
  `Catalog`** to add `track_ids`/`id_to_index`/`__contains__` (the prior DB exposed only
  `id_to_metadata`). **Rebuild, reapplying the cache + single-source-text patterns.**
- **User profile:** the prior `UserProfileDB` pattern (recoverable from the old git branches) →
  adapt into F2 `UserProfile`. Pristine equivalents (`music-crs-baselines/mcrs/db_item/music_catalog.py`,
  `.../db_user/user_profile.py`) are the simpler reference; the prior DB had already solved caching +
  canonical text. **Rebuild + adapt.**
- **Embeddings / ids / conversation→TurnContext loaders:** **new (rewrite)** — there was no central
  embedding loader, no canonical id normalizer, and no causal `TurnContext` builder in the prior tree.
- **Config shape:** clone `music-crs-baselines/config/llama1b_bm25_devset.yaml` (verified keys:
  `item_db_name`, `user_db_name`, `track_split_types`, `user_split_types`, `corpus_types`,
  `cache_dir`) for the `data.*`/`paths.*` block; the F2 schema owns the typed config object.

## 6. Eval & acceptance gate
**Integrity asserts (join-coverage + dim-consistency) — all must pass; this is the module's gate:**
1. **No dup track_ids:** `len(track_ids) == len(raw all_tracks rows)`; `id_to_index` is a bijection.
2. **Conversation ids resolve:** every `track_id` referenced in any conversation (gold + history)
   `∈ catalog.track_ids` after `canonical_track_id` (report the miss-rate; **0% is the gate**, any
   miss is a P0 finding to explain, not silently drop).
3. **User history resolves:** every `history_tids` entry resolves to a catalog id; every
   conversation `user_id` ∈ Users (report cold/missing-profile rate).
4. **Embedding row counts == metadata count** per modality; matrix rows align to `id_to_index`
   (spot-check: `matrix(m)[id_to_index[tid]]` == `vector(tid, m)`).
5. **Dim consistency:** every track in a modality has the same vector length; record each dim.
6. **Determinism:** two loads → identical `index_to_id` order and identical asserts.

Measured in F3's harness / the P0 EDA notebook; numbers logged to `reports/eda.md` + memory.

## 7. Tests
- **Unit:** `canonical_track_id` idempotent (`f(f(x))==f(x)`), strips a `track_id: ` prefix,
  maps a known raw id to its catalog id, raises/flags on a non-catalog id.
- **Integrity (the gate, on a small fixture + full data):** the six asserts in §6.
- **Causal / no-leak:** `Conversations.turns` emits `len(utterances)==turn_number`; turn-`t` context
  contains **no** turn-`>t` content and **not** the turn-`t` gold; `thought` excluded from utterances.
- **Wiring:** a `TurnContext`/`UserProfile` built here satisfies F2's constructor asserts unchanged;
  `TrackEmbeddings.matrix` rows index by the same `track_id`s a channel would canonicalize to.
- **Determinism:** fixed `seed` → stable `index_to_id`; repeated loads byte-stable.

## 8. Failure modes & guards
- **Silent id mismatch** (channel emits non-canonical id) → fusion double-counts/misses. Guard:
  single `canonical_track_id` at every channel boundary + F3 `⊆ catalog` assert (plan §7.3 req #1).
- **Embedding/metadata row misalignment** → personalization scores point at the wrong track. Guard:
  build the matrix *from* `id_to_index`, never trust file row order; assert §6.4.
- **Future-turn / gold / `thought` leak into `TurnContext`** → inflated offline recall. Guard: causal
  slicing + constructor asserts + gold held in a separate accessor; P0 leakage audit.
- **Cold user → degenerate/missing CF vector** treated as a real signal. Guard: `vector()→None`,
  `segment="cold"`, `is_cold` feature; never crash on missing.
- **Catalog-size surprise (47k vs 1M)** → wrong retrieval backend. Guard: assert + log the loaded
  `all_tracks` count; flag if it diverges from the P0-recorded expectation (don't decide here).
- **List-typed metadata fields** (`track_name` etc. are List[str]) → `KeyError`/type bugs if treated
  as scalars. Guard: `id_to_metadata` joins lists; tests cover multi-value rows.
- **Test/leak split contamination** → use only `all_tracks` for the retrieval universe; never expose
  `test_tracks` as the candidate pool (P0 confirms what `test_tracks` is for).

## 9. Config knobs (values default; types validated by the F2 loader)
- `paths.data_root` — root for the `TalkPlayData-Challenge-*` dirs (default `./data`).
- `data.dataset_names.{tracks_metadata, track_embeddings, user_embeddings, user_metadata, conversations, blind_a}` — HF repo / local dir names.
- `data.track_split_types` (default `["all_tracks"]`), `data.user_split_types` (default `["all_users"]`).
- `data.corpus_types` (default `["track_name","artist_name","album_name","release_date"]`).
- `data.embedding_modalities` — which track-embedding columns to load (lazy; default all six).
- `data.normalize_embeddings` (bool) — L2-normalize matrices at load (default per P0).
- `segment.cold_threshold` (int) — applied here; **value set in P0** (history-length distribution).
- `seed` (int) — deterministic row ordering.

## 10. Definition of Done & review checklist
- [ ] `Catalog`, `TrackEmbeddings`, `UserEmbeddings`, `Users`, `Conversations`, `canonical_track_id`,
      `segment_for` implemented; return **F2** types (no schema re-invention).
- [ ] All six integrity asserts (§6) pass on full data; miss-rates logged to `reports/eda.md`.
- [ ] Unit + causal + wiring + determinism tests green; gold never reachable from `TurnContext`.
- [ ] `canonical_track_id` is the **only** id normalizer; at least one channel (R3/R7) imports it.
- [ ] Embedding matrices verified row-aligned to `id_to_index`; per-modality dims recorded.
- [ ] Catalog-size + cold/warm-threshold + dims logged as **P0 inputs** (F1 didn't decide them).
- [ ] Code review approved; no `Any` in public signatures; loaders read-only & deterministic.

## 11. Build order & dependencies
**Built first, in parallel with F2** (F2 §11: "built second… F1 supplies the canonical id set/profile
loader"). F1 depends only on F2's *type definitions* (`TurnContext`, `UserProfile`) and the config
schema — a soft, no-cycle dependency (F2 is pure types/Protocols). **Blocks:** F3 (needs the id set +
loaders to assert parity), P0 (all EDA reads through F1), and every R/K/L/S/D module (the id space,
catalog docs, embeddings, and `TurnContext` stream). On the critical path
`F1+F2+F3 → P0 → R1+R3 → R7 → L1 → responder → D1` (plan §17, `000_INDEX.md`).
