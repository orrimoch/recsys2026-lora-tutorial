# F2 — Interfaces, Data Contracts & Config

> Foundation module. The **single source of truth** for every data object and module interface that
> flows through the pipeline, plus the config schema. All other modules import these types and wire
> against them — none re-invents a schema. See `000_INDEX.md` for the catalogue.

## 1. Purpose
Define the canonical, typed data contracts (`TurnContext`, `Query`, `Candidate`, `RankedList`, `SubmissionRow`), the four module interfaces (`RetrievalChannel`, `Reranker`, `Filter`, `Responder`), and the experiment config schema — so modules are wired correctly by construction and can be built/tested in isolation.

## 2. Interface / contract
Pure data + Protocol definitions; no heavy deps. Lives in `mcrs/contracts.py` (+ `mcrs/config.py`). Every type is causal-by-construction (carries only data available at turn `t`).

```python
# ----- record types (frozen dataclasses; serializable) -----
@dataclass(frozen=True)
class TurnContext:
    session_id: str            # "{user_id}__{date}"
    user_id: str
    turn_number: int           # 1..8, causal anchor
    utterances: list[str]      # utterances[0..t-1], i.e. turns 1..t ONLY (no future)
    goal: str | None           # conversation goal field (if present at t)
    user_profile: "UserProfile"
    history_tids: list[str]    # listening history available UP TO t (canonical track_ids)
    segment: str               # "cold" | "warm" (from F1/P0 threshold)
    # derived, optional (filled by channels/feature builder, never the gold):
    history_embedding: "np.ndarray | None" = None

@dataclass(frozen=True)
class UserProfile:
    user_id: str
    age: int | None
    gender: str | None
    country: str | None
    history_tids: list[str]

@dataclass(frozen=True)
class Query:
    text: str                              # constructed causal query (R1)
    structured: dict | None = None         # R2: {positive_attrs, negative_attrs, seed_artists, mood, genre, era}
    per_channel: dict[str, str] = field(default_factory=dict)  # channel_label -> override query (e.g. "colbert_query")

@dataclass
class Candidate:
    track_id: str                          # CANONICAL catalog id (see F1)
    channel_scores: dict[str, float] = field(default_factory=dict)  # label -> raw score
    channel_ranks:  dict[str, int]   = field(default_factory=dict)  # label -> 1-indexed rank
    rrf_score: float = 0.0
    features: dict[str, float] = field(default_factory=dict)        # K1 fills this

@dataclass
class RankedList:
    turn: TurnContext
    items: list[Candidate]                 # ordered best→worst, provenance preserved

@dataclass
class SubmissionRow:
    session_id: str
    user_id: str
    turn_number: int
    predicted_track_ids: list[str]         # ordered, UNIQUE, ≤20, valid catalog ids
    predicted_response: str                # may be ""
```

```python
# ----- module interfaces (Protocols) -----
class RetrievalChannel(Protocol):
    label: str
    # MUST return canonical track_ids over all_tracks; absent = not retrieved.
    # Signature matches the RRF sub-retriever contract (do not change shape).
    def batch_text_to_item_retrieval(
        self, queries: list[str], topk: int,
        batch_context: list[dict] | None = None, user_ids: list[str] | None = None,
    ) -> list[list[str]]: ...

class Reranker(Protocol):
    def rerank(self, ctx: TurnContext, candidates: list[Candidate]) -> RankedList: ...

class Filter(Protocol):
    def apply(self, ranked: RankedList) -> list[str]: ...   # → ≤20 unique valid track_ids

class Responder(Protocol):
    def respond(self, ctx: TurnContext, top_tracks: list[dict]) -> str: ...
```

**Wiring contract (the spine):** `R1/R2 → Query`; each `RetrievalChannel` consumes `Query` (shared text or `per_channel` override) → ranked `track_id`s; `R7` fuses channel outputs → `list[Candidate]` with `rrf_score`+ranks; `K1` fills `Candidate.features`; `K2/K3` → `RankedList`; `L1` → ≤20 ids; `S1` → response; `D1` assembles `SubmissionRow`.

## 3. Dependencies
`numpy`, stdlib `dataclasses`/`typing`. F1 supplies the canonical catalog id set + `UserProfile` loader. No model/API deps (kept light so every other module can import it cheaply).

## 4. Design & logic
- **Causal by construction:** `TurnContext` exposes only ≤t data; there is no field for the gold track or future turns. Constructors assert `len(utterances) == turn_number` and `gold ∉ inputs`.
- **One id space:** `track_id` everywhere is the canonical catalog key (F1). Any channel deriving ids canonicalizes before returning (plan §7.3 req #1).
- **`per_channel` queries** mirror the prior `rrf.resolve_sub_queries` (`query_key`) so a channel (e.g. ColBERT) can use a compact query while others use the shared one — same object in train and serve.
- **Serialization:** every record has `to_dict`/`from_dict`; `SubmissionRow` dumps with `ensure_ascii=False`.

## 5. Reuse
Channel signature is lifted verbatim from the `batch_text_to_item_retrieval` RRF sub-retriever contract (recoverable from the old git branches — recall-union-lgbm, stage-b-cross-encoder, fresh-model, exp/*) so prior channels port without reshaping. Config shape clones `music-crs-baselines/config/llama1b_*.yaml`. The contracts file itself is **new** (rewrite — there was no central contract before).

## 6. Eval & acceptance gate
Not a metric module, so its gate is correctness: **schema round-trip tests pass** (every record `from_dict(to_dict(x)) == x`), causal-assertion tests fire on violations, and a `SubmissionRow` list validates against the strict schema (≤20, unique, valid ids, all session×turns present, `ensure_ascii=False`).

## 7. Tests
- Round-trip serialization for all record types.
- Causal guards: constructing `TurnContext` with `len(utterances) != turn_number` or a gold-containing field raises.
- `RetrievalChannel` Protocol conformance: a stub channel satisfies `isinstance`-style structural check.
- Config: load `config/<exp>.yaml` → typed object → re-dump is stable; unknown keys rejected.

## 8. Failure modes & guards
- Mixed id forms across channels → enforce canonicalization at the channel boundary + F3 asserts `⊆ catalog`.
- Silent future-turn leak → causal asserts in constructors; F3 no-leak test at the harness level.
- Config drift train↔serve → one config object loaded by both; D1 records the config hash in the submission run.

## 9. Config knobs
Defines the **schema**, not values: `retrieval.channels[]` (label, type, weight, topk_internal, query_key, extra), `retrieval.fusion` (k, weights, fusion_strategy), `retrieval.topk`, `rerank.*`, `filter.*` (history_rule A/B, tail_mmr), `responder.*`, `segment.cold_threshold`, `paths.*`, `seed`, `model_revisions{}`. Every knob has a default + type; loader validates.

## 10. Definition of Done & review checklist
- [ ] All record + Protocol types defined, typed, frozen where stated.
- [ ] Round-trip + causal-assertion + config tests green.
- [ ] At least one downstream module (R3 or R7) imports and wires against these with no shape change.
- [ ] Code review approved; no `Any` leaks in public signatures.

## 11. Build order & dependencies
**Built second, in parallel with F1** (F1 supplies the canonical id set / profile loader this references). Blocks everything in Phase 1+. Depends on: F1 (id space). Blocks: all R/K/L/S/D modules.
