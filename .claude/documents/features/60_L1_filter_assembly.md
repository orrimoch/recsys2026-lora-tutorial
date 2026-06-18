# L1 — Filtering & Top-20 Assembly

> Phase-3. Turns the reranked `RankedList` (K2/K3) into the final ≤20 unique, valid `track_id`s —
> **without ever touching the catalog universe** (plan non-negotiable rule #1; this is post-retrieval
> pruning, not catalog subsetting). See `000_INDEX.md`.

## 1. Purpose
Produce a clean, schema-valid, rule-correct top-20 from the ranked candidates: dedup/uniqueness, fill exactly 20, the history-rule A/B, light tail-diversification, and a validity guard — improving (or holding) diversity with zero nDCG@20 loss.

## 2. Interface / contract
Lives in `mcrs/filter/assembly.py`. Implements F2 `Filter`.

```python
class TopKAssembler:                        # implements F2 Filter
    def __init__(self, catalog: "Catalog", cfg: "FilterConfig"): ...
    def apply(self, ranked: RankedList) -> list[str]:  ...   # → exactly ≤20 unique valid canonical track_ids
```

**Wiring:** input = K2/K3 `RankedList`; output = the ordered ≤20 ids placed on `SubmissionRow.predicted_track_ids` by D1. Pure, deterministic, CPU.

## 3. Dependencies
F2 (`Filter`, `RankedList`, `Candidate`), F1 (`Catalog` for the validity guard + `canonical_track_id`; user `history_tids` from `TurnContext`), F3 (nDCG non-regression + diversity), P0 (`gold_in_history_rate` → history-rule default). Validation: `mcrs/run/harness.py` (`validate_submission`) + the notebook's blind guards. No model/GPU.

## 4. Design & logic
- **Dedup & uniqueness (hard):** drop duplicate canonical ids, preserve best rank. Output ids must be unique (official scorer raises on dups).
- **History rule (A/B):** optionally remove tracks already in the user's `history_tids`. In music, replays are real, so this can HURT — it's an A/B knob whose **default comes from P0's `gold_in_history_rate`**: if golds are frequently re-listens (high rate), default OFF; if golds are rarely in history, default ON. Decided on data, not assumed.
- **Backfill to 20:** if dedup/history-removal leaves < 20, backfill from the next-best ranked candidates (still in-pool) so exactly 20 (or all available if the pool is smaller) are filled.
- **Tail diversification (free diversity):** light MMR applied **only to positions ~11–20** (where the gold is unlikely), to lift catalog-diversity. The **top 1–10 stay purely relevance-ranked** — tail-MMR never reorders them. Gate behind "nDCG@20 not harmed" (Carbonell & Goldstein 1998).
- **Validity guard:** every output id ∈ `Catalog.track_ids` (canonical). No catalog subsetting — this only prunes/reorders within the retrieved+reranked set.

## 5. Reuse
Dedup/validity/backfill: small new code + the checks in `mcrs/run/harness.py` (`validate_submission`) + the notebook's blind guards (reuse as the validator). Tail-MMR: **new** (gated). History-rule: new toggle. **Mostly new (rewrite), reuse the harness validator.**

## 6. Eval & acceptance gate
Via F3: output is **always ≤20 unique valid ids** (schema-valid), **nDCG@20 non-regression** vs the raw reranked top-20 (tail-MMR + history-rule must not drop nDCG), and **catalog diversity improves or holds**. The history-rule ships in whichever A/B direction wins on dev nDCG@20.

## 7. Tests
- Output always ≤20, unique, valid catalog ids; backfill reaches 20 when the pool allows.
- **Tail-MMR never reorders the top-10** (positions 1–10 identical to input order); only 11–20 may change.
- History-rule toggle covered both ways; default wired from P0's `gold_in_history_rate`.
- **nDCG@20 non-regression assertion** on a fixture; diversity computed via F3 (official function).
- Determinism: same `RankedList` + seed → same output.

## 8. Failure modes & guards
- **Catalog subsetting creep** (filtering the universe pre-retrieval) → L1 only operates on the retrieved set; validity guard asserts ⊆ catalog; never queries the catalog to add new ids.
- **History-filter destroys recall** (golds are re-listens) → A/B + P0-driven default + nDCG non-regression gate.
- **Tail-MMR harming nDCG** → restricted to positions 11–20; gate on no top-10 reorder + no nDCG loss.
- **< 20 after filtering** → backfill from next-best in-pool candidates.
- **Non-canonical ids** → canonicalize + validity guard before emit.

## 9. Config knobs
`filter.history_rule` (`auto`|`on`|`off`, default `auto` ← P0 `gold_in_history_rate`), `filter.tail_mmr.{enabled, lambda, start_pos:11, end_pos:20}`, `filter.top_k` (20), `filter.backfill` (true). Defaults/types from F2 loader.

## 10. Definition of Done & review checklist
- [ ] Implements F2 `Filter`; output always ≤20 unique valid canonical ids.
- [ ] Tail-MMR confined to 11–20 (top-10 untouched); nDCG@20 non-regression test green.
- [ ] History-rule A/B both directions tested; default wired from P0; winning direction logged.
- [ ] Validity guard + precheck integration; determinism test green.
- [ ] Code review approved; diversity improves/holds at zero nDCG loss.

## 11. Build order & dependencies
**Built after the reranker (K2; K3 optional).** Depends on: F1, F2, F3, K2/K3, P0 (`gold_in_history_rate`). **Blocks:** D1 (assembles `SubmissionRow` from L1's output) and the first valid submission. On the critical path `… → K2 → L1 → responder → D1` (plan §17).
