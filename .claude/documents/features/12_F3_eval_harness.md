# F3 — Evaluation Harness (official-parity + per-module metrics)

> Foundation module. A **thin wrapper** that scores predictions with the **official
> `music-crs-evaluator` code verbatim** (the only metric used for *selection*), plus a set of
> **lightweight diagnostic metrics** the retrieval/rerank modules need during development
> (recall@K, nDCG@{1,10,20}, mean hit-rank, catalog diversity, Distinct-2, cold/warm splits).
> Consumes `SubmissionRow`/`RankedList` (F2) + ground truth from `make_ground_truth.py`.
> Its eval gate is a **metric-parity test**: our official path == `music-crs-evaluator` output on a
> fixed fixture (within float tolerance). See `000_INDEX.md` for the catalogue.

## 1. Purpose
Provide one trustworthy scoring surface: (a) reproduce the **official leaderboard score locally** by calling/replicating the in-repo evaluator with **zero re-derivation** of any metric formula, asserted by a parity fixture; and (b) emit **diagnostic, segmented metrics** so R/K/L modules can be developed and gated offline without ever re-implementing nDCG. Official metrics are for **model selection**; diagnostic metrics are for **module dev only** and never replace the official number.

## 2. Interface / contract
Lives in `mcrs/eval/harness.py` (+ `mcrs/eval/official.py`, `mcrs/eval/diagnostics.py`). Pure-Python + numpy + pandas; reuses the official metric functions by **import**, never by copy.

```python
# ----- ground-truth record (mirrors make_ground_truth.py output exactly) -----
@dataclass(frozen=True)
class GoldRow:
    session_id: str
    user_id: str
    turn_number: int          # 1..8
    ground_truth_track_id: str   # SINGLE gold per turn (official invariant)

# ----- official, for-selection path (parity-locked) -----
def score_official(
    predictions: list[SubmissionRow],   # F2 contract
    ground_truth: list[GoldRow],
    catalog_size: int,                  # len(all_tracks) — same source the official uses
) -> dict[str, float]:
    """Returns EXACTLY the official macro dict:
       {'ndcg@1','ndcg@10','ndcg@20','catalog_diversity','lexical_diversity','total_catalog_size'}.
       Computed by calling the official metric functions; aggregation replicates evaluate_devset.py."""

# ----- diagnostic, for-module-dev path -----
@dataclass(frozen=True)
class DiagnosticReport:
    overall: dict[str, float]                 # all diagnostic metrics, global
    by_segment: dict[str, dict[str, float]]   # "cold"/"warm" -> same metric keys
    by_turn: dict[int, dict[str, float]]      # turn_number -> metrics (debug)

def score_diagnostic(
    ranked_lists: list[RankedList] | list[SubmissionRow],  # F2; pre-truncation pools allowed
    ground_truth: list[GoldRow],
    catalog_size: int,
    k_recall: tuple[int, ...] = (1, 10, 20, 50, 100, 200, 500),
    k_ndcg:   tuple[int, ...] = (1, 10, 20),
    segment_of: dict[str, str] | None = None,  # session_id -> "cold"|"warm" (from F1/P0)
) -> DiagnosticReport: ...

# ----- low-level diagnostic primitive (shared; imported by P0 + R3–R7 + K2/K3) -----
def recall_at_k(retrieved: list[str], gold: str, k: int) -> float: ...
    # 1.0 if gold in retrieved[:k] else 0.0 (single-gold case); score_diagnostic builds on this.
def hit_rank(retrieved: list[str], gold: str) -> int | None: ...   # 1-indexed rank of gold, or None on miss

# ----- parity assertion helper (used by the gate test) -----
def assert_parity(
    predictions, ground_truth, catalog_size, *, atol: float = 1e-9
) -> None: ...   # raises if score_official(...) != reference evaluator output on the fixture
```

**Wiring contract.** `score_official` is what D1 reports/logs per submission and what selection decisions key on. `score_diagnostic` is consumed by P0 (recall-ceiling table), R3–R7 (recall@K), K2/K3 (nDCG@20 + hit-rank), L1 (diversity hold), S1 (Distinct-2). Inputs are F2 objects (`SubmissionRow` for the official path, `RankedList` *or* `SubmissionRow` for diagnostics — diagnostics may score pre-filter pools deeper than 20). Ground truth comes from `make_ground_truth.py`; F3 loads its JSON into `GoldRow`.

## 3. Dependencies
- **Code (reuse verbatim):** `music-crs-evaluator/metrics/metrics_recsys.py::compute_recsys_metrics` (nDCG), `music-crs-evaluator/metrics/metrics_diversity.py::{compute_catalog_diversity, compute_lexical_diversity}`, and the aggregation order in `music-crs-evaluator/evaluate_devset.py`. The `music-crs-evaluator` package is importable on `sys.path` (or vendored read-only).
- **Data:** dev ground truth JSON from `music-crs-evaluator/make_ground_truth.py` (`exp/ground_truth/devset.json`); catalog size from F1 (`talkpl-ai/...-Track-Metadata` `all_tracks` split) — F3 takes `catalog_size` as an arg so it never re-loads HF in a unit test.
- **Modules:** F2 (`SubmissionRow`, `RankedList`, `Candidate`, `TurnContext`); F1 (canonical id set, cold/warm `segment`).
- **Libs:** `numpy`, `pandas` (the official aggregation uses `pandas.groupby`). No GPU, no model, no API.

## 4. Design & logic

### 4.1 Official path — replicate, do not re-derive
`evaluate_devset.py` is the source of truth. F3 reproduces its exact computation:

1. Per `(session_id, turn_number)` in ground truth, fetch the matching prediction row and call
   `compute_recsys_metrics(preds=predicted_track_ids, gold=[ground_truth_track_id], k_values=[1,10,20])`.
   *Verbatim from `metrics_recsys.py`:* this dispatches **only `get_ndcg`** — `_STANDARD_METRIC_MAP` has `hit`/`mrr`/`map`/`recall`/`precision` **commented out**, so the official selection number is **nDCG@{1,10,20} only** (no official recall). It also raises `ValueError` on duplicate preds or duplicate gold.
2. **nDCG (exact, from `get_ndcg`):** `preds = preds[:k]; dcg = Σ_{i=1..len(preds)} rel_i/log2(i+1)`, `rel_i = 1 if pred∈gold else 0`; `idcg = Σ_{i=1..min(|gold|,k)} 1/log2(i+1)`; return `dcg/idcg` (or `0.0` if `idcg==0`). With a single binary gold this reduces to `1/log2(rank+1)` if gold∈top-k else 0.
3. **Aggregation (exact, from `evaluate_devset.py`):** build a per-turn DataFrame, then
   `df.drop(columns=['session_id']).groupby('turn_number').agg('mean')` (mean **within each turn across all session-turns**), then `.mean(axis=0)` (macro-mean **over the 8 turn buckets**). There is **no session-level grouping** — see §4.4 discrepancy.
4. **Diversity (global, exact):** `catalog_diversity = len(set(all_recommended_ids)) / catalog_size` over the concatenation of **every** row's `predicted_track_ids`; `lexical_diversity = compute_lexical_diversity(all_responses)` = Distinct-2, lowercasing + whitespace tokenization (`len(unique_bigrams)/total_bigrams`, `0.0` if no bigrams). Both computed once over the full file, **not** per turn.
5. Output dict keys match the official file byte-for-byte: `ndcg@1, ndcg@10, ndcg@20, catalog_diversity, lexical_diversity, total_catalog_size`.

F3 does **not** re-implement `get_ndcg` etc.; it imports them. The only F3-owned logic is the *orchestration* (pairing rows + the two `pandas` reductions), and that orchestration is itself parity-tested against running `evaluate_devset.py` end-to-end on a fixture.

### 4.2 Diagnostic path — for module dev only
These are **our** metrics (clearly namespaced), used to drive R/K/L decisions; they are **not** the official score and must never be reported as such.
- **recall@K**, `K ∈ {1,10,20,50,100,200,500}`: single gold ⇒ `1 if gold ∈ preds[:K] else 0`, then averaged. (Identical to `get_hit` for single gold; we may import `get_hit`/`get_recall` from the official module to stay aligned, just over more K and not gated by the comment-out.)
- **nDCG@{1,10,20}**: call the same official `get_ndcg` (parity by construction).
- **mean hit-rank**: over turns where gold ∈ pool, mean 1-indexed rank of the gold (lower = better; plan §4 gate ≤ ~2.5). Turns with a miss are excluded (and `hit_rate` reported alongside so the exclusion is visible).
- **catalog_diversity**, **Distinct-2**: official functions, reused.
- **Aggregation choice:** diagnostics report a **flat macro-mean over all session-turns** (simple mean), *and* a `by_turn` breakdown, *and* a `by_segment` breakdown. The flat mean is deliberately the *simpler* aggregation (not the official turn-bucket macro) because it is the natural per-example diagnostic; the official number remains the only selection signal. Both aggregations are emitted so they can be compared.

### 4.3 Cold/warm segmentation
`segment_of[session_id] ∈ {"cold","warm"}` (from F1/P0 history-length threshold; cold ≈ turn-1 / empty-history). Every diagnostic metric is recomputed on each segment subset → `by_segment`. This is the table P0/R5/cold-warm work reads. The **official** path is never segmented (the leaderboard isn't).

### 4.4 Edge cases & invariants
- **Single gold per turn** asserted on load (`make_ground_truth.py` emits exactly one per turn, turns 1..8). A turn with ≠1 gold is a fixture/data bug → raise.
- **Duplicate predicted ids** → official `compute_recsys_metrics` raises `ValueError`; F3 surfaces it (do not silently dedupe — that would mask an L1/D1 bug). The submission-schema validator (below) is the place dedup is enforced *before* scoring.
- **Missing (session,turn) prediction:** official code does `.iloc[0]` and would `IndexError`. F3 pre-checks coverage (every gold (session,turn) has exactly one prediction row) and raises a clear error first.
- **Empty preds / `idcg==0`** → 0.0 (matches official).
- **`catalog_size` source:** must be the same `all_tracks` length the official uses; passed in to keep tests hermetic.

### 4.5 No-leak / disjointness responsibilities (plan §14)
The harness owns the **harness-level** leak checks (F2 owns the per-record causal asserts):
- **session-disjoint split check:** assert Train/Dev/Blind `session_id` sets are pairwise disjoint (plan §5 step 8, §16 leakage).
- **turn-t ⊆ ≤t:** given a `TurnContext`, assert `len(utterances)==turn_number` and that no scored input contains the gold for turn t (catches a future-turn leak feeding retrieval).
- **gold ∉ inputs:** the gold track id for (session,turn) must not appear in that turn's `history_tids`/query construction inputs.
These run as tests (§7) and as optional runtime asserts in `score_diagnostic(strict=True)`.

## 5. Reuse
- **Pristine pointer (plan §6.3):** the official evaluator package `music-crs-evaluator/` is **pristine** — import its three metric functions and replicate `evaluate_devset.py`'s aggregation. **Keep verbatim; do not port or rewrite the metric math.**
- F3 itself (`mcrs/eval/*`) is **new** (rewrite): the prior tree had only the official script and no diagnostic/segmented layer, no parity test, no F2 typing.
- Config-shape clone from `music-crs-baselines/config/llama1b_*.yaml` for the `eval.*` block.

## 6. Eval & acceptance gate
**Gate = metric-parity (plan §4 pipeline DoD, §14 test #1).** On a fixed fixture (a small frozen `predictions.json` + `ground_truth/devset.json` slice + known `catalog_size`), `score_official(...)` must equal the value produced by running the **unmodified** `music-crs-evaluator/evaluate_devset.py` on the same fixture, for **every** key, within `atol=1e-9` (nDCG keys) / exact for `total_catalog_size`. The gate fails if any key diverges. Secondary: diagnostic nDCG equals official nDCG on the shared k's (since both call `get_ndcg`).

## 7. Tests
- **Metric-parity (the gate):** `assert_parity` on the frozen fixture; also a property test that perturbing one rank changes our nDCG identically to the official function.
- **Aggregation order:** a fixture with uneven turn counts proves we do turn-bucket-mean→macro (official) for `score_official`, and flat-mean for `score_diagnostic`, and that they differ exactly as expected.
- **Single-gold / duplicate guards:** ≠1 gold per turn raises; duplicate preds propagates the official `ValueError`; missing (session,turn) raises a clear coverage error.
- **Recall@K monotonicity:** recall@K non-decreasing in K; recall@1 == nDCG@1 (single binary gold).
- **Hit-rank correctness:** gold at rank r ⇒ hit-rank r; misses excluded and reflected in `hit_rate`.
- **Segmentation:** cold+warm subsets partition the dev set; segment metrics recombine to the flat overall (weighted by count).
- **No-leak suite (plan §14 #2):** session-disjoint split assertion; turn-t⊆≤t; gold∉inputs — each with a passing fixture and a deliberately-leaky fixture that must fail.
- **Submission-schema validator (plan §14 #3):** ≤20, unique, valid catalog ids, all session×turns present, `ensure_ascii=False`; reused by D1.
- **Determinism:** same inputs → byte-identical dicts; no RNG.

## 8. Failure modes & guards
- **Silently reimplementing a metric** → drift from leaderboard. Guard: import official functions; parity test in CI on every push (§14).
- **Wrong aggregation (session vs turn macro)** → local≠leaderboard (plan §16). Guard: aggregation-order test pinned to `evaluate_devset.py`; never "fix" the official macro to a session-macro in `score_official`.
- **Catalog-size mismatch** → wrong `catalog_diversity`. Guard: assert `catalog_size == total_catalog_size` against F1's count; fail loud.
- **Treating diagnostic recall as the official metric** (official has recall commented out). Guard: namespaced keys (`diag.recall@K`) + doc banner; selection code may only read `score_official` keys.
- **Dedupe masking an upstream bug.** Guard: never dedupe inside the scorer; the schema validator dedupes/validates *before* scoring and reports it.
- **Fixture rot** if the official code changes. Guard: pin the evaluator commit/hash; parity test re-runs the actual script, so a real upstream change fails the gate visibly.

## 9. Config knobs
`eval.*` block (schema in F2): `eval.k_recall` (default `[1,10,20,50,100,200,500]`), `eval.k_ndcg` (`[1,10,20]`), `eval.parity_atol` (`1e-9`), `eval.ground_truth_path`, `eval.fixture_path`, `eval.catalog_size` (or `eval.catalog_size_source: F1`), `eval.distinct_n` (`2`), `eval.strict_no_leak` (`true`), `segment.cold_threshold` (shared with F1/P0 for `segment_of`). Values defaulted + type-validated by the F2 loader.

## 10. Definition of Done & review checklist
- [ ] `score_official` returns the exact official key set and **passes `assert_parity`** against unmodified `evaluate_devset.py` on the fixture (the gate).
- [ ] Diagnostic recall@K / nDCG / hit-rank / diversity + cold/warm `by_segment` implemented; official functions imported, not copied.
- [ ] No-leak suite (session-disjoint, turn-t⊆≤t, gold∉inputs) and submission-schema validator green; leaky fixtures fail as intended.
- [ ] Official vs diagnostic separation enforced (namespacing + docstring banner); selection path reads only official keys.
- [ ] Determinism test green; evaluator commit pinned; aggregation-order test pinned.
- [ ] Code review approved; consumed by ≥1 downstream (P0 recall table or D1 report) with no shape change.

## 11. Build order & dependencies
**Built in the foundation tier, after F2 (types) and alongside F1 (catalog size / segment).** Depends on: F2 (`SubmissionRow`/`RankedList`/`TurnContext`), F1 (`catalog_size`, `segment`), and the pristine `music-crs-evaluator/`. **Blocks:** P0 (recall-ceiling probe needs `score_diagnostic`) and every later gate (R7 recall, K2/K3 nDCG+hit-rank, L1 diversity, S1 Distinct-2, D1 official score). On the critical path to first submission: `F1+F2+F3 → P0 → …`.
```

---
**Discrepancy log (official code vs plan §2 prose)**
- **Aggregation:** plan §2 table says nDCG is *"Macro-averaged over turns, then over sessions"*, but `evaluate_devset.py` does **turn-bucket mean → macro-mean over turns only** — there is **no session-level averaging** (it `drop(session_id).groupby(turn_number).mean().mean()`). §2's body prose ("averaged within turn → macro across turns") matches the code; the table's "then over sessions" clause does **not**. `score_official` follows the **code**.
- **Recall is not an official metric:** plan §2/§4 reference recall@K as a gate; in `metrics_recsys.py` only `ndcg` is active — `recall`/`hit`/`mrr`/`map`/`precision` are commented out of `_STANDARD_METRIC_MAP`. So recall is **diagnostic-only**, never in the leaderboard score; F3 keeps it namespaced.
- **nDCG form:** plan writes DCG as `(2^rel−1)/log2(i+1)`; the code uses `rel/log2(i+1)`. Identical for binary `rel∈{0,1}` (the single-gold case here), so values match — but the implemented form is the binary one, not the gain-exponent one.
```
