# Loop Experiments Log — Pre-Registration Journal

Pre-registration journal for the autonomous research loop (see RESEARCH_CHARTER.md).
NOTE: distinct from `experiments_log.md` (the pre-existing W1–W8 gate decision log).

Each experiment is pre-registered BELOW (hypothesis + gate + decision rule) BEFORE the run.
After results, fill the `--- run ---` block. INCONCLUSIVE is a first-class verdict.
Template:

## EXP-NNN — <lever> — YYYY-MM-DD
Hypothesis:   <falsifiable claim>
Lever:        recall | reranker | responder
Change:       <config NNN / code diff summary>
Pre-registered gate:  <metric> on <Tier-1 signal> must beat <baseline> by <delta>
Baseline:     <honest number + source>
Decision rule: PASS -> keep/promote; FAIL -> revert;
               INCONCLUSIVE (|Δ| < <noise band>) -> no change, counts toward switch rule
Smoke:        pytest <paths> green before handoff
--- run ---
Result:       <pasted RESULTS_JSON>
Verdict:      PASS | FAIL | INCONCLUSIVE
Local→Blind:  local Δ <>, blind Δ <>   (only if submitted)
Memory:       <file written/updated>

---

## BUILD NOTES / open refinements (2026-06-13)
- nb74 emitter reports a RETRIEVAL-ONLY composite (cat_div/lex_div = 0; it is the retrieval
  gate). The loop MUST judge nb74 on `ndcg@20` (turn-1), not its `composite` field.
- nb80 emitter records BLIND (Tier-2) metrics the human pastes from the CodaBench result page.
  It is NOT the Tier-1 LLM signal — blind scores can't be computed locally.
- FOLLOW-UP (before the first responder iteration): wire a RESULTS_JSON emitter onto the DEV
  responder + OFFLINE Gemini judge path (gemini_judge_responses.py — likely nb79/nb75) so the
  Tier-1 LLM-axis reward is captured locally and leak-free. The plan wired nb74+nb80 only.

## EXP-000 — example (do not run) — 2026-06-13
Hypothesis:   Gemini responder on config-203 track_ids (nDCG 0.30, no Q*) lifts composite to ~0.47.
Lever:        responder
Change:       config 205 = 203 recall track_ids + gemini_responder.py
Pre-registered gate:  blindA composite must beat 0.44 (config 204) by > 0.05 (noise band)
Baseline:     config 204 composite 0.44 (benchmarks.md / memory project_blind_a_state_responder_lever)
Decision rule: PASS -> promote to blind candidate; FAIL -> revert; INCONCLUSIVE (|Δ|<0.05) -> next hypothesis
Smoke:        n/a (example)
--- run ---
Result:       (not run)
Verdict:      (n/a)

---

## EXP-001 — responder base swap (203 tracks + Gemini) — 2026-06-13
Hypothesis:   Putting config-203 Blind-A track_ids (Blind nDCG 0.30, no intent_state Q*) under
              the SAME Gemini best-of-3 responder (LLM ~4.2) lifts Blind composite from 0.44
              (config 204, nDCG 0.24) to ~0.47. Both axes are independently Blind-grounded
              (203 nDCG 0.30 from its 0.37 submission; Gemini LLM ~4.2 from the 204 submission),
              so this banks the best-known config to the leaderboard.
Lever:        responder base / recall (drop intent_state Q* by using the 203 recall stack)
Change:       config-203 Blind-A track_ids + nb80 Gemini responder swap (TOP_N=1 known-good,
              BEST_OF=3, plain prompt, STRUCTURED_PERSONALITY=False). NO code change — track_ids
              untouched, only predicted_response regenerated. (TOP_N=1 is the v5-kto-winning
              setting that produced LLM ~4.2; TOP_N=3 was a speculative dilution risk + slower.)
Pre-registered gate:  Blind-A composite (CodaBench) ≥ 0.44 (current best, config 204); target ≈ 0.47.
Baseline:     config 204 = Blind 0.44 (nDCG 0.24 / Cat 0.03 / Lex 0.79 / LLM 4.2);
              config 203 = Blind 0.37 (nDCG 0.30 / LLM 2.85 v5-kto).
              Source: memory project_blind_a_state_responder_lever_2026_06_11 + 203/204 yaml headers.
Decision rule: PASS if composite ≥ 0.46 → new best (update benchmarks.md + memory).
              INCONCLUSIVE if 0.42–0.46 (within ±0.05 noise of 0.44) → still adopt the 203 base
              (its nDCG edge is Blind-proven) but flag the LLM axis didn't separate; counts
              toward the switch rule.
              FAIL if < 0.42 → responder regressed on 203 tracks; investigate before next submit.
De-risk:      nb80 SMOKE (--limit 5) first to confirm the Gemini responder fires cleanly on 203
              tracks before the full 80-row run + submit (protects the scarce Blind slot).
Budget:       1 of 3 weekly Blind slots (0/3 used as of 2026-06-13).
Smoke:        n/a (no code change; harness pytest suite already green, 17/17).
Reviews:      code-review N/A (no diff); RecSys-researcher review REQUIRED on the verdict before banking.
--- run ---
Result:       (pending human run: nb80 Gemini swap on config-203 Blind-A track_ids)
Verdict:      (pending)
