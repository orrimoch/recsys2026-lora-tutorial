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
