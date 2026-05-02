"""Shared reward functions for Component-B fine-tuning + W1 correlation study.

Five reward terms (per RecSys_Challenge_Plan §6.1):
    R_retr       — nDCG-shaped retrieval accuracy (uses the leaderboard `get_ndcg`)
    R_rule       — regex/length response checks
    R_user_prof  — user-profile mention (Personalization driver)
    R_judge      — local distilled judge (cross-encoder; loaded lazily, optional here)
    R_format     — well-formed <user_state>...<response> envelope

Plus session-level shaping (R_session) and hard guards:
    catalog-membership filter (zero R_turn on hallucinated track_ids)
    post-fusion dedupe helper

Design constraints:
    - Pure Python; no torch / transformers / mcrs imports at module top-level.
    - Reuse `metrics_recsys.get_ndcg` from the official evaluator (same fn the
      leaderboard uses) to avoid surrogate drift.
    - All file writes must use `ensure_ascii=False` (P0 fix #5).
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_DIR = REPO_ROOT / "music-crs-evaluator"

# Make the leaderboard nDCG fn importable without installing the evaluator pkg.
if str(EVALUATOR_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATOR_DIR))
from metrics.metrics_recsys import get_ndcg  # noqa: E402

# ---------------------------------------------------------------------------
# Regexes (English-only assumption documented in plan §6.1; R_judge captures
# multilingual responses where these regexes underscore).
# ---------------------------------------------------------------------------

ENVELOPE = re.compile(
    r"<user_state>(.*?)</user_state>\s*<response>(.*?)</response>",
    re.DOTALL,
)

BAD = re.compile(
    r"\b(absolutely|fantastic|perfectly|amazing|sorry|unfortunately|i (cannot|can't))\b",
    re.IGNORECASE,
)

WHY = re.compile(
    r"\b(because|since|features|leans|driven by|atmosphere|tempo|groove|"
    r"arrangement|released|from \d{4}|era|decade|vibe|texture|timbre|harmony|melody|rhythm)\b",
    re.IGNORECASE,
)

# Allowed user_state keys (matches the actual prompt envelope at
# music-crs-baselines/mcrs/system_prompts/response_generation_cot_user_state.txt).
ALLOWED_STATE_KEYS = {
    "mood", "intent", "energy", "sonic_pref", "era_pref", "familiarity",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def dedupe_keep_first(ids: Sequence[str]) -> list[str]:
    """Drop duplicates, preserve first-seen order. Mandate at every fusion step."""
    seen: set[str] = set()
    out: list[str] = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def parse_envelope(text: str) -> Optional[tuple[str, str]]:
    """Return (user_state_block, response_block) if envelope parses, else None."""
    m = ENVELOPE.search(text)
    if not m:
        return None
    return m.group(1).strip(), m.group(2).strip()


def parse_user_state(block: str) -> dict[str, str]:
    """Parse `key: value` lines from a <user_state> block. Tolerant: ignores blanks/garbage."""
    state: dict[str, str] = {}
    for line in block.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        if key in ALLOWED_STATE_KEYS:
            state[key] = value.strip()
    return state


def filter_catalog_membership(
    predicted_track_ids: Sequence[str],
    valid_set: set[str],
    retrieval_pool: Sequence[str],
    target_len: int = 20,
) -> list[str]:
    """A6: drop hallucinated UUIDs, backfill from the retrieval pool.

    Order: keep valid IDs in original order, then append valid retrieval-pool IDs
    not already used until length == target_len.
    """
    seen: set[str] = set()
    out: list[str] = []
    for tid in predicted_track_ids:
        if tid in valid_set and tid not in seen:
            out.append(tid)
            seen.add(tid)
        if len(out) >= target_len:
            return out
    for tid in retrieval_pool:
        if tid in valid_set and tid not in seen:
            out.append(tid)
            seen.add(tid)
        if len(out) >= target_len:
            break
    return out


# ---------------------------------------------------------------------------
# Per-turn reward terms
# ---------------------------------------------------------------------------

def r_retr(predicted_track_ids: Sequence[str], gold_track_id: str) -> float:
    """Weighted nDCG@{1,10,20}. Uses the same `get_ndcg` as the leaderboard."""
    if not gold_track_id:
        return 0.0
    g = [gold_track_id]
    preds = list(predicted_track_ids)
    return (
        0.5 * get_ndcg(g, preds, 20)
        + 0.3 * get_ndcg(g, preds, 10)
        + 0.2 * get_ndcg(g, preds, 1)
    )


def r_rule(
    response: str,
    top1_meta: Optional[dict] = None,
    user_state: Optional[dict] = None,
    history_text: str = "",
) -> float:
    """Cheap regex/length response checks. Sub-scores cap at 1.0.

    Each sub-check is independent. If `top1_meta` or `user_state` is None, those
    sub-checks contribute 0 — used during W1 correlation study before A1/A5 land.
    """
    if not response:
        return 0.0
    r = response
    rl = r.lower()
    score = 0.0

    # Track-name + artist mentions (require top1_meta).
    if top1_meta:
        name = (top1_meta.get("track_name") or "").lower()
        artist = (top1_meta.get("artist_name") or "").lower()
        if name and name in rl:
            score += 0.20
        if artist and artist in rl:
            score += 0.15

    # "Why" clause (musical-detail vocabulary).
    if WHY.search(r):
        score += 0.15

    # User-state echo (lemma-level: only values not in the user query).
    if user_state:
        for v in user_state.values():
            if isinstance(v, str) and v not in {"unknown", ""}:
                # Split commas so multi-value fields count once per match.
                for tok in re.split(r"[,;]", v):
                    tok = tok.strip().lower()
                    if tok and tok in rl:
                        score += 0.10
                        break
                if score >= 1.0:
                    break

    # Length band (60-110 words, per plan).
    n_words = len(r.split())
    if 60 <= n_words <= 110:
        score += 0.10

    # 1-4 sentence terminators.
    n_term = r.count(".") + r.count("?") + r.count("!")
    if 1 <= n_term <= 4:
        score += 0.10

    # No banned phrases.
    if not BAD.search(r):
        score += 0.10

    # History grounding: at least one ≥5-char history token appears.
    if history_text:
        for tok in history_text.lower().split():
            if len(tok) >= 5 and tok in rl:
                score += 0.10
                break

    return min(score, 1.0)


def r_user_prof(response: str, user_profile: Optional[dict] = None) -> float:
    """User-profile (country, age_group, gender) mention. Personalization driver."""
    if not response or not user_profile:
        return 0.0
    rl = response.lower()
    score = 0.0
    country = (user_profile.get("country_name") or "").strip().lower()
    age = (user_profile.get("age_group") or "").strip().lower()
    gender = (user_profile.get("gender") or "").strip().lower()
    if country and country in rl:
        score += 0.5
    if age:
        if age in rl or age.replace("-", " ") in rl or age.replace("_", " ") in rl:
            score += 0.3
    if gender and gender in rl:
        score += 0.2
    return min(score, 1.0)


def r_format(text: str) -> float:
    """1.0 iff the <user_state>...</user_state>\\s*<response>...</response> envelope parses
    AND the user_state block contains at least one allowed key."""
    parsed = parse_envelope(text)
    if not parsed:
        return 0.0
    state_block, _resp = parsed
    state = parse_user_state(state_block)
    return 1.0 if state else 0.0


def r_judge_stub(*args, **kwargs) -> float:
    """Placeholder for the local distilled cross-encoder judge.

    Returns 0.0 unconditionally — kept as a back-compat fallback for tests
    and pre-judge code paths. Real callers should use `DistilledJudge`
    (below) once a checkpoint is trained via `scripts/train_distilled_judge.py`.
    """
    return 0.0


class DistilledJudge:
    """Lazy-loading runtime wrapper around a trained distilled cross-encoder.

    Replaces `r_judge_stub` in the W6/W7 GRPO reward closure (Option B
    refactor). The model is a regression cross-encoder trained on
    `data/reward_calibration_anchors.parquet` to predict a [0,1]-normalized
    judge score. See `scripts/train_distilled_judge.py` for training.

    Lazy load: the model is NOT pulled from Hub until the first `score()`
    call. This keeps test imports cheap and lets the W6 reward closure
    instantiate the judge at trainer-init time without an immediate Hub
    download.

    Graceful degradation: when `checkpoint=None`, `score()` returns 0.0 —
    same behavior as the legacy stub. Lets all 297 existing tests pass
    without a checkpoint installed and lets pre-W7 code paths use the same
    interface as post-W7 code.
    """

    def __init__(
        self,
        checkpoint: "str | None" = None,
        max_length: int = 512,
        device: "str | None" = None,
    ) -> None:
        self.checkpoint = checkpoint
        self.max_length = int(max_length)
        self._device = device
        self._model = None
        self._tokenizer = None
        # Score cache — keyed by (context, response) hash. GRPO rollouts
        # often see the same (prompt, completion) twice during evaluation;
        # deduping there saves ~30-50% of judge-forward-pass time.
        self._cache: dict[int, float] = {}

    def _ensure_loaded(self) -> bool:
        """Load model + tokenizer on first call. Returns False if no checkpoint
        was configured (graceful no-op path)."""
        if self.checkpoint is None:
            return False
        if self._model is not None:
            return True
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError:
            # Test environment without torch installed — silently degrade.
            return False
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.checkpoint,
        ).to(self._device).eval()
        return True

    def warmup(self) -> bool:
        """Force model load + a single forward pass.

        Deep-review P1-4 fix: the trainer's first reward step would
        otherwise eat the ~80MB cross-encoder Hub download. Call after
        instantiation to amortize the load before training starts.

        Returns True if the model loaded; False on no-checkpoint fallback.
        """
        if not self._ensure_loaded():
            return False
        # Single dummy forward to trigger CUDA/MPS allocation + first kernel.
        try:
            self.score("warmup-context", "warmup-response")
        except Exception:
            pass
        return True

    def score(self, context: str, response: str) -> float:
        """Score a single (context, response) pair, returning a float in [0, 1].

        When no checkpoint is configured (or torch is unavailable), returns
        0.0 — preserves stub semantics for back-compat.

        Calibration note (sanity-pass P1 finding): the regression head is
        trained with `problem_type="regression"` + MSE loss DIRECTLY against
        [0,1] labels — there is no sigmoid in the loss path. The head's
        raw logits are already calibrated to the [0,1] range. Applying
        sigmoid at inference would squash a calibrated 0.0 → 0.5, 1.0 →
        0.731, collapsing the dynamic range ~3×. So we DO NOT apply
        sigmoid; the [0,1] clamp is the safety net for occasional out-of-
        band predictions.
        """
        if not self._ensure_loaded():
            return 0.0
        key = hash((context, response))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        import torch  # safe — _ensure_loaded already imported it
        with torch.no_grad():
            enc = self._tokenizer(
                context, response,
                truncation=True, padding=True,
                max_length=self.max_length, return_tensors="pt",
            ).to(self._device)
            logits = self._model(**enc).logits  # (1, 1) regression head
            # No sigmoid — see calibration note in the docstring.
            raw = float(logits.squeeze().cpu().item())
        # Clamp [0, 1] as the safety net for occasional out-of-band logits.
        raw = max(0.0, min(1.0, raw))
        self._cache[key] = raw
        return raw

    def score_batch(
        self, contexts: "list[str]", responses: "list[str]",
        batch_size: int = 16,
    ) -> "list[float]":
        """Batched score for a list of (context, response) pairs.

        Same back-compat: returns a list of 0.0s when no checkpoint is
        configured. Useful for the W6 reward closure which receives a list
        of completions per group.
        """
        if not self._ensure_loaded():
            return [0.0] * len(contexts)
        out: list[float] = []
        import torch
        for start in range(0, len(contexts), batch_size):
            batch_ctx = contexts[start:start + batch_size]
            batch_resp = responses[start:start + batch_size]
            with torch.no_grad():
                enc = self._tokenizer(
                    batch_ctx, batch_resp,
                    truncation=True, padding=True,
                    max_length=self.max_length, return_tensors="pt",
                ).to(self._device)
                logits = self._model(**enc).logits  # (B, 1)
                # No sigmoid — model is regression-trained against [0,1] labels
                # directly; sigmoid at inference would collapse calibration.
                vals = logits.squeeze(-1).cpu().tolist()
            for v in vals:
                v = max(0.0, min(1.0, float(v)))
                out.append(v)
        return out


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

# Weights — Option B refactor v4 (gap-analysis Step 3 user_profile pipe).
#
# v0 (pre-W1):       R_retr=0.50  R_judge=0.20  R_rule=0.20  R_format=0.05  R_user_prof=0.05
# v1 (Option A, W1): R_retr=0.70  R_judge=0.10  R_rule=0.15  R_format=0.05  R_user_prof=0.00
# v2 (Option B):     R_retr=0.40  R_judge=0.30  R_rule=0.15  R_format=0.10  R_user_prof=0.05
# v3 (P1-6 honesty): R_retr=0.40  R_judge=0.30  R_rule=0.20  R_format=0.10  R_user_prof=0.00
# v4 (NOW):          R_retr=0.40  R_judge=0.30  R_rule=0.15  R_format=0.10  R_user_prof=0.05
#
# Rationale for v4 (gap-analysis Step 3):
#   - W_USER_PROF 0.00 → 0.05 RESTORED. The data path is now wired end-to-end
#     (`build_reward_dataset.py` looks up user from User-Metadata DB and emits
#     `user_profile_json`; `build_grpo_dataset.py` carries it through;
#     reward closures in colab/31p/32/33 JSON-decode and pass it). With the
#     data piped, R_user_prof can carry actual Personalization gradient
#     (one of the two Gemini judge axes per project_blind_judge_gemini).
#   - W_RULE 0.20 → 0.15: shift the 0.05 back from rule (where v3 parked it)
#     to user_prof. Rule is mechanical regex with AUC 0.51 vs Gemini (W1);
#     under-weighting it slightly is a feature, not a regression.
W_RETR = 0.40
W_JUDGE = 0.30
W_RULE = 0.15
W_FORMAT = 0.10
W_USER_PROF = 0.05


def compose_r_turn(
    predicted_track_ids: Sequence[str],
    gold_track_id: str,
    response_text: str,
    valid_catalog: Optional[set[str]] = None,
    top1_meta: Optional[dict] = None,
    user_state: Optional[dict] = None,
    user_profile: Optional[dict] = None,
    history_text: str = "",
    judge_score: Optional[float] = None,
    judge_trust: float = 1.0,
    include_format: bool = True,
    group_responses: Optional[Sequence[str]] = None,
) -> dict[str, float]:
    """Compute R_turn and components.

    Hard guards (return R_turn=0.0):
        - any predicted_track_id ∉ valid_catalog (if valid_catalog provided)

    `group_responses` (W6-review Option B): when provided with > 1 element,
    adds a small intra-rollout diversity bonus computed via
    `lex_div_distinct2`. This wires the conversation-data signal that
    `compose_r_session` codified but never reached training. Cap: +0.05.

    Returns dict with all sub-scores so callers can analyse component-wise.
    """
    rr = r_retr(predicted_track_ids, gold_track_id)
    ru = r_rule(response_text, top1_meta=top1_meta, user_state=user_state, history_text=history_text)
    rp = r_user_prof(response_text, user_profile=user_profile)
    rf = r_format(response_text) if include_format else 0.0
    rj = float(judge_score) if judge_score is not None else 0.0

    catalog_ok = True
    if valid_catalog is not None:
        for tid in predicted_track_ids:
            if tid not in valid_catalog:
                catalog_ok = False
                break

    if not catalog_ok or (include_format and rf == 0.0):
        # Hard zero — see §6.1 multiplicative penalty + §6.6 guardrails.
        r_turn = 0.0
    else:
        r_turn = (
            W_RETR * rr
            + W_JUDGE * judge_trust * rj
            + W_RULE * ru
            + W_FORMAT * rf
            + W_USER_PROF * rp
        )

    # Across-rollout diversity bonus (deep-review P1-1 fix).
    #
    # OLD (v2): used `lex_div_distinct2` over the JOINED text of all rollouts
    # → identical rollouts gave a ~0.013 floor because within-text bigrams
    # always vary. WRONG signal — we want "did this rollout differ from peers?",
    # not "is the joined corpus diverse?".
    #
    # NEW (v3): pairwise Jaccard distance over per-rollout bigram sets,
    # averaged. Identical rollouts → 0 (all pairwise sets equal → distance 0).
    # Completely disjoint bigrams → 1. Symmetric, principled.
    r_lex_div_group = 0.0
    if group_responses is not None and len(group_responses) > 1:
        r_lex_div_group = min(lex_div_pairwise(list(group_responses)), 1.0)
        # Additive bonus capped at +0.05 — keeps the diversity signal a
        # tie-breaker, NOT a primary objective. Hard-zero rows (broken format
        # / catalog) still get 0 — bonus only added on top of non-zero base.
        if r_turn > 0.0:
            r_turn = r_turn + 0.05 * r_lex_div_group

    # Deep-review P0-3 fix: clamp to [0, 1]. Without this, max base
    # (0.40+0.30+0.20+0.10 = 1.00) plus +0.05 bonus = 1.05, breaking the
    # gate threshold semantics in colab/32 cell 14 where Δ R_turn ≥ +0.03
    # could be inflated by up to +0.05 of unclamped bonus.
    r_turn = min(1.0, max(0.0, r_turn))

    return {
        "r_turn": r_turn,
        "r_retr": rr,
        "r_rule": ru,
        "r_user_prof": rp,
        "r_format": rf,
        "r_judge": rj,
        "r_lex_div_group": r_lex_div_group,
        "catalog_ok": float(catalog_ok),
    }


def compose_r_turn_no_judge(components: dict[str, float]) -> float:
    """W1 correlation-gate target: R_turn with the R_judge term removed.

    Used to test whether mechanical terms alone correlate with the Gemini-anchored
    composite (plan §6.2 step 3 — bug fix vs the earlier circular gate).
    """
    if components["r_format"] == 0.0 or components["catalog_ok"] == 0.0:
        return 0.0
    return (
        W_RETR * components["r_retr"]
        + W_RULE * components["r_rule"]
        + W_FORMAT * components["r_format"]
        + W_USER_PROF * components["r_user_prof"]
    )


# ---------------------------------------------------------------------------
# Session-level shaping
# ---------------------------------------------------------------------------

def lex_div_distinct2(responses: Sequence[str]) -> float:
    """Distinct-2 across all responses, joined; matches metrics_diversity behavior."""
    text = " ".join(r for r in responses if r)
    toks = text.split()
    if len(toks) < 2:
        return 0.0
    bigrams = list(zip(toks, toks[1:]))
    if not bigrams:
        return 0.0
    return len(set(bigrams)) / len(bigrams)


def lex_div_pairwise(responses: Sequence[str]) -> float:
    """Pairwise across-rollout bigram distance (Jaccard), averaged.

    For each pair of responses, compute 1 - |a ∩ b| / |a ∪ b| over their
    bigram sets. Returns the mean across all pairs.

    Used for the GRPO group_responses bonus (deep-review P1-1 fix). Unlike
    `lex_div_distinct2`, this is zero when all rollouts are identical and
    one when their bigram sets are disjoint — the right signal for "did
    this rollout differ from peers?" in GRPO.
    """
    if len(responses) < 2:
        return 0.0

    def _bigrams(text: str) -> set:
        toks = (text or "").split()
        return set(zip(toks, toks[1:]))

    sets = [_bigrams(r) for r in responses]
    distances: list[float] = []
    n = len(sets)
    for i in range(n):
        for j in range(i + 1, n):
            union = sets[i] | sets[j]
            if not union:
                continue
            intersection = sets[i] & sets[j]
            jaccard = len(intersection) / len(union)
            distances.append(1.0 - jaccard)
    if not distances:
        return 0.0
    return sum(distances) / len(distances)


def cat_div(track_ids_per_turn: Sequence[Sequence[str]], catalog_size: int) -> float:
    """Catalog Diversity = |unique tracks| / catalog_size."""
    if catalog_size <= 0:
        return 0.0
    flat = [tid for ids in track_ids_per_turn for tid in ids]
    return len(set(flat)) / catalog_size


def compose_r_session(
    r_turns: Sequence[float],
    r_judge_per_turn: Sequence[float],
    responses: Sequence[str],
    track_ids_per_turn: Sequence[Sequence[str]],
    catalog_size: int,
) -> dict[str, float]:
    """Session shaping per plan §6.1.

    R_session = mean(R_turn)
              + 0.10 · monotonicity(R_judge)
              − 0.05 · var(R_turn)
              + 0.10 · clipped(LexDiv across responses)
              + 0.05 · clipped(CatDiv across all top-20)
    """
    if not r_turns:
        return {"r_session": 0.0, "mean_r_turn": 0.0, "var_r_turn": 0.0,
                "monotonicity": 0.0, "lex_div": 0.0, "cat_div": 0.0}

    mean_r = sum(r_turns) / len(r_turns)
    var_r = sum((x - mean_r) ** 2 for x in r_turns) / len(r_turns)

    # Monotonicity: count non-decreasing pairs / total pairs ∈ [0,1].
    if len(r_judge_per_turn) >= 2:
        n_inc = sum(1 for a, b in zip(r_judge_per_turn, r_judge_per_turn[1:]) if b >= a)
        mono = n_inc / (len(r_judge_per_turn) - 1)
    else:
        mono = 0.0

    ld = min(lex_div_distinct2(responses), 1.0)
    cd = min(cat_div(track_ids_per_turn, catalog_size), 1.0)

    r_sess = (
        mean_r
        + 0.10 * mono
        - 0.05 * var_r
        + 0.10 * ld
        + 0.05 * cd
    )
    return {
        "r_session": r_sess,
        "mean_r_turn": mean_r,
        "var_r_turn": var_r,
        "monotonicity": mono,
        "lex_div": ld,
        "cat_div": cd,
    }


# ---------------------------------------------------------------------------
# I/O helpers — every JSON write must use ensure_ascii=False (plan P0 fix #5)
# ---------------------------------------------------------------------------

def dump_json(obj, path: Path | str) -> None:
    """Single-source helper: writes JSON with ensure_ascii=False, indent=2."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Self-test (run as `python scripts/reward_fns.py`)
# ---------------------------------------------------------------------------

def _selftest() -> None:
    print("[reward_fns] self-test")

    # Hit at rank 1 → R_retr = 1.0
    assert abs(r_retr(["a", "b", "c"], "a") - 1.0) < 1e-9
    # Hit at rank 2 → 0.5*nDCG@20 + 0.3*nDCG@10 + 0.2*0
    rr2 = r_retr(["b", "a", "c"], "a")
    assert 0.5 < rr2 < 0.7, rr2
    # Miss → 0
    assert r_retr(["b", "c"], "a") == 0.0

    # R_rule with full match
    meta = {"track_name": "Holocene", "artist_name": "Bon Iver"}
    state = {"mood": "reflective", "energy": "low"}
    resp = (
        "Bon Iver's Holocene leans into a layered arrangement and slow tempo, "
        "matching the reflective mood you described. Want a sparser version next?"
    )
    score = r_rule(resp, top1_meta=meta, user_state=state, history_text="winding down dreamy")
    assert score >= 0.7, f"r_rule too low: {score}"

    # Banned-phrase penalty
    bad = resp + " absolutely amazing!"
    assert r_rule(bad, top1_meta=meta, user_state=state) < score

    # User profile
    profile = {"country_name": "Japan", "age_group": "25-34", "gender": "F"}
    assert r_user_prof("This Japan-released track suits you", profile) >= 0.5

    # Format envelope
    fmt = "<user_state>\nmood: calm\nenergy: low\n</user_state>\n<response>Hello.</response>"
    assert r_format(fmt) == 1.0
    assert r_format("just text, no envelope") == 0.0
    # Envelope present but state block empty → 0
    assert r_format("<user_state></user_state><response>x</response>") == 0.0

    # Dedupe
    assert dedupe_keep_first(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]

    # Catalog filter
    valid = {"a", "b", "c", "d"}
    out = filter_catalog_membership(["zzz", "a", "yyy", "b"], valid, ["c", "d", "a"], target_len=3)
    assert out == ["a", "b", "c"], out

    # Compose
    comps = compose_r_turn(
        predicted_track_ids=["track1", "track2"],
        gold_track_id="track1",
        response_text=fmt.replace("Hello.", resp),
        top1_meta=meta,
        user_state=state,
        user_profile=profile,
        history_text="winding down dreamy",
        valid_catalog={"track1", "track2"},
    )
    assert comps["catalog_ok"] == 1.0
    assert comps["r_format"] == 1.0
    assert 0.0 < comps["r_turn"] <= 1.0, comps

    # Hard zero on hallucinated track
    bad_comp = compose_r_turn(
        predicted_track_ids=["hallucinated"],
        gold_track_id="track1",
        response_text=fmt,
        valid_catalog={"track1"},
    )
    assert bad_comp["r_turn"] == 0.0

    # R_turn_no_judge subtraction sanity
    rtnj = compose_r_turn_no_judge(comps)
    assert rtnj == comps["r_turn"]  # judge term is 0 anyway in this fixture

    # Session shaping
    sess = compose_r_session(
        r_turns=[0.4, 0.5, 0.55, 0.55, 0.6, 0.6, 0.6, 0.65],
        r_judge_per_turn=[0.3, 0.4, 0.5, 0.5, 0.55, 0.6, 0.6, 0.65],
        responses=["The track has groove and warmth."] * 8,
        track_ids_per_turn=[["a", "b"], ["c", "d"], ["e", "f"], ["g", "h"],
                            ["i", "j"], ["k", "l"], ["m", "n"], ["o", "p"]],
        catalog_size=50000,
    )
    assert sess["mean_r_turn"] > 0
    assert sess["monotonicity"] == 1.0  # all non-decreasing

    print("[reward_fns] all self-tests passed")


if __name__ == "__main__":
    _selftest()
