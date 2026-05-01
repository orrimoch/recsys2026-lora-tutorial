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

    The real implementation loads music-crs-baselines/mcrs/response_rerankers/
    reward_reranker.py and scores (context, response). Returns 0.0 here so callers
    can compose without forcing a model load during the W1 correlation study.
    """
    return 0.0


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

# Weights from RecSys_Challenge_Plan §6.1, REVISED post-W1 (Option B).
#
# Original (pre-W1):  R_retr=0.50  R_judge=0.20  R_rule=0.20  R_format=0.05  R_user_prof=0.05
# Revised (Option B): R_retr=0.70  R_judge=0.10  R_rule=0.15  R_format=0.05  R_user_prof=0.00
#
# Rationale (see data/reward_gate_results.json):
#   - W1 gate empirically falsified that R_rule predicts GPA labels on gold data
#     (Spearman 0.012, AUC 0.507 — random). R_rule mass-shifted to R_retr.
#   - R_user_prof had mean 0.009 on gold data — gold assistants never mention
#     user country/age/gender. Term dropped (kept callable for future analysis).
#   - R_judge weight kept low (0.10) with trust-gating until cross-encoder
#     calibrated against an external signal (Blind-B scores, future Gemini API).
#   - R_retr weight raised to 0.70 because Source B sanity confirmed it IS the
#     leaderboard nDCG term (Spearman = 1.0) and Source C confirmed nDCG term
#     dominates the leaderboard composite (max delta 0.005).
W_RETR = 0.70
W_JUDGE = 0.10
W_RULE = 0.15
W_FORMAT = 0.05
W_USER_PROF = 0.00


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
) -> dict[str, float]:
    """Compute R_turn and components.

    Hard guards (return R_turn=0.0):
        - any predicted_track_id ∉ valid_catalog (if valid_catalog provided)

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

    return {
        "r_turn": r_turn,
        "r_retr": rr,
        "r_rule": ru,
        "r_user_prof": rp,
        "r_format": rf,
        "r_judge": rj,
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
