"""intent_state — Q* query rewriter for retrieval (plan §3/§5, W1.a).

Turns each (session, turn) into a single SELF-CONTAINED retrieval query `query_star`,
seeded from the session's persistent anchors so it is non-empty even at turn-1/Blind:

    state0 = parse(listener_goal)  (the non-decaying TARGET)  +  prior(user_profile)
    state_t = apply_delta(state_{t-1}, turn_t)               (recent words win on conflict)

It also emits `state` (the structured intent) and `info_content` (0..1, how much NEW intent
the recent turn adds — feeds confidence-adaptive fusion weights, plan §7). The rewrite runs on
the Gemini API (gemini-2.5-flash-lite, temp 0), cached by (session_id, turn). On any failure it
falls back to the caller's raw query, so the channels always have a usable query.
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional

DEFAULT_MODEL = "gemini-2.5-flash-lite"

DEFAULT_SYSTEM_INSTRUCTION = (
    "You rewrite one turn of a music-recommendation conversation into a single, self-contained "
    "retrieval query for finding catalog tracks. You are given the listener's GOAL (the fixed "
    "session target), PROFILE (taste prior), HISTORY (prior turns), and the RECENT user message.\n"
    "Output STRICT JSON only, no prose, no code fences:\n"
    '{"state": {"genre": "", "mood": "", "era": "", "energy": "", "region": "", "instrument": "", '
    '"liked": [], "disliked": []}, "query_star": "<one self-contained query of the concrete musical '
    'facets the user wants RIGHT NOW>", "info_content": <0..1>}\n'
    "Rules: query_star MUST be self-contained — resolve coreference ('that', 'more like that') using "
    "HISTORY+GOAL, and contain only concrete musical facets (genre/era/mood/energy/region/artist-"
    "style/use-case), no chit-chat. It MUST be non-empty even when RECENT is empty or vague (fall "
    "back to the GOAL). The GOAL is the non-decaying target; the user's RECENT words win on conflict; "
    "a correction adds to disliked[]. info_content = how much NEW intent RECENT adds vs the GOAL "
    "(1.0 = a fully new substantive request, 0.0 = empty/'more')."
)


def parse_intent_state(raw: str, fallback_query: str = "") -> dict:
    """Parse the rewriter's JSON output into {query_star, state, info_content}.

    Robust to code fences and surrounding prose. On any failure (or empty/blank
    query_star) returns `fallback_query` so the channel still has a usable query."""
    out = {"query_star": fallback_query, "state": {}, "info_content": 0.0}
    if not raw:
        return out
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    obj = re.search(r"\{.*\}", text, re.DOTALL)
    if obj:
        text = obj.group(0)
    try:
        d = json.loads(text)
    except Exception:
        return out
    qs = d.get("query_star")
    if isinstance(qs, str) and qs.strip():
        out["query_star"] = qs.strip()
    if isinstance(d.get("state"), dict):
        out["state"] = d["state"]
    try:
        out["info_content"] = float(d.get("info_content"))
    except (TypeError, ValueError):
        pass
    return out


def build_intent_user_content(
    history_text: str, recent_text: str, goal_text: str, profile: Optional[dict] = None
) -> str:
    """Assemble the [GOAL]/[PROFILE]/[HISTORY]/[RECENT] user content for the rewriter."""
    p = profile or {}
    culture = str(p.get("preferred_musical_culture") or "").strip()
    demo = ", ".join(str(p.get(k)) for k in ("age_group", "country_name", "gender") if p.get(k))
    prof = culture + (" | " + demo if demo else "")
    return (
        f"[GOAL] {goal_text or ''}\n"
        f"[PROFILE] {prof.strip()}\n"
        f"[HISTORY]\n{history_text or ''}\n"
        f"[RECENT] {recent_text or ''}"
    )


class IntentStateRewriter:
    """Gemini-backed Q* rewriter with (session_id, turn) caching. Inject `client` for
    tests; in production it lazy-loads GeminiClient (gemini-2.5-flash-lite, temp 0)."""

    def __init__(
        self,
        client=None,
        system_instruction: Optional[str] = None,
        cache_dir: str = "./cache",
        model: str = DEFAULT_MODEL,
        max_output_tokens: int = 256,
    ):
        self._client = client
        self._model = model
        self._max_out = max_output_tokens
        self.system_instruction = system_instruction or DEFAULT_SYSTEM_INSTRUCTION
        self.cache_root = os.path.join(cache_dir, "intent_state")
        os.makedirs(self.cache_root, exist_ok=True)
        self._mem: dict[tuple[str, int], dict] = {}

    def _ensure_client(self):
        if self._client is None:
            from .gemini_propose import GeminiClient
            self._client = GeminiClient(
                model=self._model, temperature=0.0, max_output_tokens=self._max_out)
        return self._client

    def _cache_path(self, session_id, turn) -> str:
        safe = re.sub(r"[^A-Za-z0-9_\-]", "_", str(session_id))
        return os.path.join(self.cache_root, f"{safe}__{int(turn)}.json")

    def rewrite(self, session_id, turn, history_text, recent_text, goal_text,
                profile=None, fallback_query: str = "") -> dict:
        key = (str(session_id), int(turn))
        if key in self._mem:
            return self._mem[key]
        path = self._cache_path(session_id, turn)
        if os.path.isfile(path):
            try:
                with open(path) as f:
                    r = json.load(f)
                self._mem[key] = r
                return r
            except Exception:
                pass
        user = build_intent_user_content(history_text, recent_text, goal_text, profile)
        try:
            raw = self._ensure_client().generate(self.system_instruction, user)
        except Exception:
            raw = ""
        r = parse_intent_state(raw, fallback_query=(fallback_query or recent_text or ""))
        self._mem[key] = r
        try:
            with open(path, "w") as f:
                json.dump(r, f, ensure_ascii=False)
        except Exception:
            pass
        return r

    def batch_rewrite(self, session_ids, turns, history_texts, recent_texts, goal_texts,
                      profiles, fallback_queries=None) -> list[dict]:
        fb = fallback_queries or [""] * len(session_ids)
        return [self.rewrite(s, t, h, rc, g, p, f) for s, t, h, rc, g, p, f in
                zip(session_ids, turns, history_texts, recent_texts, goal_texts, profiles, fb)]
