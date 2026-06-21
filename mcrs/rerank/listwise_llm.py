"""LLM listwise reranker — stage 2 of a two-stage rerank (K2 LGBM -> LLM).

Stage 1 (K2 LGBM) coarse-orders the fused 500-pool. Stage 2 hands the LGBM top
`window` (default 50) to an LLM that reorders them listwise, RankGPT-style, so a
gold the LGBM ranked 21-50 can be promoted into the final top-20 (the addressable
nDCG bucket per the 2026-06-20 ranking-bound diagnosis).

Design constraints (all enforced/tested here):
- the reorder is a PERMUTATION of the window — ids the LLM forgets are appended in
  their original order, so nothing is hallucinated, dropped, or duplicated;
- the tail (ranks window+1..) is never touched (it can't reach top-20 anyway);
- ANY failure (empty generation, unparseable order) falls back to the LGBM order —
  never an exception, never a shrunk list. Same fail-safe discipline as the responder.

The LLM call is INJECTED as `generate_fn(prompt) -> str | None`, so the ordering
logic is pure + offline-testable; make_gemini_generate_fn supplies the live wiring.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Callable, Optional

from mcrs.contracts import RankedList


LISTWISE_INSTRUCTIONS = """You re-rank candidate music tracks for a user mid-conversation.
Given the conversation (and the listener's goal, if any) and a numbered list of candidate tracks,
order the candidates from MOST to LEAST likely to be the track the user wants to hear NEXT.
Judge fit from the conversation: the mood, activity, genre, era, artists, and any track the user
named or liked. Rank ALL candidates; do not add, drop, or invent any.
Output ONLY the ranking as candidate numbers in brackets, best first, e.g. [4] > [1] > [7] > ...
No prose, no explanation."""


def build_listwise_prompt(conversation: str, goal: Optional[str],
                          candidate_descs: list[str]) -> str:
    """Assemble the listwise prompt: instructions, the conversation, the (optional)
    listener goal, then the candidates numbered [1]..[N]. The numbering is what
    parse_ranking reads back."""
    parts = [LISTWISE_INSTRUCTIONS, "", "=== CONVERSATION ===", conversation]
    if goal:
        parts += ["", f"Listener goal: {goal}"]
    parts += ["", "=== CANDIDATES ==="]
    parts += [f"[{i}] {d}" for i, d in enumerate(candidate_descs, start=1)]
    parts += ["", "Ranking:"]
    return "\n".join(parts)


def parse_ranking(text: str, n: int) -> Optional[list[int]]:
    """Read the model's ranking into a list of distinct 1-based positions in 1..n,
    in stated order. Prefers bracketed `[k]` tokens; falls back to bare integers.
    Out-of-range and duplicate positions are dropped (first occurrence wins).
    Returns None if nothing parseable -> caller falls back to the LGBM order."""
    if not text:
        return None
    nums = re.findall(r"\[(\d+)\]", text)
    if not nums:
        nums = re.findall(r"\d+", text)
    out: list[int] = []
    seen: set[int] = set()
    for s in nums:
        k = int(s)
        if 1 <= k <= n and k not in seen:
            seen.add(k)
            out.append(k)
    return out or None


def listwise_reorder(ids: list[str], order: Optional[list[int]], window: int) -> list[str]:
    """Reorder the top-`window` of `ids` by `order` (1-based positions into the
    window); append any window ids the order omitted, in original order; keep the
    tail (window..) unchanged. Falsy order -> `ids` unchanged (fallback)."""
    if not order:
        return ids
    win = ids[:window]
    tail = ids[window:]
    new_win: list[str] = []
    used: set[int] = set()
    for k in order:
        if 1 <= k <= len(win) and k not in used:
            used.add(k)
            new_win.append(win[k - 1])
    for i, t in enumerate(win, start=1):
        if i not in used:
            new_win.append(t)
    return new_win + tail


def rerank_ids(conversation: str, goal: Optional[str], ids: list[str],
               describe_fn: Callable[[str], str], generate_fn: Callable[[str], Optional[str]],
               window: int = 50, parse_fn: Callable[[str, int], Optional[list[int]]] = parse_ranking
               ) -> list[str]:
    """Build the prompt over the top-`window` of `ids`, call the (injected) LLM,
    parse its ranking, and return the reordered ids. Falls back to `ids` unchanged
    on an empty pool, empty generation, or unparseable ranking."""
    if not ids:
        return ids
    win = ids[:window]
    prompt = build_listwise_prompt(conversation, goal, [describe_fn(t) for t in win])
    raw = generate_fn(prompt)
    if not raw:
        return ids
    order = parse_fn(raw, len(win))
    if not order:
        return ids
    return listwise_reorder(ids, order, window)


def listwise_rerank(ranked: RankedList, describe_fn: Callable[[str], str],
                    generate_fn: Callable[[str], Optional[str]], window: int = 50) -> RankedList:
    """RankedList wrapper around rerank_ids: render the conversation from the turn's
    utterances, reorder the candidate ids, then re-emit the Candidates in the new
    order (preserving each candidate's features/scores). Pure given `generate_fn`."""
    turn = ranked.turn
    conversation = "\n".join(turn.utterances)
    ids = [c.track_id for c in ranked.items]
    new_ids = rerank_ids(conversation, turn.goal, ids, describe_fn, generate_fn, window)
    # bucket candidates by id so a reordering (and any rare duplicate id) maps back
    # to the right Candidate object without dropping one.
    buckets: dict[str, list] = {}
    for c in ranked.items:
        buckets.setdefault(c.track_id, []).append(c)
    new_items = [buckets[i].pop(0) for i in new_ids]
    return RankedList(turn, new_items)


def cache_generate_fn(generate_fn: Callable[[str], Optional[str]],
                      cache_dir: Optional[str]) -> Callable[[str], Optional[str]]:
    """Wrap `generate_fn` with a disk cache keyed by the prompt, so dev reruns don't
    re-pay the LLM. A failed (falsy) generation is NOT cached, so a fixed key/quota
    regenerates next run. cache_dir falsy -> passthrough (no caching)."""
    if not cache_dir:
        return generate_fn
    root = Path(cache_dir)

    def wrapped(prompt: str) -> Optional[str]:
        h = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:24]
        cp = root / f"{h}.json"
        if cp.exists():
            try:
                return json.loads(cp.read_text(encoding="utf-8"))["text"]
            except (OSError, json.JSONDecodeError, KeyError, ValueError):
                pass
        out = generate_fn(prompt)
        if out:
            root.mkdir(parents=True, exist_ok=True)
            cp.write_text(json.dumps({"text": out}, ensure_ascii=False), encoding="utf-8")
        return out

    return wrapped


def make_gemini_generate_fn(api_key: str, model: str = "gemini-2.5-flash",
                            thinking_budget: int = 512, max_output_tokens: int = 2048,
                            max_attempts: int = 4, sleep_fn: Callable[[float], None] = time.sleep
                            ) -> Callable[[str], Optional[str]]:
    """Live wiring (integration-only — needs the google-genai SDK + a key). Returns
    generate_fn(prompt) -> ranking text or None. Unlike the responder, thinking is
    LEFT ON (ordering is a reasoning task) but BUDGETED so it can't eat the whole
    output and trigger the empty-Part/finish_reason=STOP failure; max_output_tokens
    must exceed thinking_budget to leave room for the ranking. pro can't run at a 0
    budget, so a 0 is clamped to its 128 minimum. Returns None on a rate-limit
    exhaustion / API error / blocked-empty candidate so the caller falls back."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    tb = 128 if thinking_budget == 0 and "pro" in model.lower() else thinking_budget

    def generate(prompt: str) -> Optional[str]:
        for attempt in range(max_attempts):
            try:
                resp = client.models.generate_content(
                    model=model, contents=prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=max_output_tokens, temperature=0.0,
                        thinking_config=types.ThinkingConfig(thinking_budget=tb)))
                return resp.text or None
            except Exception as e:  # noqa: BLE001 — API raises a variety of types
                es = str(e).lower()
                rate_limited = ("429" in es or "resourceexhausted" in es
                                or "quota" in es or "exhausted" in es)
                if rate_limited and attempt < max_attempts - 1:
                    sleep_fn(min(60.0, 5.0 * (attempt + 1)))
                else:
                    print(f"  listwise API error: {e!r}")
                    return None
        return None

    return generate
