"""S1 — grounded Gemini responder.

Generates `SubmissionRow.predicted_response`: a short, grounded, personalized reply that the
official Gemini judge scores on Personalization + Explanation Quality (decoupled from nDCG — it
only reads the final top tracks + conversation context). See `.claude/documents/features/70_S1_responder.md`.

The prompts and the context/track rendering reuse the proven prior responder (single-shot / plain
path only — best-of-N, the structured-personality mode, and the self-judge are intentionally
dropped). What's new: the modern `google-genai` async client (injected, so this module imports no
SDK and stays unit-testable), prompt-injection `sanitize`-ing of untrusted track/utterance text, a
non-empty grounded `fallback_response`, and an F2 `Responder`-conformant class.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from mcrs.contracts import TurnContext

# ── prompts (proven responder prompts, single-shot path) ─────────────────────
RESPONDER_INSTRUCTIONS = """You are an expert music recommender replying to a user mid-conversation.
Write ONE reply, 2-3 short sentences, no lists, no preamble.

Voice: be helpful, warm, and genuinely kind, but to the point — like a friend with great taste, not a
salesperson. Read the user's vibe — their mood, energy, and how they seem to be feeling right now — and
match it: if they're excited, share it; if they're winding down, stay easy and calm.

First, directly answer the user's MOST RECENT message — the reply must read as an on-point response to
what they just asked for, not a generic pitch.

From the candidate track(s) below, recommend the SINGLE best-fitting one and build the reply around it.
Mention a second track ONLY if it genuinely fits the same request; never pad the reply with extra tracks
just because they were provided, and never connect a track on a shallow coincidence (e.g. a word in the
title) — the link must be a real musical or taste fit.

Continuity: if earlier in this conversation the user liked a track, briefly connect your pick to it
through a real shared attribute (e.g. "like [earlier track]'s warm guitar, this one…"), and steer away
from qualities they rejected. Only do this when the connection is genuine — never invent a past
preference, and skip it entirely if there is no relevant history.

Cold start: if there is no prior history to build on, personalize from the user's current request and
stated goal — their mood, activity, and the qualities they just named — and ground the pick in the
track's real attributes. Do not fabricate past taste, and never fall back to a generic "here are some
songs you might like" line.

To score well you MUST do BOTH:
- PERSONALIZATION: tie the pick to something THIS user actually said AND to their current vibe (intent,
  mood, activity, taste). Reference it concretely; never be generic, and never present guesses about the
  user as facts.
- EXPLANATION: justify the pick with at least one real attribute of the recommended track (artist,
  title, genre, mood, instrumentation, era) drawn ONLY from what you were given, and say WHY it fits.

Rules: lead with the recommendation; never open with a generic line; keep it tight — every sentence earns
its place; never invent or guess attributes you were not given; if unsure of something, leave it out so
nothing sounds made up. Return ONLY the reply text — no labels, headers, quotation marks, or description
of your reasoning."""

# Few-shot STYLE references (real tracks + real high-GPA train turns, hand-verified). Labeled
# "do not reuse" so the model copies STYLE, not these example tracks.
FEW_SHOT_EXAMPLES = """=== EXAMPLES (style reference only — do not reuse these tracks) ===
[user]: I love 'With A Little Help From My Friends' — that collaborative, Sgt. Pepper's-era feel. Could you suggest another Beatles track from around 1967, something that showcases their more psychedelic, artistic side?
[tracks]: Strawberry Fields Forever - Remastered by The Beatles
[reply]: For that 1967 psychedelic side, go straight to "Strawberry Fields Forever" — it's the Beatles at their most experimental and dreamlike, all layered, surreal textures, but still rooted in the melodic pop-rock you love from that era.

[user]: This is exactly what I wanted — a beautiful, piano-focused jazz track. Got any more like it?
[tracks]: Lucky To Be Me by Bill Evans
[reply]: Then you'll love "Lucky To Be Me" by Bill Evans — it's classic piano-led jazz, sparse and introspective, the kind of delicate, melodic playing that rewards close listening.

[user]: "Dog Days Are Over" is exactly the vivid, dramatic storytelling I was hoping for — that emotional intensity is perfect. More like this?
[tracks]: Cosmic Love by Florence + The Machine
[reply]: Stay with Florence + The Machine and try "Cosmic Love" — just as epic and dramatic, built on Florence's soaring vocals and a surging, cathartic swell that matches the emotional intensity you're after."""


# ── injection safety (spec §8) ────────────────────────────────────────────────
def sanitize(text: Any) -> str:
    """Neutralize prompt-injection in untrusted track names / utterances before templating:
    collapse newlines/tabs to spaces (so injected text can't add prompt lines) and strip `===`
    section markers (so it can't fake a header). Content itself is preserved, not censored."""
    if text is None:
        return ""
    s = str(text).replace("\r", " ").replace("\n", " ").replace("\t", " ")
    s = re.sub(r"=+", " ", s)            # kill === SYSTEM === style markers
    return re.sub(r"\s+", " ", s).strip()


# ── context / track rendering (with sanitize added) ──────────────────────────
def _first(v: Any) -> Any:
    """HF metadata fields are often singleton lists; unwrap to a scalar."""
    if isinstance(v, list):
        return v[0] if v else None
    return v


def _track_inline(meta: dict) -> str:
    """One-line 'Name by Artist — tag1, tag2' for a prior in-session rec, so the model can connect a
    new pick to what the user already liked via shared traits."""
    name = sanitize(_first(meta.get("track_name")))
    artist = sanitize(_first(meta.get("artist_name")))
    tags = meta.get("tags") or meta.get("tag_list") or []
    tags = tags if isinstance(tags, list) else [str(tags)]
    s = f"{name} by {artist}"
    if tags:
        s += f" — {', '.join(sanitize(x) for x in tags[:4])}"
    return s


def render_context(conversations: list[dict], item_meta: dict, target_turn: int) -> str:
    """Render the user-visible conversation up to (and including) the target user turn. Music turns
    are expanded to the recommended track's name + attributes (continuity); the target turn's OWN
    assistant/music reply is excluded — that is what we are generating."""
    lines: list[str] = []
    for t in conversations:
        if t["turn_number"] >= target_turn:
            continue
        role, content = t["role"], t["content"]
        if role == "music":
            role = "assistant"
            content = f"[recommended: {_track_inline(item_meta.get(str(content), {}))}]"
        else:
            content = sanitize(content)
        lines.append(f"{role}: {content}")
    for t in conversations:
        if t["turn_number"] == target_turn and t["role"] == "user":
            lines.append(f"user: {sanitize(t['content'])}")
            break
    return "\n".join(lines)


def _render_tracks(metas: list[Optional[dict]], n: int = 1) -> str:
    """Format up to n track metas as grounding for the reply. Skips entries with no metadata
    (never emits a bare 'None by None')."""
    out: list[str] = []
    for m in (metas or [])[:n]:
        if not m:
            continue
        name, artist = _first(m.get("track_name")), _first(m.get("artist_name"))
        if name is None and artist is None:
            continue
        part = f"{sanitize(name)} by {sanitize(artist)}"
        album = _first(m.get("album_name"))
        if album:
            part += f" (album {sanitize(album)})"
        tags = m.get("tags") or m.get("tag_list") or []
        tags = tags if isinstance(tags, list) else [str(tags)]
        if tags:
            part += f" [{', '.join(sanitize(x) for x in tags[:5])}]"
        out.append(part)
    return "; ".join(out) if out else "(none)"


def format_tracks(tids: Optional[list], item_meta: dict, n: int = 1) -> str:
    """Resolve the top-n predicted track ids to metadata and render them for the prompt."""
    metas = [item_meta.get(str(tid)) for tid in (tids or [])[:n]]
    return _render_tracks(metas, n)


def build_prompt(context: str, tracks_str: str, listener_goal: Optional[str]) -> str:
    """Assemble the responder prompt: rubric instructions + few-shot style + conversation + the
    recommended tracks + (optionally) the listener goal."""
    parts = [RESPONDER_INSTRUCTIONS, "", FEW_SHOT_EXAMPLES, "", "=== CONVERSATION ===", context]
    if listener_goal:
        parts += ["", f"Listener goal: {listener_goal}"]
    parts += ["", "=== RECOMMENDED TRACK(S) TO PRESENT ===", tracks_str, "", "Reply:"]
    return "\n".join(parts)


def fallback_response(tracks_str: str) -> str:
    """Non-empty, grounded fallback when the model returns empty/blocked or a row errors (spec §8:
    never ship an empty response, and never the 'ok' stub)."""
    if tracks_str and tracks_str != "(none)":
        return f"Here's a great pick for you: {tracks_str.split(';')[0].strip()}."
    return "Here are a few recommendations I think you'll enjoy."


def build_output_row(pred: dict, response: str) -> dict:
    """Submission row: preserve identity + predicted_track_ids, replace only predicted_response."""
    return {
        "session_id": pred["session_id"], "user_id": pred["user_id"],
        "turn_number": pred["turn_number"],
        "predicted_track_ids": pred["predicted_track_ids"],
        "predicted_response": response,
    }


# ── config + Gemini calls ─────────────────────────────────────────────────────
@dataclass
class ResponderConfig:
    model_revision: str = "gemini-2.5-flash"
    top_n_for_prompt: int = 3
    # gemini-2.5 models spend output tokens on internal THINKING before the reply, and that counts
    # against max_output_tokens. A small cap (e.g. 256) is consumed entirely by thinking -> empty
    # reply -> every row falls back. Leave generous room for thinking + the short reply.
    max_tokens: int = 2048
    temperature: float = 0.7
    concurrency: int = 16
    cache_dir: Optional[str] = None


def _gen_config(cfg: ResponderConfig) -> dict:
    # google-genai coerces a dict into a GenerateContentConfig, so the module needs no SDK import.
    return {"temperature": cfg.temperature, "max_output_tokens": cfg.max_tokens}


def _safe_text(resp: Any) -> str:
    """`.text` raises on a blocked/empty candidate — treat that as no output."""
    try:
        return (resp.text or "").strip()
    except Exception:
        return ""


def _cache_path(cache_dir: str, prompt: str, model: str) -> Path:
    h = hashlib.sha1("\n".join([prompt, str(model)]).encode("utf-8")).hexdigest()[:24]
    return Path(cache_dir) / f"{h}.json"


def _read_cache(cache_dir: Optional[str], prompt: str, model: str) -> Optional[str]:
    if not cache_dir:
        return None
    p = _cache_path(cache_dir, prompt, model)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))["response"]
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            return None
    return None


def _write_cache(cache_dir: Optional[str], prompt: str, model: str, resp: str) -> None:
    # A fallback is never cached (so a fixed key/quota regenerates next run).
    if not cache_dir or not resp:
        return
    p = _cache_path(cache_dir, prompt, model)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"response": resp}, ensure_ascii=False), encoding="utf-8")


async def generate_responses(
    preds: list[dict], sessions_by_id: dict, item_meta: dict,
    cfg: ResponderConfig, client: Any, fallback_text: Optional[str] = None,
) -> list[dict]:
    """Batch driver: for each prediction row, rejoin its conversation context + the recommended
    tracks' metadata, generate a grounded reply via the async `google-genai` client (single-shot,
    capped at `cfg.concurrency` in-flight), and swap it into the row. `predicted_track_ids` are left
    untouched. Row count is always conserved — a per-row failure falls back to a non-empty response,
    never aborts the run. Disk-cached per (prompt, model) when `cfg.cache_dir` is set."""
    sem = asyncio.Semaphore(max(1, cfg.concurrency))

    async def _one(pred: dict) -> dict:
        try:
            sess = sessions_by_id.get(pred["session_id"])
            tracks = format_tracks(pred.get("predicted_track_ids"), item_meta, n=cfg.top_n_for_prompt)
            if sess is None:
                return build_output_row(pred, fallback_text or fallback_response(tracks))
            ctx = render_context(sess["conversations"], item_meta, pred["turn_number"])
            goal = ((sess.get("conversation_goal") or {}).get("listener_goal") or "").strip()
            prompt = build_prompt(ctx, tracks, goal)
            cached = _read_cache(cfg.cache_dir, prompt, cfg.model_revision)
            if cached is not None:
                return build_output_row(pred, cached)
            async with sem:
                resp = await client.aio.models.generate_content(
                    model=cfg.model_revision, contents=prompt, config=_gen_config(cfg))
            text = _safe_text(resp)
            if not text:
                return build_output_row(pred, fallback_text or fallback_response(tracks))
            _write_cache(cfg.cache_dir, prompt, cfg.model_revision, text)
            return build_output_row(pred, text)
        except Exception:
            try:
                tracks = format_tracks(pred.get("predicted_track_ids"), item_meta,
                                       n=cfg.top_n_for_prompt)
            except Exception:
                tracks = "(none)"
            return build_output_row(pred, fallback_text or fallback_response(tracks))

    return list(await asyncio.gather(*[_one(p) for p in preds]))


class GeminiResponder:
    """F2 `Responder` Protocol implementation (single-shot). Used for spec conformance + a
    per-turn API; the production notebook path is `generate_responses` (batch/async)."""

    def __init__(self, cfg: ResponderConfig, client: Any):
        self.cfg, self.client = cfg, client

    def respond(self, ctx: TurnContext, top_tracks: list[dict]) -> str:
        context = "\n".join(f"user: {sanitize(u)}" for u in ctx.utterances)
        tracks = _render_tracks(top_tracks, n=self.cfg.top_n_for_prompt)
        prompt = build_prompt(context, tracks, ctx.goal or "")
        try:
            resp = self.client.models.generate_content(
                model=self.cfg.model_revision, contents=prompt, config=_gen_config(self.cfg))
            text = _safe_text(resp)
        except Exception:
            text = ""
        return text or fallback_response(tracks)
