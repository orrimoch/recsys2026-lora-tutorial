"""EXP-015 — artist-hypothesis generative recall (free local LLM).

The new-artist WALL (~43% of turn-1 golds, ~99% new artists) is unreachable by
similarity — the data audit confirmed there is NO user listening-history to exploit,
so the only path is EXTERNAL music knowledge: have an LLM name candidate ARTISTS the
listener would enjoy, then ground them EXACTLY (artist_name -> catalog tracks). Artist
granularity is far more groundable than track-level propose-ground (no fuzzy title match).

Pure helpers here (prompt build, robust list parse, name normalization, gold-artist
hit, and a no-model "named in conversation" baseline) are unit-tested; the LLM call is
done in the notebook with a free local Qwen (no API). Defaults are model-agnostic.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Optional

_ENUM = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")        # leading bullet / "1." / "2)"
_NOTE = re.compile(r"\s+[-–—:]\s+.*$")                  # trailing " - dreamy guitars"


def _first(v):
    if isinstance(v, list):
        return v[0] if v else None
    return v


def build_artist_prompt(conversation: str, goal: str,
                        liked_artists: Optional[list[str]] = None,
                        culture: Optional[str] = None, n: int = 20) -> tuple[str, str]:
    """(system, user) asking the LLM to name `n` artists the listener would enjoy.
    Seeds with the goal (sharpest cue), any liked/named artists, and culture."""
    system = ("You are a music expert with deep knowledge of artists across genres, "
              "eras, and scenes — including lesser-known and non-mainstream artists. "
              "Given a listener's request, name real recording artists they would enjoy.")
    lines = []
    if culture:
        lines.append(f"Listener's preferred musical culture: {culture}")
    if liked_artists:
        lines.append("Artists the listener already likes: " + ", ".join(liked_artists))
    lines.append("Conversation / request:")
    lines.append(conversation.strip())
    if goal and goal.strip():
        lines.append(f"Goal: {goal.strip()}")
    lines.append("")
    lines.append(f"List the {n} real artists (one per line, name only — no songs, no "
                 f"commentary) this listener would MOST enjoy for this request. Favor "
                 f"artists that genuinely fit, including non-obvious / lesser-known ones.")
    return system, "\n".join(lines)


def parse_artist_list(text: str, max_n: Optional[int] = None) -> list[str]:
    """Parse a model's artist list -> clean names. Handles numbered / bulleted / comma
    forms, strips quotes + trailing notes, dedupes case-insensitively (keep first), caps."""
    if not text:
        return []
    raw = text.splitlines() if "\n" in text else text.split(",")
    out, seen = [], set()
    for tok in raw:
        name = _ENUM.sub("", tok.strip())
        name = _NOTE.sub("", name)
        name = name.strip().strip('"“”‘’\'')  # quotes
        name = name.strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(name)
        if max_n and len(out) >= max_n:
            break
    return out


def normalize_artist(name) -> str:
    """Canonical key: lowercased, accent-stripped, punctuation-removed, leading 'the '
    dropped, whitespace collapsed. So 'The Beatles' / 'Sigur Rós' / 'Beyoncé' match."""
    s = _first(name)
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))   # drop accents
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)                          # punctuation -> space
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"^the\s+", "", s)                               # leading 'the '
    return s.strip()


def artist_hit(gold_artist, generated: list[str]) -> bool:
    """True iff the gold's artist (normalized) is among the generated artists."""
    g = normalize_artist(gold_artist)
    if not g:
        return False
    return g in {normalize_artist(a) for a in (generated or [])}


def artist_named_in_text(gold_artist, conversation_text: str) -> bool:
    """No-model baseline: is the gold's artist already named in the conversation?
    (If yes, BM25/dense usually already retrieves it — so the TRUE wall is the
    not-named subset, which only external knowledge can reach.)"""
    g = normalize_artist(gold_artist)
    if not g:
        return False
    return g in normalize_artist(conversation_text)
