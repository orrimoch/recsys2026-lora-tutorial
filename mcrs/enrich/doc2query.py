"""A1 — catalog enrichment / doc2query.

Generates the queries/blurb a track answers and appends them to its doc, closing the
conversational↔metadata vocabulary gap (the top recall lever per P0). Pure helpers + a driver
that takes an injected `generate_fn(system, user) -> str` (Gemini at serve; a fake in tests).
Output: {canonical track_id -> enriched_doc} consumed by F1 `Catalog.id_to_metadata(enriched=True)`.
Reads track metadata ONLY (no conversations/golds) — no leak.
"""
from __future__ import annotations

import re
from typing import Callable, Iterable, Optional

from mcrs.data.ids import canonical_track_id


def _join(v) -> str:
    if isinstance(v, list):
        return ", ".join(str(x) for x in v)
    return "" if v is None else str(v)


def _year(release_date) -> str:
    m = re.search(r"\d{4}", str(release_date or ""))
    return m.group() if m else ""


def meta_text(meta: dict) -> str:
    fields = [
        ("name", _join(meta.get("track_name"))),
        ("artist", _join(meta.get("artist_name"))),
        ("album", _join(meta.get("album_name"))),
        ("tags", _join(meta.get("tag_list"))),
        ("year", _year(meta.get("release_date"))),
    ]
    return ", ".join(f"{k}: {v}" for k, v in fields if v)


def build_enrich_prompt(meta: dict, n_requests: int = 4,
                        examples: Optional[list[str]] = None) -> tuple[str, str]:
    system = (
        "You are a music metadata expert. Given a track's metadata, write "
        f"{n_requests} varied retrieval queries a listener might actually use to find this track "
        "(mood, genre, era, similar artists, use-cases/activities). Vary length from terse keywords "
        "to a full conversational request. One per line, no numbering, no extra commentary."
    )
    if examples:
        # Few-shot STYLE anchors only (real listener phrasings, sampled from TRAIN — not tied to
        # this track). They calibrate register/length to the real query distribution.
        shots = "\n".join(f"- {e}" for e in examples)
        system += (
            "\n\nReal listener queries look like this (match their style and variety, "
            f"do NOT copy their content):\n{shots}"
        )
    return system, meta_text(meta)


def clean_enrichment(raw: str, max_chars: int = 1500) -> str:
    """Flatten the model output into one indexable line: drop markdown/bullets/numbering/newlines."""
    lines = []
    for line in (raw or "").splitlines():
        # strip only leading list markers (bullets or "N."/"N)") — NOT content digits like "90s"
        line = re.sub(r"^\s*(?:[-*•]+|\d+[.)])\s*", "", line).strip()
        if line:
            lines.append(line)
    return " ".join(lines)[:max_chars]


def enriched_document(meta: dict, enrichment: str) -> str:
    base = meta_text(meta)
    cleaned = clean_enrichment(enrichment)
    return f"{base} | {cleaned}" if cleaned else base


def enrich_catalog(
    rows: Iterable[dict],
    generate_fn: Callable[[str, str], str],
    n_requests: int = 4,
    limit: int = 0,
    done: Optional[dict[str, str]] = None,
) -> dict[str, str]:
    """{canonical track_id -> enriched_doc}. `done` lets the caller resume (skip already-enriched)."""
    out: dict[str, str] = dict(done or {})
    for i, r in enumerate(rows):
        if limit and i >= limit:
            break
        tid = canonical_track_id(r["track_id"])
        if tid in out:
            continue
        system, user = build_enrich_prompt(r, n_requests)
        out[tid] = enriched_document(r, generate_fn(system, user))
    return out
