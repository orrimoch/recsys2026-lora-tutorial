"""EXP-016 — doc2query catalog enrichment (document expansion), one-time + resumable.

The wall is a rich-query / sparse-doc mismatch: users describe tracks by attributes
("energetic late-90s electronic dance party") but catalog docs are thin (name/artist/
album/few tags), so similarity buries the canonical answer (e.g. Daft Punk, pop 82).
Fix = document expansion (doc2query): for each track, an LLM writes a rich description
+ example listener-requests the track answers; we APPEND that to the original metadata
(expansion, not replacement — exact name/artist match preserved) and re-index BM25 +
a bi-encoder (e5). NOT the bge CROSS-encoder — that's a reranker, it can't pre-index docs.

Output parquet: track_id, enriched_text (LLM expansion), doc_text (metadata + expansion).
RESUMABLE: re-running skips track_ids already in --out (checkpointed every CHECKPOINT),
so an interrupt never re-pays and the file is reused forever (no re-run needed).

Pure helpers (meta_text / build_enrich_prompt / clean_enrichment / enriched_document /
select_pending) are unit-tested; the Gemini call + 47k loop + parquet I/O live in main().
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

CHECKPOINT = 2000   # write the parquet every N newly-enriched tracks (crash-safe resume)


def _first(v):
    if isinstance(v, list):
        return v[0] if v else None
    return v


def _year(release_date) -> str:
    m = re.search(r"\d{4}", str(_first(release_date) or ""))
    return m.group(0) if m else ""


def meta_text(meta: dict) -> str:
    """Original catalog text (kept verbatim so exact name/artist/album match survives)."""
    artist = _first(meta.get("artist_name")) or ""
    title = _first(meta.get("track_name")) or ""
    album = _first(meta.get("album_name")) or ""
    core = " - ".join(p for p in (artist, title, album) if p)
    tags = meta.get("tag_list") or []
    if tags:
        core += " | tags: " + ", ".join(str(t) for t in tags[:12])
    y = _year(meta.get("release_date"))
    if y:
        core += f" | {y}"
    return core.strip()


_SYSTEM = (
    "You are a music metadata expert. Given a track's metadata you write retrieval-oriented "
    "text: a rich description plus example listener requests the track would satisfy. Ground "
    "STRICTLY in the provided metadata, genre tags, and only widely-known facts. If you do not "
    "recognize the specific track, infer the likely style FROM THE TAGS and DO NOT invent "
    "specific facts (no fake dates, collaborators, labels, or chart positions). Plain text only."
)


def build_enrich_prompt(meta: dict, n_requests: int = 4) -> tuple[str, str]:
    """(system, user). Asks for a description + `n_requests` example listener requests —
    the doc2query signal that injects real user vocabulary into the indexable doc."""
    artist = _first(meta.get("artist_name")) or "Unknown artist"
    title = _first(meta.get("track_name")) or "Unknown title"
    album = _first(meta.get("album_name")) or ""
    tags = ", ".join(str(t) for t in (meta.get("tag_list") or [])[:12]) or "(none)"
    y = _year(meta.get("release_date"))
    pop = meta.get("popularity")
    user = (
        f"Track: {artist} - {title}"
        + (f" (album: {album})" if album else "")
        + (f", {y}" if y else "")
        + f". Tags: {tags}." + (f" Popularity: {pop}." if pop is not None else "")
        + "\n\nBe concise. Plain text, no preamble. Write:\n"
        "1) 2 sentences: genre, era, mood, energy/instrumentation, and 2 similar artists.\n"
        f"2) exactly {n_requests} short example listener requests (max ~8 words each), "
        "varied by mood, activity, genre, and era — phrased like a person talking to a music app."
    )
    return _SYSTEM, user


_MD = re.compile(r"[*#`>_]+")
_LABEL = re.compile(r"^\s*(?:\d+[.)]|[-•])\s*", re.M)


def clean_enrichment(raw: str, max_chars: int = 1500) -> str:
    """Flatten the model output into one indexable line: drop markdown / bullets / list
    numbering, collapse whitespace, cap length (keep the embedding within seq limits)."""
    if not raw:
        return ""
    s = _MD.sub(" ", raw)
    s = _LABEL.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:max_chars].strip()


def enriched_document(meta: dict, enrichment: str) -> str:
    """APPEND the enrichment to the original metadata (expansion, not replacement)."""
    base = meta_text(meta)
    enr = clean_enrichment(enrichment)
    return f"{base} | {enr}" if enr else base


def select_pending(rows, done_ids) -> list:
    """Tracks not yet enriched (resume support)."""
    return [r for r in rows if r.get("track_id") not in done_ids]


def build_records(done: dict, by_id: dict, prev_doc: dict | None = None) -> list:
    """All enriched rows to persist. Recompute doc_text from metadata for tracks in this
    run (by_id); carry forward the saved doc_text for prior-only tracks. This makes a
    limited/SMOKE re-run over an existing full parquet NON-destructive (it preserves every
    prior row instead of truncating to the current subset)."""
    prev_doc = prev_doc or {}
    recs = []
    for tid, enr in done.items():
        r = by_id.get(tid)
        doc = enriched_document(r, enr) if r is not None else prev_doc.get(tid, "")
        recs.append({"track_id": tid, "enriched_text": enr, "doc_text": doc})
    return recs


# ---- runner (IO; lazy imports so the pure helpers import without heavy deps) ----
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="doc2query catalog enrichment (resumable).")
    ap.add_argument("--out", required=True, help="enriched catalog parquet (resumable checkpoint)")
    ap.add_argument("--item-db", default="talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
    ap.add_argument("--split", default="all_tracks")
    ap.add_argument("--model", default="gemini-2.5-flash-lite")
    ap.add_argument("--n-requests", type=int, default=4)
    ap.add_argument("--max-output-tokens", type=int, default=320)  # thinking OFF -> ~180 output fits
    ap.add_argument("--batch-size", type=int, default=24, help="concurrent API calls")
    ap.add_argument("--limit", type=int, default=0, help="cap rows (0=all; for a smoke run)")
    args = ap.parse_args(argv)

    from concurrent.futures import ThreadPoolExecutor
    import pandas as pd
    from datasets import load_dataset
    try:
        from mcrs.query_rewriters.gemini_propose import GeminiClient
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "music-crs-baselines"))
        from mcrs.query_rewriters.gemini_propose import GeminiClient

    rows = list(load_dataset(args.item_db, split=args.split))
    if args.limit:
        rows = rows[: args.limit]

    done, prev_doc = {}, {}
    if os.path.exists(args.out):
        prev = pd.read_parquet(args.out)
        done = dict(zip(prev["track_id"], prev["enriched_text"]))
        prev_doc = dict(zip(prev["track_id"], prev["doc_text"]))   # carry-forward for limited re-runs
        print(f"[enrich] resume: {len(done)} already enriched in {args.out}")
    pending = select_pending(rows, set(done))
    print(f"[enrich] {len(rows)} tracks total | {len(pending)} pending | model={args.model}")

    # thinking_budget=0: enrichment is descriptive, not reasoning -> no thinking tokens
    # (cheaper, faster, and avoids thinking eating the output budget -> no truncation).
    client = GeminiClient(model=args.model, max_output_tokens=args.max_output_tokens,
                          thinking_budget=0)
    by_id = {r["track_id"]: r for r in rows}

    def _one(r):
        system, user = build_enrich_prompt(r, n_requests=args.n_requests)
        for _ in range(3):
            try:
                txt = client.generate(system, user)
                if txt and txt.strip():
                    return r["track_id"], clean_enrichment(txt)
            except Exception:
                continue
        return r["track_id"], ""        # failed -> empty, NOT cached (retried next run)

    def _flush():
        # code-review N1: emit EVERY enriched track (this run + prior) so a limited/SMOKE
        # re-run over an existing full parquet never truncates it.
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        pd.DataFrame(build_records(done, by_id, prev_doc)).to_parquet(args.out, index=False)

    n_new = 0
    with ThreadPoolExecutor(max_workers=max(1, args.batch_size)) as ex:
        for tid, enr in ex.map(_one, pending):
            if enr:                      # only persist successful, non-empty enrichments
                done[tid] = enr
                n_new += 1
                if n_new % CHECKPOINT == 0:
                    _flush()
                    print(f"[enrich] {n_new}/{len(pending)} enriched (checkpoint saved)")
    _flush()
    ok = sum(1 for v in done.values() if v)
    print(f"[enrich] DONE -> {args.out} | enriched {ok}/{len(rows)} (new this run: {n_new})")
    if ok < len(rows):
        print(f"[enrich] {len(rows)-ok} still un-enriched (API failures) -> re-run to fill them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
