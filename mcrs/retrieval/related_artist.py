"""R6 — related-artist channel (cross-session artist co-occurrence).

Reaches NEW artists — the new-artist recall wall that same_artist (session artists only) and the
content/semantic channels structurally miss. Mechanism: from the TRAIN split, count how often two
artists share a session (`build_artist_cooc`, one count per shared session, replays de-duped). At
serve: from the session's seen artists, expand to their top co-occurring NEW artists and emit those
artists' most-popular tracks (minus played). F2 RetrievalChannel; canonical ids; [] on no history.
Train-only co-occurrence (no dev/blind) → no leak. Port of salvage `related_artist.py` onto F1/F2.
See `.claude/documents/features/45_R6_extension_channels.md` §4.3.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable, Optional

from mcrs.data.ids import canonical_track_id


def _norm_artist(a) -> str:
    return str(a or "").strip().lower()


def _artists_of(meta: dict) -> list[str]:
    a = meta.get("artist_name") or []
    if isinstance(a, str):
        a = [a]
    out, seen = [], set()
    for x in a:                       # de-dup, order-preserving
        na = _norm_artist(x)
        if na and na not in seen:
            seen.add(na)
            out.append(na)
    return out


def build_artist_cooc(
    sessions: Iterable[dict], tid_to_artists: dict[str, list[str]]
) -> dict[str, dict[str, int]]:
    """{artist -> {co-occurring artist -> #shared sessions}} over TRAIN sessions.

    A session's artist set is de-duplicated first (replays don't inflate). Count every ordered
    pair of distinct artists once per session. Build from train only — never dev/blind."""
    cooc: dict[str, Counter] = defaultdict(Counter)
    for sess in sessions:
        arts: list[str] = []
        for turn in sess.get("conversations", []):
            if turn.get("role") == "music":
                arts.extend(tid_to_artists.get(canonical_track_id(str(turn.get("content"))), []))
        uniq = list(dict.fromkeys(arts))     # de-dup per session, order-preserving
        for a in uniq:
            for b in uniq:
                if a != b:
                    cooc[a][b] += 1
    return {a: dict(c) for a, c in cooc.items()}


def tid_to_artists_from_catalog(catalog) -> dict[str, list[str]]:
    """{canonical track_id -> [normalized artists]} for build_artist_cooc (same norm as the channel)."""
    return {tid: _artists_of(catalog.metadata(tid)) for tid in catalog.index_to_id}


class RelatedArtistChannel:
    label = "related_artist"

    def __init__(self, catalog, cooc: dict[str, dict[str, int]]) -> None:
        self.cooc = cooc
        self.tid_to_artists: dict[str, list[str]] = {}
        artist_tracks: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for tid in catalog.index_to_id:
            meta = catalog.metadata(tid)
            arts = _artists_of(meta)
            if not arts:
                continue
            self.tid_to_artists[tid] = arts
            pop = float(meta.get("popularity") or 0.0)
            for a in arts:
                artist_tracks[a].append((tid, pop))
        # popularity-sorted track ids per artist (stable on ties via catalog order)
        self.artist_to_tids: dict[str, list[str]] = {
            a: [t for t, _ in sorted(v, key=lambda x: -x[1])] for a, v in artist_tracks.items()
        }

    def batch_text_to_item_retrieval(
        self, queries: list[str], topk: int,
        batch_context: Optional[list[dict]] = None, user_ids: Optional[list[str]] = None,
    ) -> list[list[str]]:
        out: list[list[str]] = []
        for i in range(len(queries)):
            ctx = batch_context[i] if batch_context and i < len(batch_context) else None
            played = [canonical_track_id(t) for t in ((ctx or {}).get("history_tids") or [])]
            if not played:
                out.append([])
                continue
            played_set = set(played)
            seen_artists = {a for t in played for a in self.tid_to_artists.get(t, [])}
            # expand seen artists -> co-occurring NEW artists, summed by shared-session count
            expanded: Counter = Counter()
            for a in seen_artists:
                for b, c in self.cooc.get(a, {}).items():
                    if b not in seen_artists:
                        expanded[b] += c
            ranked: list[str] = []
            emitted: set[str] = set()
            for artist, _cnt in expanded.most_common():
                for tid in self.artist_to_tids.get(artist, []):
                    if tid in played_set or tid in emitted:
                        continue
                    emitted.add(tid)
                    ranked.append(canonical_track_id(tid))
                    if len(ranked) >= topk:
                        break
                if len(ranked) >= topk:
                    break
            out.append(ranked[:topk])
        return out

    def text_to_item_retrieval(self, query: str, topk: int, user_id=None) -> list[str]:
        return self.batch_text_to_item_retrieval([query], topk)[0]
