"""Track A — LLM listwise reranker (plan: drastically improve nDCG@20).

Mirrors the TalkPlayData generator's own *Recsys LLM*: given the listener's
profile + goal + dialog (an indirect, lossy description of the target track) and
a numbered candidate pool, an LLM reasons over real-world music knowledge and
returns a ranked ordering. This attacks the CONVERSION gap (golds recall surfaces
at ranks 21..100 but LGBM/cross-encoder/SASRec never lift into top-20) — the one
ranker class never actually tried. It can only REORDER the pool, never recall a
gold the pool missed (recall@100 0.496 stays the hard ceiling).

The pure helpers (`render_candidate`, `build_listwise_prompt`, `parse_ranking`,
`merge_order`) are unit-tested with a fake client; the Gemini call is integration
(Colab). Determinism: temperature 0 + deterministic parse + sha1 cache by
(query, pool-head, model), so re-runs of the dev gate are free on cache hits.
"""
from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

_DEFAULT_PROMPT = Path(__file__).resolve().parents[1] / "system_prompts" / "llm_listwise_rerank.txt"
_INT = re.compile(r"\d+")


def _first(v):
    if isinstance(v, list):
        return v[0] if v else None
    return v


def render_candidate(idx: int, tid: str, meta_lookup: dict) -> str:
    """Compact, index-keyed candidate line: "[idx] artist - title - album [- tags]".
    The UUID is dropped (wasted context the LLM can't use); unknown tids fall back
    to the bare tid so the line is never empty."""
    md = meta_lookup.get(tid)
    if not md:
        return f"[{idx}] {tid}"
    artist = _first(md.get("artist_name")) or ""
    title = _first(md.get("track_name")) or ""
    album = _first(md.get("album_name")) or ""
    core = " - ".join(p for p in (artist, title, album) if p) or tid
    tags = md.get("tag_list") or []
    if tags:
        core += " - " + ", ".join(str(t) for t in tags[:5])
    return f"[{idx}] {core}"


def _profile_block(profile: Optional[dict]) -> str:
    """UserID->DB taste stream: culture (471 fine-grained values) + demographics.
    Empty when no profile (cold user) so the prompt degrades cleanly."""
    if not isinstance(profile, dict) or not profile:
        return ""
    bits = []
    culture = str(profile.get("preferred_musical_culture") or "").strip()
    if culture:
        bits.append(f"preferred culture: {culture}")
    age = profile.get("age_group") or profile.get("age")
    country = profile.get("country_name") or profile.get("country_code")
    gender = profile.get("gender")
    if age or country or gender:
        bits.append(f"age={age or 'unknown'} country={country or 'unknown'} "
                    f"gender={gender or 'unknown'}")
    return ("listener profile: " + "; ".join(bits)) if bits else ""


def build_listwise_prompt(query: str, tids: list[str], meta_lookup: dict, k: int,
                          system_prompt: str, profile: Optional[dict] = None,
                          goal_category: Optional[str] = None,
                          goal_specificity: Optional[str] = None) -> tuple[str, str]:
    """Return (system, user). The user content carries all three TalkPlay streams:
    UserID->DB (profile/taste), Text Query + Chat History (`query`, already
    `raw_with_goal` so it ends in the listener_goal line), then the numbered
    candidate block (first `k` only). `goal_specificity` HH ("one specific song")
    nudges sharp top-1 commitment; LL spreads."""
    lines: list[str] = []
    pb = _profile_block(profile)
    if pb:
        lines.append(pb)
    # specificity calibration (HH/HL/LH/LL: first char ~ how specific the target is)
    if goal_specificity and str(goal_specificity).upper().startswith("H"):
        lines.append("the listener wants ONE specific track — commit decisively to your single best match at rank 1.")
    lines.append("Conversation (the listener's request; the 'goal:' line is the sharpest description of the target):")
    lines.append(query.strip())
    lines.append("")
    lines.append("Candidates:")
    for j, tid in enumerate(tids[:k], start=1):
        lines.append(render_candidate(j, tid, meta_lookup))
    lines.append("")
    lines.append("Output the candidate indices in ranked order, most relevant first, comma-separated, each exactly once:")
    return system_prompt, "\n".join(lines)


def parse_ranking(text: str, n: int) -> list[int]:
    """Parse the model's ranked index list -> 0-indexed positions in [0, n).
    Accepts a comma list, a JSON array, or prose; 1-indexed input. Dedupes (keep
    first), drops out-of-range / hallucinated indices. Returns a partial
    permutation (caller fills the rest via merge_order)."""
    if not text:
        return []
    nums: list[int] = []
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            arr = json.loads(m.group(0))
            if isinstance(arr, list):
                nums = [int(x) for x in arr if str(x).strip().lstrip("-").isdigit()]
        except (json.JSONDecodeError, ValueError):
            nums = []
    if not nums:
        nums = [int(t) for t in _INT.findall(text)]
    seen, out = set(), []
    for v in nums:
        z = v - 1  # 1-indexed -> 0-indexed
        if 0 <= z < n and z not in seen:
            seen.add(z)
            out.append(z)
    return out


def merge_order(parsed: list[int], n: int) -> list[int]:
    """Complete a partial ranking to a full permutation of [0, n): the parsed
    indices first, then any missing indices in original pool order (the graceful
    fallback for missing/dup/hallucinated indices, incl. an empty parse)."""
    seen = set(parsed)
    return list(parsed) + [i for i in range(n) if i not in seen]


class LLMListwiseReranker:
    """Reranker with the LGBM_RERANKER.rerank(...) contract. Construct with either
    `item_db_name`+splits (loads MusicCatalogDB.metadata_dict) or an explicit
    `meta_lookup` (tests). `client` is any object exposing
    `.generate(system, user) -> str`; default is a GeminiClient."""

    def __init__(self, item_db_name: Optional[str] = None,
                 track_split_types: Optional[list[str]] = None,
                 corpus_types: Optional[list[str]] = None,
                 cache_dir: str = "./cache", model_path: Optional[str] = None,
                 client=None, meta_lookup: Optional[dict] = None, k: int = 50,
                 max_output_tokens: int = 512, max_retries: int = 3,
                 batch_size: int = 16, system_prompt: Optional[str] = None,
                 system_prompt_path: Optional[str] = None) -> None:
        if meta_lookup is not None:
            self.meta_lookup = meta_lookup
        else:
            from ..db_item import MusicCatalogDB
            db = MusicCatalogDB(item_db_name, track_split_types or ["all_tracks"],
                                corpus_types or ["track_name", "artist_name", "album_name"])
            self.meta_lookup = db.metadata_dict
        self.model = model_path or "gemini-2.5-flash-lite"
        if client is not None:
            self.client = client
        else:
            from ..query_rewriters.gemini_propose import GeminiClient
            self.client = GeminiClient(model=self.model, max_output_tokens=max_output_tokens)
        if system_prompt is not None:
            self.system_prompt = system_prompt
        else:
            self.system_prompt = Path(system_prompt_path or _DEFAULT_PROMPT).read_text(encoding="utf-8")
        self.k = int(k)
        self.max_retries = int(max_retries)
        self.batch_size = int(batch_size)
        self.max_output_tokens = int(max_output_tokens)
        self.cache_root = Path(cache_dir) / "llm_listwise"
        self.cache_root.mkdir(parents=True, exist_ok=True)

    # ---- cache (keyed on query + pool-head + model + max_output_tokens) -------
    def _cache_path(self, query: str, head: list[str]) -> Path:
        key = query + "\n" + "\n".join(head) + "\n" + self.model + "\n" + str(self.max_output_tokens)
        h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:24]
        return self.cache_root / f"{h}.json"

    def _load_cached(self, query: str, head: list[str]) -> Optional[dict]:
        cp = self._cache_path(query, head)
        if cp.exists():
            try:
                payload = json.loads(cp.read_text(encoding="utf-8"))
                if isinstance(payload, dict) and isinstance(payload.get("order"), list):
                    payload["order"] = [int(x) for x in payload["order"]]
                    return payload
            except (OSError, json.JSONDecodeError, ValueError):
                pass
        return None

    def _save_cached(self, query: str, head: list[str], order: list[int],
                     n_parsed: int) -> None:
        # n_parsed = count of valid indices the LLM explicitly returned (pre-merge);
        # the gate's valid-index-fraction diagnostic (mostly-passthrough check) reads it.
        self._cache_path(query, head).write_text(
            json.dumps({"order": order, "n_parsed": int(n_parsed)}, ensure_ascii=False),
            encoding="utf-8")

    # ---- generation ----------------------------------------------------------
    def _generate_one(self, su: tuple[str, str]) -> str:
        system, user = su
        for _ in range(max(1, self.max_retries)):
            try:
                return self.client.generate(system, user)
            except Exception:
                continue  # transient API error -> retry, then degrade to empty
        return ""

    def _generate_raw(self, payloads: list[tuple[str, str]]) -> list[str]:
        if not payloads:
            return []
        workers = max(1, min(self.batch_size, len(payloads)))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(self._generate_one, payloads))

    def rerank(self, queries: list[str], candidate_tids: list[list[str]], topk: int,
               user_ids: Optional[list] = None, goal_categories: Optional[list] = None,
               goal_specificities: Optional[list] = None,
               user_profiles_raw: Optional[list] = None, **kwargs) -> list[list[str]]:
        from .lgbm_rerank import _parse_user_profile  # shared defensive parser
        n = len(queries)
        goal_categories = goal_categories or [None] * n
        goal_specificities = goal_specificities or [None] * n
        user_profiles_raw = user_profiles_raw or [None] * n

        heads = [tids[: self.k] for tids in candidate_tids]
        cached: list[Optional[dict]] = [self._load_cached(queries[i], heads[i])
                                        for i in range(n)]
        n_parsed: list[Optional[int]] = [(c.get("n_parsed") if c else None) for c in cached]
        orders: list[Optional[list[int]]] = [(c["order"] if c else None) for c in cached]
        todo = [i for i in range(n) if orders[i] is None]
        if todo:
            payloads = [build_listwise_prompt(
                queries[i], heads[i], self.meta_lookup, self.k, self.system_prompt,
                profile=_parse_user_profile(user_profiles_raw[i]),
                goal_category=goal_categories[i],
                goal_specificity=goal_specificities[i]) for i in todo]
            raws = self._generate_raw(payloads)
            for i, raw in zip(todo, raws):
                parsed = parse_ranking(raw, len(heads[i]))
                order = merge_order(parsed, len(heads[i]))
                orders[i] = order
                n_parsed[i] = len(parsed)
                # Only persist a SUCCESSFUL generation. An empty parse means the
                # API failed (missing key / transient error -> degraded to "") or
                # returned garbage; caching it would pin permanent passthrough and
                # silently poison every future run. Leave it uncached so the next
                # run retries once the key/API is fixed.
                if parsed:
                    self._save_cached(queries[i], heads[i], order, len(parsed))

        # Per-call diagnostics for the gate cell (valid-index fraction = mean
        # n_parsed/head_len; <0.7 => mostly passthrough, a "win" is uninformative).
        self.diagnostics = {"n_parsed": n_parsed, "head_len": [len(h) for h in heads]}
        if todo:
            n_empty = sum(1 for i in todo if not n_parsed[i])
            if n_empty > len(todo) // 2:
                import warnings
                warnings.warn(
                    f"[llm_listwise] {n_empty}/{len(todo)} generations returned NO "
                    "valid ranking -> passthrough (recall order). Check GEMINI_API_KEY/"
                    "GOOGLE_API_KEY and the model name; the result is uninformative.",
                    stacklevel=2)

        out: list[list[str]] = []
        for i in range(n):
            order = orders[i] if orders[i] is not None else list(range(len(heads[i])))
            reordered = [heads[i][j] for j in order] + candidate_tids[i][self.k:]
            out.append(reordered[:topk])
        return out
