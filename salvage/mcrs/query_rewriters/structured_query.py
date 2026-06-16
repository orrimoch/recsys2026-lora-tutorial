"""Structured-query extraction (Tier-1 #3.1b).

An LLM distils each turn into a content-only search query (genres/moods/era/
culture/intent + a wants_new_artist pivot flag), which embeds closer to the
gold's content than the raw noisy dialogue and de-emphasizes the seed artist —
surfacing new-artist golds on pivot turns. Mirrors HyDE: reuses the project LM
wrapper (LLAMA_MODEL: .lm/.tokenizer/.device) and caches outputs by a hash of
the input so the dev pass + re-runs never re-call the model.

The deterministic core (parse_structured_json, assemble_query) is pure and
unit-tested; the LM call is integration (Colab).
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Optional

_FIELDS = ("genres", "moods", "era", "culture", "intent", "wants_new_artist")
_LIST_FIELDS = ("genres", "moods")
_STR_FIELDS = ("era", "culture", "intent")
# Assembly order: content fields only (no artist names).
_ASSEMBLE_ORDER = ("genres", "moods", "era", "culture", "intent")

_TRUE = {"true", "yes", "1", "y", "t"}


def _default_fields() -> dict:
    return {"genres": [], "moods": [], "era": "", "culture": "",
            "intent": "", "wants_new_artist": False}


def _coerce(raw: dict) -> dict:
    f = _default_fields()
    for k in _LIST_FIELDS:
        v = raw.get(k)
        if isinstance(v, list):
            f[k] = [str(x).strip() for x in v if str(x).strip()]
        elif isinstance(v, str) and v.strip():
            f[k] = [v.strip()]
    for k in _STR_FIELDS:
        v = raw.get(k)
        f[k] = str(v).strip() if v is not None else ""
    wna = raw.get("wants_new_artist")
    if isinstance(wna, bool):
        f["wants_new_artist"] = wna
    elif isinstance(wna, (int, float)):
        f["wants_new_artist"] = bool(wna)
    elif isinstance(wna, str):
        f["wants_new_artist"] = wna.strip().lower() in _TRUE
    return f


def parse_structured_json(text: str) -> dict:
    """Tolerant parse of the LM output into the fixed field schema. Accepts a bare
    JSON object or one embedded in prose; on any failure returns safe defaults."""
    if not text:
        return _default_fields()
    raw = None
    try:
        raw = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        m = re.search(r"\{.*\}", text, re.DOTALL)  # first { ... last }
        if m:
            try:
                raw = json.loads(m.group(0))
            except json.JSONDecodeError:
                raw = None
    if not isinstance(raw, dict):
        return _default_fields()
    return _coerce(raw)


def assemble_query(fields: dict) -> str:
    """Content-only synthetic query string from the structured fields. Omits
    empty fields; includes NO artist names (artist-agnostic by construction)."""
    parts = []
    for k in _ASSEMBLE_ORDER:
        v = fields.get(k)
        if k in _LIST_FIELDS:
            v = ", ".join(v) if v else ""
        v = (v or "").strip() if isinstance(v, str) else v
        if v:
            parts.append(f"{k}: {v}")
    return " | ".join(parts)


def build_messages(conversation: str, system_prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Conversation:\n{conversation.strip()}\n"},
    ]


class StructuredQueryExtractor:
    """conversation/query string -> content-only synthetic query string.

    extract_batch returns one synthetic query per input query, in order. Falls
    back to the raw query when the model yields no usable fields (so the channel
    never retrieves on an empty string)."""

    def __init__(self, lm, system_prompt_path, cache_dir: str = "./cache",
                 max_new_tokens: int = 160, batch_size: int = 16):
        self.lm = lm
        self.system_prompt = Path(system_prompt_path).read_text(encoding="utf-8")
        self.cache_root = Path(cache_dir) / "structured_query"
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.max_new_tokens = int(max_new_tokens)
        self.batch_size = int(batch_size)

    def _cache_path(self, conversation: str) -> Path:
        h = hashlib.sha1(conversation.encode("utf-8")).hexdigest()[:24]
        return self.cache_root / f"{h}.json"

    def _load_cached(self, conversation: str) -> Optional[str]:
        cp = self._cache_path(conversation)
        if cp.exists():
            try:
                payload = json.loads(cp.read_text(encoding="utf-8"))
                if isinstance(payload, dict) and "synthetic_query" in payload:
                    return payload["synthetic_query"]
            except (OSError, json.JSONDecodeError):
                pass
        return None

    def _save_cached(self, conversation: str, fields: dict, synthetic: str) -> None:
        self._cache_path(conversation).write_text(
            json.dumps({"fields": fields, "synthetic_query": synthetic},
                       ensure_ascii=False, indent=2), encoding="utf-8")

    def _generate_raw(self, conversations: list[str]) -> list[str]:
        """Greedy generation for a list of conversations (longest-first sub-batches
        with left-pad), mirroring HydeGenerator so the hash cache stays valid."""
        import torch
        if not conversations:
            return []
        tok, model = self.lm.tokenizer, self.lm.lm
        order = sorted(range(len(conversations)), key=lambda i: -len(conversations[i]))
        out: list[str] = [""] * len(conversations)
        for s in range(0, len(order), self.batch_size):
            idx = order[s:s + self.batch_size]
            prompts = [
                tok.apply_chat_template(
                    build_messages(conversations[i], self.system_prompt),
                    tokenize=False, add_generation_prompt=True)
                for i in idx
            ]
            enc = tok(prompts, return_tensors="pt", padding=True).to(self.lm.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=self.max_new_tokens,
                                     do_sample=False)
            for j, i in enumerate(idx):
                new = gen[j][enc["input_ids"].shape[1]:]
                out[i] = tok.decode(new, skip_special_tokens=True)
        return out

    def extract_batch(self, queries, session_ids=None, turn_numbers=None) -> list[str]:
        synth: list[Optional[str]] = [self._load_cached(q) for q in queries]
        todo = [i for i, s in enumerate(synth) if s is None]
        if todo:
            raws = self._generate_raw([queries[i] for i in todo])
            for i, raw in zip(todo, raws):
                fields = parse_structured_json(raw)
                q = assemble_query(fields) or queries[i]  # fall back to raw query
                synth[i] = q
                self._save_cached(queries[i], fields, q)
        return [s if s else queries[i] for i, s in enumerate(synth)]
