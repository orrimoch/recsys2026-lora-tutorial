"""RAG propose-then-ground generation (Tier-1 #3.5).

An LLM proposes real "Artist - Title" suggestions (including new artists in the
same style); the channel grounds each to a catalog track by dense NN. This sources
candidates from the LLM's world knowledge — outside the collaborative/content
graph — directly attacking the new-artist wall. Mirrors HydeGenerator: reuses the
project LM wrapper and hash-caches outputs.

parse_proposals is pure and unit-tested; the LM call is integration (Colab).
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Optional

# "1. Artist - Title" / "1) ..." / "1: ..." — capture the body after the number.
_NUM_LINE = re.compile(r"^\s*\d+\s*[.):]\s*(.+?)\s*$", re.MULTILINE)


def parse_proposals(text: str) -> list[str]:
    """Extract proposal strings ("Artist - Title") from the model output. Accepts
    a numbered list or a JSON array; tolerates surrounding prose / code fences;
    returns [] on failure. Order preserved, duplicates dropped."""
    if not text:
        return []
    items: list[str] = []
    # Try a JSON array first.
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            arr = json.loads(m.group(0))
            if isinstance(arr, list):
                items = [str(x).strip() for x in arr if str(x).strip()]
        except json.JSONDecodeError:
            items = []
    if not items:
        items = [m.group(1).strip() for m in _NUM_LINE.finditer(text) if m.group(1).strip()]
    # Dedupe, keep first occurrence.
    seen, out = set(), []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def build_messages(conversation: str, system_prompt: str, n: int) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt.replace("N", str(n))},
        {"role": "user", "content": f"Conversation:\n{conversation.strip()}\n"},
    ]


class ProposeGenerator:
    def __init__(self, lm, system_prompt_path, cache_dir: str = "./cache",
                 n_proposals: int = 20, max_new_tokens: int = 320, batch_size: int = 16):
        self.lm = lm
        self.system_prompt = Path(system_prompt_path).read_text(encoding="utf-8")
        self.cache_root = Path(cache_dir) / "propose_ground"
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.n_proposals = int(n_proposals)
        self.max_new_tokens = int(max_new_tokens)
        self.batch_size = int(batch_size)

    def _cache_path(self, conversation: str) -> Path:
        h = hashlib.sha1(conversation.encode("utf-8")).hexdigest()[:24]
        return self.cache_root / f"{h}.json"

    def _load_cached(self, conversation: str) -> Optional[dict]:
        cp = self._cache_path(conversation)
        if cp.exists():
            try:
                payload = json.loads(cp.read_text(encoding="utf-8"))
                if isinstance(payload, dict) and "proposals" in payload:
                    return payload
            except (OSError, json.JSONDecodeError):
                pass
        return None

    def _save_cached(self, conversation: str, parsed: dict) -> None:
        self._cache_path(conversation).write_text(
            json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")

    def _generate_raw(self, conversations: list[str]) -> list[str]:
        import torch
        if not conversations:
            return []
        tok, model = self.lm.tokenizer, self.lm.lm
        order = sorted(range(len(conversations)), key=lambda i: -len(conversations[i]))
        out = [""] * len(conversations)
        for s in range(0, len(order), self.batch_size):
            idx = order[s:s + self.batch_size]
            prompts = [
                tok.apply_chat_template(
                    build_messages(conversations[i], self.system_prompt, self.n_proposals),
                    tokenize=False, add_generation_prompt=True)
                for i in idx
            ]
            enc = tok(prompts, return_tensors="pt", padding=True).to(self.lm.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=self.max_new_tokens,
                                     do_sample=False)
            for j, i in enumerate(idx):
                out[i] = tok.decode(gen[j][enc["input_ids"].shape[1]:],
                                    skip_special_tokens=True)
        return out

    def generate_batch(self, queries) -> list[dict]:
        results: list[Optional[dict]] = [self._load_cached(q) for q in queries]
        todo = [i for i, r in enumerate(results) if r is None]
        if todo:
            raws = self._generate_raw([queries[i] for i in todo])
            for i, raw in zip(todo, raws):
                parsed = {"proposals": parse_proposals(raw)[: self.n_proposals]}
                results[i] = parsed
                self._save_cached(queries[i], parsed)
        return [r if r else {"proposals": []} for r in results]
