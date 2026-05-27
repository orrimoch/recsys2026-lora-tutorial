"""HyDE generation: conversation -> intent query + pseudo-track descriptions.

Reuses the project LM wrapper (LLAMA_MODEL, exposing .lm/.tokenizer/.device)
and caches outputs by a hash of the conversation so the 8K-turn dev pass and
re-runs never re-call the model.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Optional


def build_hyde_messages(conversation: str, system_prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Conversation:\n{conversation.strip()}\n"},
    ]


def parse_hyde_output(text: str) -> dict:
    """Parse model output into {"intent_query": str, "hyde_docs": list[str]}."""
    intent = ""
    docs: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m_intent = re.match(r"(?i)^intent\s*:\s*(.+)$", line)
        if m_intent:
            intent = m_intent.group(1).strip()
            continue
        m_doc = re.match(r"^\d+[.)]\s*(.+)$", line)
        if m_doc:
            docs.append(m_doc.group(1).strip())
    return {"intent_query": intent, "hyde_docs": docs}


class HydeGenerator:
    def __init__(self, lm, system_prompt_path, cache_dir: str = "./cache",
                 n_docs: int = 3, max_new_tokens: int = 192, batch_size: int = 16):
        self.lm = lm
        self.system_prompt = Path(system_prompt_path).read_text(encoding="utf-8")
        self.cache_root = Path(cache_dir) / "hyde"
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.n_docs = int(n_docs)
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
                if isinstance(payload, dict) and "hyde_docs" in payload:
                    return payload
            except (OSError, json.JSONDecodeError):
                pass
        return None

    def _save_cached(self, conversation: str, parsed: dict) -> None:
        self._cache_path(conversation).write_text(
            json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")

    def _generate_raw(self, conversations: list[str]) -> list[str]:
        """Greedy generation for a list of conversations, returned in order.

        Efficiency: process longest-first in sub-batches of `self.batch_size`
        with left-padding (the LM wrapper's tokenizer pads left), so each
        sub-batch wastes minimal padding and does one tokenize + one
        model.generate. Attention masks make the left-pad tokens inert, so the
        greedy output matches single-sequence decoding — the conversation-hash
        cache stays valid regardless of how turns were batched.
        """
        import torch
        if not conversations:
            return []
        tok, model = self.lm.tokenizer, self.lm.lm
        prompts = [
            tok.apply_chat_template(
                build_hyde_messages(c, self.system_prompt),
                tokenize=False, add_generation_prompt=True)
            for c in conversations
        ]
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]), reverse=True)
        out: list[Optional[str]] = [None] * len(prompts)
        for s in range(0, len(order), self.batch_size):
            idx = order[s:s + self.batch_size]
            enc = tok([prompts[i] for i in idx], return_tensors="pt",
                      padding=True, truncation=True, max_length=2048).to(self.lm.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=self.max_new_tokens,
                                     do_sample=False, pad_token_id=pad_id)
            decoded = tok.batch_decode(gen[:, enc.input_ids.shape[1]:],
                                       skip_special_tokens=True)
            for j, i in enumerate(idx):
                out[i] = decoded[j]
        return out  # type: ignore[return-value]

    def generate_batch(self, conversations: list[str]) -> list[dict]:
        results: list[Optional[dict]] = [None] * len(conversations)
        misses: list[int] = []
        for i, conv in enumerate(conversations):
            cached = self._load_cached(conv)
            if cached is not None:
                results[i] = cached
            else:
                misses.append(i)
        if misses:
            raw = self._generate_raw([conversations[i] for i in misses])
            for i, text in zip(misses, raw):
                parsed = parse_hyde_output(text)
                if not parsed["hyde_docs"]:
                    parsed["hyde_docs"] = [parsed["intent_query"] or conversations[i]]
                self._save_cached(conversations[i], parsed)
                results[i] = parsed
        return results  # type: ignore[return-value]
