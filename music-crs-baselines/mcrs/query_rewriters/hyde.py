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
                 n_docs: int = 3, max_new_tokens: int = 192):
        self.lm = lm
        self.system_prompt = Path(system_prompt_path).read_text(encoding="utf-8")
        self.cache_root = Path(cache_dir) / "hyde"
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.n_docs = int(n_docs)
        self.max_new_tokens = int(max_new_tokens)

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

    def _generate_one(self, conversation: str) -> str:
        import torch
        messages = build_hyde_messages(conversation, self.system_prompt)
        tok, model = self.lm.tokenizer, self.lm.lm
        prompt_text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt_text, return_tensors="pt").to(self.lm.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                 do_sample=False)
        return tok.batch_decode(
            out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

    def generate_batch(self, conversations: list[str]) -> list[dict]:
        results: list[dict] = []
        for conv in conversations:
            cached = self._load_cached(conv)
            if cached is not None:
                results.append(cached)
                continue
            parsed = parse_hyde_output(self._generate_one(conv))
            if not parsed["hyde_docs"]:
                parsed["hyde_docs"] = [parsed["intent_query"] or conv]
            self._save_cached(conv, parsed)
            results.append(parsed)
        return results
