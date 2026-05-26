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
