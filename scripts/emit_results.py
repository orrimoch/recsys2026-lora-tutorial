"""Standardized RESULTS_JSON emitter for eval notebooks (autonomous research loop).

The final cell of each eval notebook calls print_results_block(...) so the human can paste
back one fenced, machine-parseable block. Centralizing the format here (not inline in
notebooks) keeps it consistent and testable. The composite is computed by local_eval — the
single source of truth — so this module never reimplements the weights.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.local_eval import compute_composite_projected  # noqa: E402

REQUIRED_FIELDS = (
    "exp", "config", "ndcg@20", "cat_div", "lex_div",
    "llm_judge", "composite", "n_sessions", "gate",
)


def build_results_json(
    exp: str,
    config: int,
    *,
    ndcg: float,
    cat_div: float,
    lex_div: float,
    llm_judge: Optional[float],
    n_sessions: int,
    gate: str,
) -> dict[str, Any]:
    """Assemble the paste-back payload.

    llm_judge is the 1-5 judge mean, or None if the LLM axis was not evaluated this run
    (composite is then retrieval-only). Composite is delegated to local_eval.
    """
    scores = {
        "ndcg@20": float(ndcg),
        "catalog_diversity": float(cat_div),
        "lexical_diversity": float(lex_div),
    }
    composite = round(compute_composite_projected(scores, llm_judge), 4)
    payload = {
        "exp": exp,
        "config": int(config),
        "ndcg@20": round(float(ndcg), 4),
        "cat_div": round(float(cat_div), 4),
        "lex_div": round(float(lex_div), 4),
        "llm_judge": (round(float(llm_judge), 4) if llm_judge is not None else None),
        "composite": composite,
        "n_sessions": int(n_sessions),
        "gate": gate,
    }
    assert set(payload) == set(REQUIRED_FIELDS), f"results payload keys {set(payload)} != REQUIRED_FIELDS"
    return payload


def format_results_block(payload: dict[str, Any]) -> str:
    """Render the sentinel-tagged block (RESULTS_JSON + a JSON line) the human pastes back."""
    return "RESULTS_JSON\n" + json.dumps(payload)


def parse_results_block(text: str) -> dict[str, Any]:
    """Parse a pasted RESULTS_JSON block (inverse of format_results_block).

    Tolerates surrounding whitespace/prose: finds the RESULTS_JSON sentinel line and
    json-decodes the line that follows it.
    """
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "RESULTS_JSON":
            if i + 1 >= len(lines):
                raise ValueError("RESULTS_JSON sentinel has no following JSON line")
            return json.loads(lines[i + 1])
    raise ValueError("no RESULTS_JSON sentinel found in text")


def print_results_block(**kwargs: Any) -> dict[str, Any]:
    """Build + print the RESULTS_JSON block; accepts the same kwargs as build_results_json."""
    payload = build_results_json(**kwargs)
    print(format_results_block(payload))
    return payload
