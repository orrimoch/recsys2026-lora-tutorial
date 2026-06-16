# scripts/build_bge_ce_training_data.py  (top of file)
"""Plain bge cross-encoder training-data builder (Path A v2).
Walks HF train conversations, mines hard negatives from THE canonical serve union
(UNION_EC: sasrec+colbert+bm25+dense+same_artist, clap off, PG off) with real
per-turn history ctx, and writes the EXACT compact query + tag-enriched pipe doc
text the serve bge_reranker uses. One JSONL row per music turn."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "music-crs-baselines")); sys.path.insert(0, str(REPO_ROOT / "scripts"))


def select_ce_negatives(gold_tid: str, candidate_tids: list[str], max_negatives: int) -> list[str]:
    """Pool minus the gold, hardest-first (pool rank order), capped."""
    return [t for t in candidate_tids if t != gold_tid][:max_negatives]


def build_ce_row(query: str, gold_tid: str, neg_tids: list[str], text_map: dict[str, str],
                 user_id: Optional[str], session_id: Optional[str]) -> dict[str, Any]:
    """One JSONL triple; negs absent from text_map dropped from BOTH neg & neg_tids."""
    kept = [t for t in neg_tids if t in text_map]
    return {"query": query, "pos": text_map[gold_tid], "neg": [text_map[t] for t in kept],
            "pos_tid": gold_tid, "neg_tids": kept, "user_id": user_id, "session_id": session_id}
