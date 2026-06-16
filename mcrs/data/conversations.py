"""F1 — Conversations → causal TurnContext stream + a separate gold accessor.

Each turn has role entries user / music / assistant (gold = the `music` content, per the
official make_ground_truth.py). TurnContext exposes only turns 1..t (user utterances) and the
in-session prior-turn golds as history; the turn-t gold and `thought` never enter the context.
"""
from __future__ import annotations

from typing import Iterable, Iterator, Optional

from mcrs.contracts import TurnContext, UserProfile
from mcrs.data.ids import canonical_track_id
from mcrs.data.segment import segment_for


class Conversations:
    def __init__(self, rows: Iterable[dict], cold_threshold: int = 1) -> None:
        rows = list(rows)
        self._order = [r["session_id"] for r in rows]
        self._rows = {r["session_id"]: r for r in rows}
        self.cold_threshold = cold_threshold

    @staticmethod
    def _by_turn(row: dict) -> dict[int, dict[str, str]]:
        by: dict[int, dict[str, str]] = {}
        for e in row["conversations"]:
            tn = int(e["turn_number"])
            by.setdefault(tn, {})[e["role"]] = e["content"]
        return by

    def gold(self, session_id: str, turn_number: int) -> Optional[str]:
        entry = self._by_turn(self._rows[session_id]).get(int(turn_number))
        if not entry or "music" not in entry:
            return None
        return canonical_track_id(entry["music"])

    def _profile(self, row: dict) -> UserProfile:
        up = row.get("user_profile", {}) or {}
        return UserProfile(
            user_id=row["user_id"], age=up.get("age"), gender=up.get("gender"),
            country=up.get("country_name") or up.get("country_code"), history_tids=[],
        )

    def turns(self, split: Optional[str] = None) -> Iterator[TurnContext]:
        for sid in self._order:
            row = self._rows[sid]
            by = self._by_turn(row)
            profile = self._profile(row)
            goal = (row.get("conversation_goal") or {}).get("listener_goal")
            gold_turns = sorted(t for t, e in by.items() if "music" in e)
            session_golds = {t: canonical_track_id(by[t]["music"]) for t in gold_turns}
            for t in gold_turns:
                utterances = [by.get(k, {}).get("user", "") for k in range(1, t + 1)]
                history = [session_golds[k] for k in range(1, t) if k in session_golds]
                yield TurnContext(
                    session_id=sid, user_id=row["user_id"], turn_number=t,
                    utterances=utterances, goal=goal, user_profile=profile,
                    history_tids=history, segment=segment_for(history, self.cold_threshold),
                )

    @classmethod
    def from_disk(cls, path: str, split: str = "test", cold_threshold: int = 1) -> "Conversations":
        from datasets import load_from_disk

        ds = load_from_disk(path)
        d = ds[split] if hasattr(ds, "keys") else ds
        return cls(d, cold_threshold=cold_threshold)
