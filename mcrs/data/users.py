"""F1 — user profiles (demographics only; no listening history in User-Metadata).

Builds the F2 UserProfile. history_tids is empty here — per-user history is in-session
(prior-turn golds, attached by Conversations), not a metadata field.
"""
from __future__ import annotations

from typing import Iterable

from mcrs.contracts import UserProfile


class Users:
    def __init__(self, rows: Iterable[dict]) -> None:
        self._meta: dict[str, dict] = {r["user_id"]: r for r in rows}

    def __contains__(self, user_id: str) -> bool:
        return user_id in self._meta

    def profile(self, user_id: str) -> UserProfile:
        r = self._meta[user_id]
        country = r.get("country_name") or r.get("country_code")
        return UserProfile(
            user_id=user_id, age=r.get("age"), gender=r.get("gender"),
            country=country, history_tids=[],
        )

    @classmethod
    def from_disk(cls, path: str, split: str = "all_users") -> "Users":
        from datasets import load_from_disk

        ds = load_from_disk(path)
        d = ds[split] if hasattr(ds, "keys") else ds
        return cls(d)
