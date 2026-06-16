"""F2 — canonical data contracts.

The single source of truth for every data object that flows through the pipeline. All
records are causal-by-construction (carry only data available at turn t) and serializable.
See `.claude/documents/features/11_F2_interfaces_contracts_config.md`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol


@dataclass(frozen=True)
class UserProfile:
    user_id: str
    age: Optional[int]
    gender: Optional[str]
    country: Optional[str]
    history_tids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id, "age": self.age, "gender": self.gender,
            "country": self.country, "history_tids": list(self.history_tids),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "UserProfile":
        return cls(
            user_id=d["user_id"], age=d.get("age"), gender=d.get("gender"),
            country=d.get("country"), history_tids=list(d.get("history_tids", [])),
        )


@dataclass(frozen=True)
class TurnContext:
    session_id: str
    user_id: str
    turn_number: int
    utterances: list[str]          # turns 1..t ONLY (no future, no gold)
    goal: Optional[str]
    user_profile: UserProfile
    history_tids: list[str]        # history available up to turn t
    segment: str                   # "cold" | "warm"
    history_embedding: Any = None  # derived, optional (never the gold)

    def __post_init__(self) -> None:
        # Causal guard (F2 §4): utterances are turns 1..t, so exactly turn_number of them.
        if len(self.utterances) != self.turn_number:
            raise ValueError(
                f"non-causal TurnContext: turn_number={self.turn_number} but "
                f"{len(self.utterances)} utterances (must be turns 1..t only)"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id, "user_id": self.user_id,
            "turn_number": self.turn_number, "utterances": list(self.utterances),
            "goal": self.goal, "user_profile": self.user_profile.to_dict(),
            "history_tids": list(self.history_tids), "segment": self.segment,
            "history_embedding": self.history_embedding,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TurnContext":
        return cls(
            session_id=d["session_id"], user_id=d["user_id"],
            turn_number=d["turn_number"], utterances=list(d["utterances"]),
            goal=d.get("goal"), user_profile=UserProfile.from_dict(d["user_profile"]),
            history_tids=list(d.get("history_tids", [])), segment=d["segment"],
            history_embedding=d.get("history_embedding"),
        )


@dataclass
class Query:
    text: str
    structured: Optional[dict] = None
    per_channel: dict[str, str] = field(default_factory=dict)


@dataclass
class Candidate:
    track_id: str
    channel_scores: dict[str, float] = field(default_factory=dict)
    channel_ranks: dict[str, int] = field(default_factory=dict)
    rrf_score: float = 0.0
    features: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id, "channel_scores": dict(self.channel_scores),
            "channel_ranks": dict(self.channel_ranks), "rrf_score": self.rrf_score,
            "features": dict(self.features),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Candidate":
        return cls(
            track_id=d["track_id"],
            channel_scores=dict(d.get("channel_scores", {})),
            channel_ranks=dict(d.get("channel_ranks", {})),
            rrf_score=d.get("rrf_score", 0.0),
            features=dict(d.get("features", {})),
        )


@dataclass
class RankedList:
    turn: TurnContext
    items: list[Candidate]


@dataclass(frozen=True)
class SubmissionRow:
    session_id: str
    user_id: str
    turn_number: int
    predicted_track_ids: list[str]
    predicted_response: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id, "user_id": self.user_id,
            "turn_number": self.turn_number,
            "predicted_track_ids": list(self.predicted_track_ids),
            "predicted_response": self.predicted_response,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SubmissionRow":
        return cls(
            session_id=d["session_id"], user_id=d["user_id"],
            turn_number=d["turn_number"],
            predicted_track_ids=list(d["predicted_track_ids"]),
            predicted_response=d["predicted_response"],
        )


# ----- module interfaces (structural Protocols) -----
class RetrievalChannel(Protocol):
    label: str
    def batch_text_to_item_retrieval(
        self, queries: list[str], topk: int,
        batch_context: Optional[list[dict]] = None, user_ids: Optional[list[str]] = None,
    ) -> list[list[str]]: ...


class Reranker(Protocol):
    def rerank(self, ctx: TurnContext, candidates: list[Candidate]) -> RankedList: ...


class Filter(Protocol):
    def apply(self, ranked: RankedList) -> list[str]: ...


class Responder(Protocol):
    def respond(self, ctx: TurnContext, top_tracks: list[dict]) -> str: ...
