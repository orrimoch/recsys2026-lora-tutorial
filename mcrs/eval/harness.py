"""F3 — ground-truth record + parity assertion."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mcrs.contracts import SubmissionRow


@dataclass(frozen=True)
class GoldRow:
    session_id: str
    user_id: str
    turn_number: int
    ground_truth_track_id: str  # single gold per turn (official invariant)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GoldRow":
        return cls(d["session_id"], d["user_id"], int(d["turn_number"]),
                   d["ground_truth_track_id"])


def assert_parity(
    predictions: list[SubmissionRow], ground_truth: list[GoldRow],
    catalog_size: int, reference: dict[str, float], *, atol: float = 1e-9,
) -> None:
    """Raise if score_official(...) diverges from a reference official-scorer output."""
    from mcrs.eval.official import score_official

    got = score_official(predictions, ground_truth, catalog_size)
    for key, ref in reference.items():
        if key == "total_catalog_size":
            if got[key] != ref:
                raise AssertionError(f"parity: {key} {got[key]} != {ref}")
        elif abs(float(got[key]) - float(ref)) > atol:
            raise AssertionError(f"parity: {key} {got[key]} != {ref} (atol={atol})")
