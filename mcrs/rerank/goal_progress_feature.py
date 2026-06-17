"""K1 feature: causal prior-turn goal-progress rate (session-engagement signal).

Computes, for every (session_id, turn_number) pair, the fraction of PRIOR turns
(1..t-1) that have been labeled MOVES_TOWARD_GOAL.  This is strictly causal: for
turn t, only turns k < t contribute.  Turn 1 defaults to 0.5 (no prior evidence).

IMPORTANT — SERVE/BLIND AVAILABILITY CAVEAT
============================================
This feature requires `goal_progress_assessments` for ALL context turns (1..t-1) to be
AVAILABLE AT SERVE AND BLIND-TEST TIME.  The official competition data supplies
goal_progress_assessments only for the training split; if the blind/dev set rows do NOT
include these assessments, this feature cannot be computed from the input and will fall
back to the neutral default (0.5) for every turn — causing a train/serve distribution
skew that will inflate training importance scores without any actual serve lift.

Before enabling this feature in a submission pipeline:
  1. Confirm that the blind-test rows expose goal_progress_assessments.
  2. Measure dev nDCG@20 with vs. without this feature; gate on a positive measured lift.
  3. If blind availability is unconfirmed, exclude the feature from the final model or
     ensure the serve code hard-defaults to 0.5 (neutral, no skew).

Usage example (inject into FeatureBuilder as a score_fn):
    rate_map = build_prior_goal_progress_rate(conversations)
    gp_fn = make_prior_goal_progress_score_fn(rate_map)
    fb = FeatureBuilder(catalog, channel_labels, score_fns={"prior_gp_rate": gp_fn})
"""
from __future__ import annotations

from typing import Callable

from mcrs.contracts import TurnContext

# Label string for the "moving toward goal" assessment.
_MOVES = "MOVES_TOWARD_GOAL"

# Neutral default returned when there are no prior non-null assessments (e.g. turn 1).
_NEUTRAL = 0.5


def build_prior_goal_progress_rate(conversations) -> dict[tuple[str, int], float]:
    """Build a causal rate map: (session_id, turn_number) -> float in [0, 1].

    For each turn t in each session, the value is the fraction of prior turns
    1..t-1 (that carry a non-null goal_progress_assessment) labeled
    MOVES_TOWARD_GOAL.

    Turns with no prior assessable turns (turn 1, or sessions where all prior
    labels are null) return the neutral default 0.5.

    Causal guarantee: only turns k < t are inspected, never turn t itself.

    Parameters
    ----------
    conversations:
        Any object that supports iteration over raw rows (dicts with keys
        ``session_id`` and ``goal_progress_assessments``) AND a ``.turns()``
        method that yields TurnContext objects for the same sessions.
        NOTE: this is iterated TWICE (once over raw rows, once via ``.turns()``),
        so it must be a RE-ITERABLE object — a one-shot generator would be
        exhausted after the first pass and yield an empty rate map.

    Returns
    -------
    dict mapping (session_id, turn_number) -> float
    """
    # Step 1: build a per-session lookup: turn_number -> assessment label (or None)
    # from the raw rows.
    session_labels: dict[str, dict[int, str | None]] = {}
    for row in conversations:
        sid = row["session_id"]
        assessments = row.get("goal_progress_assessments") or []
        turn_to_label: dict[int, str | None] = {}
        for entry in assessments:
            tn = int(entry["turn_number"])
            label = entry.get("goal_progress_assessment")  # may be None
            turn_to_label[tn] = label
        session_labels[sid] = turn_to_label

    # Step 2: for each (session, turn_t) emitted by conversations.turns(), compute
    # the fraction of PRIOR turns (k < t) with a non-null label that equal MOVES.
    rate_map: dict[tuple[str, int], float] = {}
    for ctx in conversations.turns():
        sid = ctx.session_id
        t = ctx.turn_number
        labels = session_labels.get(sid, {})

        # Gather only prior turns (strictly k < t) with a non-null assessment.
        prior_values = [
            v for k, v in labels.items()
            if k < t and v is not None
        ]

        if not prior_values:
            rate_map[(sid, t)] = _NEUTRAL
        else:
            moves_count = sum(1 for v in prior_values if v == _MOVES)
            rate_map[(sid, t)] = moves_count / len(prior_values)

    return rate_map


def make_prior_goal_progress_score_fn(
    rate_map: dict[tuple[str, int], float],
    default: float = _NEUTRAL,
) -> Callable[[TurnContext, str], float]:
    """Return a score_fn compatible with FeatureBuilder's ``score_fns`` interface.

    The returned function signature is ``fn(ctx: TurnContext, tid: str) -> float``.
    The score is session- and turn-level (broadcast to all candidates); it does NOT
    depend on the track_id argument — every candidate in the same turn receives the
    same value.

    Parameters
    ----------
    rate_map:
        Output of ``build_prior_goal_progress_rate``.
    default:
        Value returned for (session_id, turn_number) pairs absent from rate_map
        (e.g. blind-set turns with no goal_progress_assessments available).
        Defaults to 0.5 (neutral — no information, no skew).

    Returns
    -------
    score_fn: fn(TurnContext, str) -> float
    """

    def score_fn(ctx: TurnContext, tid: str) -> float:  # noqa: ARG001 (tid intentionally unused)
        return rate_map.get((ctx.session_id, ctx.turn_number), default)

    return score_fn
