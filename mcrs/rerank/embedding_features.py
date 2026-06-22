"""Embedding-space session features for K2 (idea 5: extend the proven `session_emb_cos`).

`session_emb_cos` (mean of in-session played-track DOC embeddings vs the candidate's doc
embedding) was K2's top-3 new feature. This factory generalizes that "session vibe" cosine to
ANY embedding space — audio-CLAP, CF-bpr, … — so the model gets orthogonal session signal
(sonic vibe, behavioral neighborhood) on top of the text-doc version.

Leak-free (frozen catalog embeddings), per-candidate, causal (only `ctx.history_tids`, the
plays up to turn t). Wired as a `score_fn` in the notebooks exactly like `dense_cos`, so it
auto-gets a per-turn `<name>_norm` column from FeatureBuilder.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from mcrs.contracts import TurnContext


def l2_normalize_rows(mat) -> np.ndarray:
    """Row-wise L2 normalize (float32). Zero rows (e.g. cold tracks with no cf-bpr) stay zero
    — no NaN — so a missing-embedding candidate contributes a 0 cosine, not garbage."""
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def make_session_cos_fn(id_to_index: dict, unit_matrix: np.ndarray
                        ) -> Callable[[TurnContext, str], float]:
    """Return a score_fn(ctx, track_id) -> cosine between the candidate's row in `unit_matrix`
    and the L2-normalized MEAN of the in-session played tracks' rows (`ctx.history_tids`).

    `unit_matrix` rows MUST be L2-normalized (pass it through `l2_normalize_rows`). Returns 0.0
    when the candidate has no row, the session has no in-matrix history (cold turn), or the mean
    is degenerate. The per-turn session vector is cached by (session_id, turn_number) so the mean
    is computed once per turn, not once per candidate.
    """
    cache: dict = {}

    def fn(ctx: TurnContext, track_id: str) -> float:
        j = id_to_index.get(track_id)
        if j is None:
            return 0.0
        key = (ctx.session_id, ctx.turn_number)
        if key in cache:
            sv = cache[key]
        else:
            idx = [id_to_index[h] for h in ctx.history_tids if h in id_to_index]
            if idx:
                m = unit_matrix[idx].mean(axis=0)
                n = float(np.linalg.norm(m))
                sv = (m / n) if n > 0 else None
            else:
                sv = None
            cache[key] = sv
        return float(sv @ unit_matrix[j]) if sv is not None else 0.0

    return fn
