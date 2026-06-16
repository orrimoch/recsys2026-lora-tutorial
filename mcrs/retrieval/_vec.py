"""Shared vector helpers for dense/personalization channels."""
from __future__ import annotations

import numpy as np


def l2_normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    if m.ndim == 1:
        n = np.linalg.norm(m)
        return m / n if n else m
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest scores, sorted descending (deterministic tie-break by index)."""
    k = min(k, scores.shape[0])
    if k <= 0:
        return np.empty(0, dtype=int)
    part = np.argpartition(-scores, k - 1)[:k]
    return part[np.argsort(-scores[part], kind="stable")]
