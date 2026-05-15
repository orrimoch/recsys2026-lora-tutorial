"""Pure functions for SID quantizer preprocessing + collision handling."""
from __future__ import annotations

import numpy as np


def concat_modalities(text: np.ndarray, cf: np.ndarray, audio: np.ndarray) -> np.ndarray:
    """L2-normalize each modality independently, concatenate.

    Cold-start CF rows (zero vectors) are preserved as zeros — no divide-by-zero.
    Output dtype is float32 for memory efficiency downstream.
    """
    def _l2(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        if norm == 0:
            return v.astype(np.float32)
        return (v / norm).astype(np.float32)

    return np.concatenate([_l2(text), _l2(cf), _l2(audio)])
