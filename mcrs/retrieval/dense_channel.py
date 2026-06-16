"""R4 — dense-text channel. Brute-force cosine of the encoded query vs the catalog matrix
(no ANN — the ~47k catalog is one matmul). The encoder is injected (`encode_fn`) so the
retrieval logic is testable without a GPU/model; production wires BGE/E5/qwen3 in. F2 channel.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from mcrs.data.ids import canonical_track_id
from mcrs.retrieval._vec import l2_normalize, topk_indices

EncodeFn = Callable[[list[str]], np.ndarray]


class DenseChannel:
    label = "dense"

    def __init__(self, index_to_id: list[str], matrix: np.ndarray,
                 encode_fn: EncodeFn, normalize: bool = True) -> None:
        self.index_to_id = list(index_to_id)
        self.encode_fn = encode_fn
        self.normalize = normalize
        m = np.asarray(matrix, dtype=np.float32)
        self.matrix = l2_normalize(m) if normalize else m

    def batch_text_to_item_retrieval(
        self, queries: list[str], topk: int,
        batch_context: Optional[list[dict]] = None, user_ids: Optional[list[str]] = None,
    ) -> list[list[str]]:
        if not queries:
            return []
        q = np.asarray(self.encode_fn(queries), dtype=np.float32)
        if q.ndim == 1:
            q = q[None, :]
        if self.normalize:
            q = l2_normalize(q)
        sims = q @ self.matrix.T  # (n_queries, n_tracks)
        out = []
        for row in sims:
            idx = topk_indices(row, topk)
            out.append([canonical_track_id(self.index_to_id[int(i)]) for i in idx])
        return out
