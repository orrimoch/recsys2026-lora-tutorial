"""CLAP text->audio RECALL channel (wRRF sub-retriever) — P4 / W1.c of the ColBERT plan.

Encodes the QUERY TEXT into the LAION-CLAP joint embedding space (the `larger_clap_music`
text tower) and returns the catalog tracks whose precomputed `audio-laion_clap` audio is
nearest — "tracks that SOUND like what the user is asking for." Because it reads the query
text (not played history), it is COLD-FIRABLE: it fires on turn-1 / Blind, unlike
`clap_recall` which mean-pools PLAYED tracks and contributes [] when there is no history.

This is the plan's orthogonal-to-ColBERT recall lever: it reaches new-artist WALL golds by
ACOUSTICS, a signal the lexical/metadata/late-interaction text channels structurally lack.

The CLAP text tower loads via transformers `ClapModel` (no extra pip dep); the item-side
audio embeddings reuse `load_clap_lookup` (the same `audio-laion_clap` column the audio
channel uses), so query and document live in the same joint space.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from .clap_similarity import load_clap_lookup

DEFAULT_CLAP_MODEL = "laion/larger_clap_music"

# Module-level cache so repeated instantiations share one loaded text tower.
_SHARED_CLAP_TEXT_ENCODER: dict[str, Callable] = {}


def cosine_topk(
    q_mat: np.ndarray, track_mat: np.ndarray, track_ids: Sequence[str], topk: int
) -> list[list[str]]:
    """Per-query top-`topk` track ids by cosine similarity. `q_mat` (Q, d) and
    `track_mat` (N, d) are assumed L2-normalized, so the dot product is cosine."""
    scores = np.asarray(q_mat, dtype=np.float32) @ np.asarray(track_mat, dtype=np.float32).T
    out: list[list[str]] = []
    for i in range(scores.shape[0]):
        row = scores[i]
        if topk >= row.shape[0]:
            order = np.argsort(-row)
        else:
            cand = np.argpartition(-row, topk)[:topk]
            order = cand[np.argsort(-row[cand])]
        out.append([track_ids[j] for j in order[:topk]])
    return out


def _load_clap_text_encoder(model_name: str = DEFAULT_CLAP_MODEL):
    """Lazy-load the LAION-CLAP text tower -> callable: list[str] -> (Q, d) L2-normed.
    Imported lazily so the pure core + unit tests need no transformers/CLAP weights."""
    if model_name in _SHARED_CLAP_TEXT_ENCODER:
        return _SHARED_CLAP_TEXT_ENCODER[model_name]

    import torch
    from transformers import ClapModel, ClapProcessor

    model = ClapModel.from_pretrained(model_name).eval()
    processor = ClapProcessor.from_pretrained(model_name)

    def encode(texts):
        inputs = processor(text=list(texts), return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            emb = model.get_text_features(**inputs)
        v = emb.cpu().numpy().astype(np.float32)
        n = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.maximum(n, 1e-9)

    _SHARED_CLAP_TEXT_ENCODER[model_name] = encode
    return encode


class ClapTextRetriever:
    """CLAP text->audio recall channel. Conforms to `batch_text_to_item_retrieval`.

    Inject `text_encoder` + `track_ids`/`track_mat` for unit tests; in production omit
    them to load the audio lookup (`load_clap_lookup`) and lazy-load the CLAP text tower.
    """

    def __init__(
        self,
        dataset_name: Optional[str] = None,
        split_types: Optional[Sequence[str]] = None,
        corpus_types: Optional[Sequence[str]] = None,
        cache_dir: str = "./cache",
        extra_config: Optional[dict] = None,
        text_encoder: Optional[Callable[[Sequence[str]], np.ndarray]] = None,
        track_ids: Optional[list[str]] = None,
        track_mat: Optional[np.ndarray] = None,
    ):
        self._ec = extra_config or {}
        if track_ids is None or track_mat is None:
            lk = load_clap_lookup(cache_dir)
            track_ids = list(lk.keys())
            track_mat = (np.stack([lk[t] for t in track_ids]).astype(np.float32)
                         if track_ids else np.zeros((0, 512), dtype=np.float32))
        self.track_ids = track_ids
        self.track_mat = track_mat
        self._text_encoder = text_encoder

    def _encode(self, queries):
        if self._text_encoder is None:
            self._text_encoder = _load_clap_text_encoder(
                self._ec.get("clap_text_model", DEFAULT_CLAP_MODEL))
        return self._text_encoder(list(queries))

    def batch_text_to_item_retrieval(
        self, queries, topk, user_ids=None, batch_context=None
    ) -> list[list[str]]:
        if self.track_mat.shape[0] == 0:
            return [[] for _ in queries]
        q_mat = self._encode(queries)
        return cosine_topk(q_mat, self.track_mat, self.track_ids, topk)

    def text_to_item_retrieval(self, query, topk, user_id=None) -> list[str]:
        return self.batch_text_to_item_retrieval([query], topk=topk)[0]
