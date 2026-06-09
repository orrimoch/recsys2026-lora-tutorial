"""Regression tests: the dense query cache must NOT be re-pickled on every
batch.

Root cause of the nb81 cell-2 "disk explosion": DENSE_PRECOMPUTED
(dense_metadata_qwen3_instruct) wrote its query cache to
``{cache_dir}/dense/query_embeddings/*.pkl`` and called ``_save_query_cache()``
on EVERY batch with a cache miss — re-pickling the entire, growing dict each
time. With ``{cache_dir}/dense`` symlinked to a Google Drive FUSE mount (nb81
cell 1), hundreds of full rewrites of a hundreds-of-MB pickle pile up in the
DriveFS upload cache -> the VM disk fills with hundreds of GB and the run
stalls on Drive upload throughput.

dense_multimodal_local.py already fixed this (count-throttled save + explicit
flush). These tests pin the same behavior onto DENSE_PRECOMPUTED.
"""
import numpy as np

from mcrs.retrieval_modules.dense_precomputed import DENSE_PRECOMPUTED
from mcrs.retrieval_modules.dense_local import DENSE_LOCAL


def _bare_instance(tmp_path, save_every=1000):
    """Construct a DENSE_PRECOMPUTED without its disk-loading __init__."""
    inst = object.__new__(DENSE_PRECOMPUTED)
    inst.track_ids = [f"t{i}" for i in range(5)]
    inst.track_mat = np.eye(5, 4, dtype=np.float32)  # (N=5, dim=4)
    inst.instruct = None
    inst._query_cache = {}
    inst._query_cache_dirty = False
    inst._query_cache_path = str(tmp_path / "q.pkl")
    inst._cache_save_every = save_every
    inst._cache_entries_since_save = 0
    return inst


def test_batch_retrieval_does_not_save_every_batch(tmp_path, monkeypatch):
    inst = _bare_instance(tmp_path, save_every=1000)
    saves = []
    monkeypatch.setattr(inst, "_save_query_cache", lambda: saves.append(1))
    monkeypatch.setattr(
        inst, "_encode_queries",
        lambda qs: np.ones((len(qs), 4), dtype=np.float32),
    )

    # 10 batches × 5 unique queries = 50 new entries, well under save_every=1000.
    for b in range(10):
        qs = [f"b{b}_q{i}" for i in range(5)]
        out = inst.batch_text_to_item_retrieval(qs, topk=3)
        assert len(out) == 5

    # The bug wrote once per batch (==10). Throttled: zero mid-loop writes.
    assert saves == [], f"expected no per-batch saves, got {len(saves)}"


def test_flush_query_cache_forces_final_write(tmp_path, monkeypatch):
    inst = _bare_instance(tmp_path, save_every=1000)
    saves = []

    def _spy_save():
        # Mirror the real _save_query_cache contract: clears the dirty flag.
        saves.append(1)
        inst._query_cache_dirty = False

    monkeypatch.setattr(inst, "_save_query_cache", _spy_save)
    monkeypatch.setattr(
        inst, "_encode_queries",
        lambda qs: np.ones((len(qs), 4), dtype=np.float32),
    )
    inst.batch_text_to_item_retrieval(["a", "b"], topk=2)

    inst.flush_query_cache()
    assert len(saves) == 1, "flush must persist the cache once"

    # Nothing new since the flush -> a second flush is a no-op.
    inst.flush_query_cache()
    assert len(saves) == 1


def test_maybe_save_crosses_threshold_then_resets(tmp_path, monkeypatch):
    inst = _bare_instance(tmp_path, save_every=100)
    saves = []
    monkeypatch.setattr(inst, "_save_query_cache", lambda: saves.append(1))
    inst._query_cache_dirty = True

    inst._maybe_save_query_cache(99)
    assert saves == []                      # below threshold
    inst._maybe_save_query_cache(1)
    assert len(saves) == 1                  # crossed 100 -> one write
    assert inst._cache_entries_since_save == 0  # counter reset


def test_dense_local_also_throttles(tmp_path, monkeypatch):
    """DENSE_LOCAL (4B / bge variants) carried the same per-batch-save bug."""
    inst = object.__new__(DENSE_LOCAL)
    inst.track_ids = [f"t{i}" for i in range(5)]
    inst.track_mat = np.eye(5, 4, dtype=np.float32)
    inst.instruct = None
    inst._query_cache = {}
    inst._query_cache_dirty = False
    inst._query_cache_path = str(tmp_path / "q.pkl")
    inst._cache_save_every = 1000
    inst._cache_entries_since_save = 0

    saves = []
    monkeypatch.setattr(inst, "_save_query_cache", lambda: saves.append(1))
    monkeypatch.setattr(
        inst, "_encode_queries",
        lambda qs: np.ones((len(qs), 4), dtype=np.float32),
    )
    for b in range(10):
        inst.batch_text_to_item_retrieval([f"b{b}_q{i}" for i in range(5)], topk=3)
    assert saves == [], f"expected no per-batch saves, got {len(saves)}"
