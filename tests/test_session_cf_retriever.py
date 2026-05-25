import numpy as np
from mcrs.retrieval_modules.session_cf import SessionCFRetriever

def _build():
    r = SessionCFRetriever.__new__(SessionCFRetriever)
    r.track_ids = ["t1", "t2", "t3", "t4"]
    r.tid_to_idx = {t: i for i, t in enumerate(r.track_ids)}
    mat = np.array([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    r.track_mat = mat / np.linalg.norm(mat, axis=1, keepdims=True)
    r.catalog_tids = set(r.track_ids)
    return r

def test_session_centroid_retrieves_nearest_unplayed():
    r = _build()
    ctx = {"history_tids": ["t1"]}
    out = r.batch_text_to_item_retrieval(["q"], topk=2, batch_context=[ctx])[0]
    assert out[0] == "t2"
    assert "t3" not in out[:1]

def test_no_history_returns_empty():
    r = _build()
    assert r.batch_text_to_item_retrieval(["q"], topk=2, batch_context=[{}])[0] == []
