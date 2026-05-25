import inspect
from mcrs.crs_baseline import CRS_BASELINE

def test_batch_context_includes_history_tids():
    src = inspect.getsource(CRS_BASELINE.batch_chat)
    assert '"history_tids"' in src, "batch_context must carry history_tids for session channels"

def test_history_tids_prefer_raw_track_id():
    # _played must source the RAW track_id (preferred over the expanded-text
    # content) so the session channels + LGBM reranker get catalog ids.
    src = inspect.getsource(CRS_BASELINE.batch_chat)
    assert 't.get("track_id")' in src, "history_tids must prefer t['track_id']"

def test_reranker_receives_extra_session_info():
    # The reranker call must forward extra_session_info=[{played_tids: ...}] so
    # the LGBM reranker can compute session-continuity features at inference.
    src = inspect.getsource(CRS_BASELINE.batch_chat)
    assert "extra_session_info" in src, "reranker must receive extra_session_info"
    assert '"played_tids"' in src, "extra_session_info entries must carry played_tids"
