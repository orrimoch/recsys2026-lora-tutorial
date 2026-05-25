import inspect
from mcrs.crs_baseline import CRS_BASELINE

def test_batch_context_includes_history_tids():
    src = inspect.getsource(CRS_BASELINE.batch_chat)
    assert '"history_tids"' in src, "batch_context must carry history_tids for session channels"
