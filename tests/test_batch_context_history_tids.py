import inspect
from mcrs.crs_baseline import CRS_BASELINE

def test_batch_context_includes_history_tids():
    src = inspect.getsource(CRS_BASELINE.batch_chat)
    assert '"history_tids"' in src, "batch_context must carry history_tids for session channels"

def test_history_tids_prefer_raw_track_id():
    # _played must source the RAW track_id (preferred over the expanded-text
    # content) so the session channels + LGBM reranker get catalog ids. After
    # the bug #1 fix this is delegated to _played_tids_for (which reuses
    # played_tids_from_context, accepting role in music/assistant). Behaviour is
    # covered end-to-end in test_serve_history_tids.py.
    assert "_played_tids_for" in inspect.getsource(CRS_BASELINE.batch_chat), \
        "batch_chat must delegate played-id recovery to _played_tids_for"
    helper = inspect.getsource(CRS_BASELINE._played_tids_for)
    assert "played_tids_from_context" in helper and 'track_id' in helper, \
        "history_tids must source the raw track_id via played_tids_from_context"

def test_reranker_receives_extra_session_info():
    # The reranker call must forward extra_session_info=[{played_tids: ...}] so
    # the LGBM reranker can compute session-continuity features at inference.
    src = inspect.getsource(CRS_BASELINE.batch_chat)
    assert "extra_session_info" in src, "reranker must receive extra_session_info"
    assert '"played_tids"' in src, "extra_session_info entries must carry played_tids"


def test_batch_context_includes_user_dialog_for_sasrec():
    # SASRec was trained + dev-validated on user-turns-only dialog
    # (build_user_dialog). At serve, batch_context must carry user_dialog so the
    # weight-1.0 SASRec channel sees the clean dialog it was trained on, instead
    # of falling back to the noisy full raw query (sasrec_seq.py:59 warns loudly).
    src = inspect.getsource(CRS_BASELINE.batch_chat)
    assert '"user_dialog"' in src, "batch_context must carry user_dialog for SASRec"
    assert "build_user_dialog" in src, \
        "user_dialog must be built via build_user_dialog (user-turns-only)"
