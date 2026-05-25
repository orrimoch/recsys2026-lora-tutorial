import inspect
import re
from mcrs.retrieval_modules import load_retrieval_module

def test_factory_source_registers_new_types():
    src = inspect.getsource(load_retrieval_module)
    # session_cf is still a registered standalone branch (reused for the LGBM
    # cfbpr feature) even though it was dropped from the wrrf_union_v1 channels.
    for key in ('"same_artist"', '"session_cf"', '"wrrf_union_v1"'):
        assert key in src, f"missing factory branch {key}"

def _union_v1_branch_src():
    """Just the wrrf_union_v1 elif block, so assertions about its channels
    aren't satisfied by the unrelated standalone session_cf branch."""
    src = inspect.getsource(load_retrieval_module)
    start = src.index('elif retrieval_type == "wrrf_union_v1":')
    rest = src[start:]
    m = re.search(r'\n    (elif retrieval_type ==|else:)', rest[1:])
    return rest[: m.start() + 1] if m else rest

def test_union_v1_is_three_channels_without_session_cf():
    block = _union_v1_branch_src()
    for t in ('"bm25"', '"dense_metadata_qwen3"', '"same_artist"'):
        assert t in block, f"union missing kept channel {t}"
    # Dropped after G1 ablation (marginal recall@100 +0.002 < 0.01 keep rule).
    assert '"session_cf"' not in block, "session_cf should be dropped from wrrf_union_v1"
