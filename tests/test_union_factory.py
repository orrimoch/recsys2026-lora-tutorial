import inspect
from mcrs.retrieval_modules import load_retrieval_module

def test_factory_source_registers_new_types():
    src = inspect.getsource(load_retrieval_module)
    for key in ('"same_artist"', '"session_cf"', '"wrrf_union_v1"'):
        assert key in src, f"missing factory branch {key}"

def test_union_v1_has_four_subspecs():
    src = inspect.getsource(load_retrieval_module)
    for t in ('"bm25"', '"dense_metadata_qwen3"', '"same_artist"', '"session_cf"'):
        assert t in src
