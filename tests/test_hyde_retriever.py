from mcrs.retrieval_modules.hyde_qwen3 import rrf_fuse, HydeQwen3Retriever


def test_rrf_fuse_combines_per_doc_lists():
    lists = [["a", "b", "c"], ["b", "a", "d"]]
    fused = rrf_fuse(lists, rrf_k=60, topk=3)
    assert fused[:2] == ["a", "b"]  # appear high in both
    assert len(fused) == 3


class _FakeGen:
    def generate_batch(self, conversations):
        return [{"intent_query": "i", "hyde_docs": [f"{c}-d1", f"{c}-d2"]}
                for c in conversations]


class _FakeInner:
    """Returns a deterministic ranked list keyed off the query string."""

    def batch_text_to_item_retrieval(self, queries, topk, user_ids=None):
        return [[f"T:{q}", "shared", "x"] for q in queries]


def test_retriever_generates_then_fuses_per_query():
    r = HydeQwen3Retriever(_FakeGen(), _FakeInner(), topk_per_doc=3, rrf_k=60)
    out = r.batch_text_to_item_retrieval(["Q1", "Q2"], topk=5)
    assert len(out) == 2
    # Q1's fused list draws only from Q1's two docs (T:Q1-d1, T:Q1-d2, shared, x)
    assert "shared" in out[0]
    assert any(t.startswith("T:Q1") for t in out[0])
    assert not any(t.startswith("T:Q2") for t in out[0])
