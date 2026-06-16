"""Two-stage LLM listwise reranker (reranker lever 2).

Stage 1: coarse listwise over ALL k1 candidates (plain/wide) -> catches wall golds at pool-rank
51-100 that the single-stage k=50 reranker never sees. Stage 2: rich listwise over the top-k2
shortlist (short list -> the LLM ranks it well; rich tags+year = lever 1). Final = stage-2 order,
then stage-1's leftover appended (so nothing recalled is dropped below the shortlist).
"""
from mcrs.rerankers.two_stage_listwise import compose_two_stage, TwoStageListwiseReranker


# ---- compose_two_stage: the pure merge (stage-2 ranking first, stage-1 leftover after) ----
def test_compose_stage2_first_then_stage1_leftover():
    s1 = ['a', 'b', 'c', 'd', 'e']      # stage-1 full ordering
    s2 = ['c', 'a', 'b']                # stage-2 reranked the top-3 shortlist
    assert compose_two_stage([s1], [s2], topk=5) == [['c', 'a', 'b', 'd', 'e']]


def test_compose_truncates_to_topk():
    assert compose_two_stage([['a', 'b', 'c', 'd', 'e']], [['c', 'a', 'b']], topk=2) == [['c', 'a']]


def test_compose_no_duplicates_leftover_excludes_stage2():
    out = compose_two_stage([['a', 'b', 'c', 'd']], [['c', 'a']], topk=4)[0]
    assert out == ['c', 'a', 'b', 'd']
    assert len(out) == len(set(out))


def test_compose_empty_stage2_is_stage1_passthrough():
    assert compose_two_stage([['a', 'b', 'c']], [[]], topk=3) == [['a', 'b', 'c']]


def test_compose_handles_multiple_queries():
    out = compose_two_stage([['a', 'b', 'c'], ['x', 'y', 'z']],
                            [['c'], ['z', 'y']], topk=3)
    assert out == [['c', 'a', 'b'], ['z', 'y', 'x']]


# ---- TwoStageListwiseReranker: composition of two stage rerankers (stub stages) ----
class _ReverseStage:
    """Stub stage reranker: reverses each candidate list (deterministic, no LLM)."""
    def rerank(self, queries, candidate_tids, topk, **kwargs):
        return [list(reversed(c))[:topk] for c in candidate_tids]


def test_two_stage_rerank_composes_stage1_wide_then_stage2_short():
    rr = TwoStageListwiseReranker(stage1=_ReverseStage(), stage2=_ReverseStage(), k1=5, k2=3)
    # stage1 reverses 5 -> [e,d,c,b,a]; shortlist top-3 [e,d,c]; stage2 reverses -> [c,d,e];
    # compose -> [c,d,e] + stage1 leftover [b,a] = [c,d,e,b,a]
    out = rr.rerank(['q'], [['a', 'b', 'c', 'd', 'e']], topk=5)
    assert out == [['c', 'd', 'e', 'b', 'a']]


def test_two_stage_threads_kwargs_to_both_stages():
    seen = {'stage1': None, 'stage2': None}

    class _Recorder:
        def __init__(self, tag): self.tag = tag
        def rerank(self, queries, candidate_tids, topk, **kwargs):
            seen[self.tag] = kwargs.get('goal_categories')
            return [list(c)[:topk] for c in candidate_tids]

    rr = TwoStageListwiseReranker(stage1=_Recorder('stage1'), stage2=_Recorder('stage2'), k1=3, k2=2)
    rr.rerank(['q'], [['a', 'b', 'c']], topk=3, goal_categories=['discovery'])
    assert seen['stage1'] == ['discovery'] and seen['stage2'] == ['discovery']
