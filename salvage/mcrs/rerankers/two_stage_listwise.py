"""Two-stage LLM listwise reranker (reranker lever 2).

Single-stage llm_listwise reranks only the top-`k`=50 of the pool and ranks all 50 in one call,
which (a) HIDES wall golds recalled at pool-rank 51-100 and (b) degrades because LLMs rank long
lists poorly. Two stages fix both:

  Stage 1 (coarse, WIDE): listwise over ALL k1 candidates, plain candidate lines -> a coarse
                          ordering whose top-k2 becomes the shortlist (now wall golds at 51-100
                          can be promoted into it).
  Stage 2 (fine, SHORT):  rich listwise over the k2 shortlist -> the LLM ranks a short, rich list
                          well (rich tags+year = lever 1).
  Compose:                stage-2 order first, then stage-1's leftover appended (so nothing
                          recalled below the shortlist is dropped).

`compose_two_stage` is a pure function (unit-tested). `TwoStageListwiseReranker` composes any two
objects exposing the `LLMListwiseReranker.rerank(queries, candidate_tids, topk, **kwargs)` contract,
so it is a drop-in reranker and trivially testable with stub stages.
"""
from typing import Any, Optional


def compose_two_stage(s1_orders: list[list[str]], s2_orders: list[list[str]],
                      topk: int) -> list[list[str]]:
    """Merge per query: the stage-2 ranking first, then stage-1's leftover (items not in stage-2)
    in stage-1 order, truncated to topk. The carefully-ranked shortlist leads; nothing recalled
    below the shortlist is lost. Degrades to stage-1 passthrough when stage-2 is empty."""
    out: list[list[str]] = []
    for s1, s2 in zip(s1_orders, s2_orders):
        seen = set(s2)
        final = list(s2) + [t for t in s1 if t not in seen]
        out.append(final[:topk])
    return out


class TwoStageListwiseReranker:
    """Compose two stage rerankers into one. Stage 1 ranks the wide pool (k1); its top-k2 is the
    shortlist; stage 2 ranks the shortlist; compose_two_stage merges. Same rerank() contract as
    LLMListwiseReranker (drop-in). The factory builds stage1 (plain/wide) + stage2 (rich/short);
    tests inject stubs."""

    def __init__(self, stage1: Any, stage2: Any, k1: int, k2: int) -> None:
        self.stage1 = stage1
        self.stage2 = stage2
        self.k1 = int(k1)
        self.k2 = int(k2)
        self.diagnostics: Optional[dict] = None

    def rerank(self, queries: list[str], candidate_tids: list[list[str]], topk: int,
               **kwargs) -> list[list[str]]:
        # Stage 1: coarse over the wide pool (caps each list at k1).
        s1 = self.stage1.rerank(queries, [c[: self.k1] for c in candidate_tids],
                                topk=self.k1, **kwargs)
        # Shortlist: the top-k2 of each stage-1 ordering.
        shortlists = [order[: self.k2] for order in s1]
        # Stage 2: fine/rich over the shortlist.
        s2 = self.stage2.rerank(queries, shortlists, topk=self.k2, **kwargs)
        # Surface stage-2's gate diagnostics (the fine pass is the one that ranks the head).
        self.diagnostics = getattr(self.stage2, "diagnostics", None)
        return compose_two_stage(s1, s2, topk)
