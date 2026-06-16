"""L1 — filtering & top-20 assembly (F2 Filter)."""
from __future__ import annotations

import numpy as np

from mcrs.contracts import Candidate, RankedList, TurnContext, UserProfile
from mcrs.data.catalog import Catalog
from mcrs.filter.assembly import TopKAssembler

_CAT = Catalog([{"track_id": str(i)} for i in range(30)], corpus_types=[])


def _ranked(ids, history=None):
    ctx = TurnContext("s", "u", 1, ["hi"], None,
                      UserProfile("u", 1, "f", "US", []), history or [], "cold")
    return RankedList(turn=ctx, items=[Candidate(track_id=t) for t in ids])


def test_dedup_and_cap_to_20():
    ids = [str(i) for i in range(25)] + ["0", "1"]   # 25 unique + 2 dups
    out = TopKAssembler(_CAT).apply(_ranked(ids))
    assert len(out) == 20 and len(set(out)) == 20
    assert out == [str(i) for i in range(20)]         # order preserved, dups dropped


def test_history_rule_on_removes_history_off_keeps():
    rl = _ranked(["0", "1", "2", "3"], history=["1"])
    assert TopKAssembler(_CAT, history_rule="on").apply(rl) == ["0", "2", "3"]
    assert TopKAssembler(_CAT, history_rule="off").apply(rl) == ["0", "1", "2", "3"]


def test_validity_guard_drops_non_catalog_ids():
    out = TopKAssembler(_CAT).apply(_ranked(["0", "ghost", "2"]))
    assert out == ["0", "2"]


def test_tail_mmr_never_reorders_top_10():
    ids = [str(i) for i in range(20)]
    vecs = {t: np.array([float(int(t)), 1.0], dtype=np.float32) for t in ids}
    plain = TopKAssembler(_CAT).apply(_ranked(ids))
    mmr = TopKAssembler(_CAT, tail_mmr=True, track_vector=lambda t: vecs[t]).apply(_ranked(ids))
    assert mmr[:10] == plain[:10]          # top-10 untouched
    assert set(mmr) == set(plain)          # same items, only tail order may change
    assert len(mmr) == 20
