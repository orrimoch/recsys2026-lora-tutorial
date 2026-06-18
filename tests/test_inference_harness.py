"""D1 — inference & submission harness: orchestration, strict JSON, precheck."""
from __future__ import annotations

import json

import pytest

from mcrs.contracts import SubmissionRow, TurnContext, UserProfile
from mcrs.data.catalog import Catalog
from mcrs.filter.assembly import TopKAssembler
from mcrs.retrieval.fusion import RRFFusion
from mcrs.retrieval.query import QueryBuilder
from mcrs.run.harness import InferenceHarness, validate_submission, write_submission

_CAT = Catalog([{"track_id": t} for t in ["a", "b", "c", "d"]], corpus_types=[])


class _Fake:
    def __init__(self, label, results):
        self.label, self._results = label, results

    def batch_text_to_item_retrieval(self, queries, topk, batch_context=None, user_ids=None):
        return [r[:topk] for r in self._results]


def _turns():
    p = UserProfile("u", 1, "f", "US", [])
    return [
        TurnContext("s1", "u", 1, ["hello"], "goal", p, [], "cold"),
        TurnContext("s1", "u", 2, ["hello", "more"], "goal", p, ["a"], "warm"),
    ]


def _harness(responder=None):
    fusion = RRFFusion([_Fake("bm25", [["a", "b"], ["c", "d"]]),
                        _Fake("dense", [["b", "a"], ["d", "c"]])], k=60)
    return InferenceHarness(QueryBuilder(), fusion, TopKAssembler(_CAT), responder=responder)


def test_harness_rejects_routing_key_matching_no_channel():
    # A per_channel_query_builders key that matches no channel's query_key would silently no-op
    # (channel falls back to the full query); fail loud at construction instead (review #3).
    fusion = RRFFusion([_Fake("bm25", [["a"]]), _Fake("dense", [["b"]])], k=60)
    with pytest.raises(ValueError):
        InferenceHarness(QueryBuilder(), fusion, TopKAssembler(_CAT),
                         per_channel_query_builders={"colbert": QueryBuilder(recency_window=1)})


def test_harness_routes_focused_query_to_colbert_channel():
    # ColBERT must see the focused (recency_window=1) query; other channels see the full query.
    rec = {}

    class _RecCh:
        label = "colbert"
        query_key = "colbert"

        def batch_text_to_item_retrieval(self, queries, topk, batch_context=None, user_ids=None):
            rec["seen"] = list(queries)
            return [["a"] for _ in queries]

    fusion = RRFFusion([_Fake("bm25", [["a", "b"], ["c", "d"]]), _RecCh()], k=60)
    h = InferenceHarness(QueryBuilder(), fusion, TopKAssembler(_CAT),
                         per_channel_query_builders={"colbert": QueryBuilder(recency_window=1)})
    h.run(_turns())
    # turn1 utt ["hello"] goal "goal" -> "hello goal"; turn2 utt ["hello","more"] -> focused "more goal"
    assert rec["seen"] == ["hello goal", "more goal"]


def test_run_produces_one_row_per_turn():
    rows = _harness().run(_turns())
    assert [r.turn_number for r in rows] == [1, 2]
    assert all(isinstance(r, SubmissionRow) for r in rows)
    assert all(len(r.predicted_track_ids) <= 20 for r in rows)
    assert rows[0].predicted_track_ids and set(rows[0].predicted_track_ids) <= _CAT.track_ids


def test_run_without_reranker_uses_fusion_order():
    rows = _harness().run(_turns())
    assert rows[0].predicted_track_ids[0] in {"a", "b"}      # fused top
    assert rows[0].predicted_response == ""                  # no responder => empty


def test_run_with_responder_fills_response():
    class R:
        def respond(self, ctx, top_tracks):
            return f"{len(top_tracks)} picks"
    rows = _harness(responder=R()).run(_turns())
    assert rows[0].predicted_response.endswith("picks")


def test_write_submission_roundtrip_ensure_ascii_false(tmp_path):
    rows = [SubmissionRow("s1", "u", 1, ["a", "b"], "café ☕")]
    p = tmp_path / "prediction.json"
    write_submission(rows, str(p))
    raw = p.read_text()
    assert "café ☕" in raw                                   # ensure_ascii=False
    loaded = json.loads(raw)
    assert loaded[0]["predicted_track_ids"] == ["a", "b"]


def test_validate_passes_for_good_rows_and_full_coverage():
    rows = [SubmissionRow("s1", "u", 1, ["a", "b"], ""),
            SubmissionRow("s1", "u", 2, ["c"], "")]
    validate_submission(rows, catalog=_CAT, expected_keys=[("s1", 1), ("s1", 2)])


def test_validate_raises_on_dupes_overflow_invalid_and_missing():
    with pytest.raises(ValueError):  # duplicate ids
        validate_submission([SubmissionRow("s1", "u", 1, ["a", "a"], "")], catalog=_CAT)
    with pytest.raises(ValueError):  # > 20
        validate_submission([SubmissionRow("s1", "u", 1, [str(i) for i in range(21)], "")])
    with pytest.raises(ValueError):  # non-catalog id
        validate_submission([SubmissionRow("s1", "u", 1, ["ghost"], "")], catalog=_CAT)
    with pytest.raises(ValueError):  # missing (session,turn)
        validate_submission([SubmissionRow("s1", "u", 1, ["a"], "")],
                            expected_keys=[("s1", 1), ("s1", 2)])
