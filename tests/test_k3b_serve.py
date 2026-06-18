"""Tests for mcrs/run/k3b_serve.py — GPU-free (injected fake score_fn).

Five assertions:
1. build_finetuned_k3_chain(..., score_fn=fake) returns a ChainReranker.
2. Running the chain reorders the top cross_encoder_k candidates by fake scores while leaving
   the tail (beyond cross_encoder_k) in K2 (incoming) order.
3. model_revision is carried onto the inner NeuralReranker (adapter_revision pin when present,
   else the adapter path).
4. k3b_submission_rows end-to-end happy path.
5. score_fn=None path calls build_cross_encoder_score_fn with correct lora_adapter/lora_revision
   kwargs (monkeypatched — no GPU load).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mcrs.contracts import Candidate, Query, RankedList, TurnContext, UserProfile
from mcrs.rerank.neural import ChainReranker, NeuralReranker
from mcrs.run.k3b_serve import build_finetuned_k3_chain, k3b_submission_rows


# ---------------------------------------------------------------------------
# Shared fakes
# ---------------------------------------------------------------------------

ADAPTER_PATH = "sha256:abc123"   # local dir or Hub repo id
ADAPTER_REV = "v1.0"             # Hub revision / git SHA pin
CROSS_ENCODER_K = 3  # small so we can assert ordering cleanly


def _make_turn(session_id="s1", turn_number=1) -> TurnContext:
    profile = UserProfile(
        user_id="u1", age=None, gender=None, country=None, history_tids=[]
    )
    return TurnContext(
        session_id=session_id,
        user_id="u1",
        turn_number=turn_number,
        utterances=["I like jazz"] * turn_number,
        goal=None,
        user_profile=profile,
        history_tids=[],
        segment="cold",
    )


def _make_candidates(track_ids: list[str]) -> list[Candidate]:
    return [Candidate(track_id=tid) for tid in track_ids]


class FakeCatalog:
    """Minimal catalog: id_to_metadata just returns the track_id string."""

    def id_to_metadata(self, track_id: str, enriched: bool = True) -> str:
        return f"doc:{track_id}"


class FakeQueryBuilder:
    """Always returns a fixed Query."""

    def build(self, ctx: TurnContext) -> Query:
        return Query(text="jazz piano")


class FakeK2Reranker:
    """Pass-through: returns candidates in the same order they arrived."""

    def rerank(self, ctx: TurnContext, candidates: list[Candidate]) -> RankedList:
        return RankedList(turn=ctx, items=list(candidates))


def make_fake_score_fn(preferred_ids: list[str]):
    """Returns a score_fn that gives preferred_ids scores 100, 99, 98, … and others 0."""

    def score_fn(pairs: list[tuple[str, str]]) -> list[float]:
        scores = []
        for _q, doc in pairs:
            # doc is "doc:<track_id>" — parse the track_id back out
            tid = doc.split("doc:", 1)[1]
            if tid in preferred_ids:
                scores.append(float(100 - preferred_ids.index(tid)))
            else:
                scores.append(0.0)
        return scores

    return score_fn


# ---------------------------------------------------------------------------
# Test 1: return type is ChainReranker
# ---------------------------------------------------------------------------

def test_build_returns_chain_reranker():
    catalog = FakeCatalog()
    qb = FakeQueryBuilder()
    k2 = FakeK2Reranker()
    fake_score = make_fake_score_fn(["t1", "t2", "t3"])

    chain = build_finetuned_k3_chain(
        catalog,
        qb,
        k2,
        adapter=ADAPTER_PATH,
        cross_encoder_k=CROSS_ENCODER_K,
        score_fn=fake_score,
    )

    assert isinstance(chain, ChainReranker), f"expected ChainReranker, got {type(chain)}"


# ---------------------------------------------------------------------------
# Test 2: reordering — top slice sorted by fake scores; tail stays in K2 order
# ---------------------------------------------------------------------------

def test_chain_reorders_top_slice_and_preserves_tail():
    catalog = FakeCatalog()
    qb = FakeQueryBuilder()
    k2 = FakeK2Reranker()

    # preferred = t3 > t2 > t1 (scores 100, 99, 98)
    fake_score = make_fake_score_fn(["t3", "t2", "t1"])

    # cross_encoder_k=3 → only t1/t2/t3 scored; t4/t5 are the tail
    chain = build_finetuned_k3_chain(
        catalog,
        qb,
        k2,
        adapter=ADAPTER_PATH,
        cross_encoder_k=CROSS_ENCODER_K,
        score_fn=fake_score,
    )

    ctx = _make_turn()
    # incoming order: t1, t2, t3 (top-3), then t4, t5 (tail)
    candidates = _make_candidates(["t1", "t2", "t3", "t4", "t5"])
    result = chain.rerank(ctx, candidates)

    ranked_ids = [c.track_id for c in result.items]

    # top slice should be t3 > t2 > t1 (descending fake score)
    assert ranked_ids[:3] == ["t3", "t2", "t1"], (
        f"Top slice mismatch: {ranked_ids[:3]}"
    )
    # tail should stay in original (K2) order: t4, t5
    assert ranked_ids[3:] == ["t4", "t5"], (
        f"Tail order mismatch: {ranked_ids[3:]}"
    )


# ---------------------------------------------------------------------------
# Test 3: model_revision is carried onto the inner NeuralReranker
#   3a: adapter_revision pin takes precedence when supplied
#   3b: falls back to adapter path when adapter_revision is None
# ---------------------------------------------------------------------------

def test_model_revision_uses_adapter_revision_when_supplied():
    catalog = FakeCatalog()
    qb = FakeQueryBuilder()
    k2 = FakeK2Reranker()
    fake_score = make_fake_score_fn(["t1"])

    chain = build_finetuned_k3_chain(
        catalog,
        qb,
        k2,
        adapter=ADAPTER_PATH,
        adapter_revision=ADAPTER_REV,
        cross_encoder_k=CROSS_ENCODER_K,
        score_fn=fake_score,
    )

    neural = chain.rerankers[-1]
    assert isinstance(neural, NeuralReranker), (
        f"Expected last reranker to be NeuralReranker, got {type(neural)}"
    )
    assert neural.model_revision == ADAPTER_REV, (
        f"model_revision should be the revision pin; got {neural.model_revision!r}"
    )


def test_model_revision_falls_back_to_adapter_path_when_no_revision():
    catalog = FakeCatalog()
    qb = FakeQueryBuilder()
    k2 = FakeK2Reranker()
    fake_score = make_fake_score_fn(["t1"])

    chain = build_finetuned_k3_chain(
        catalog,
        qb,
        k2,
        adapter=ADAPTER_PATH,
        # adapter_revision omitted → defaults to None
        cross_encoder_k=CROSS_ENCODER_K,
        score_fn=fake_score,
    )

    neural = chain.rerankers[-1]
    assert isinstance(neural, NeuralReranker)
    assert neural.model_revision == ADAPTER_PATH, (
        f"model_revision should fall back to adapter path; got {neural.model_revision!r}"
    )


# ---------------------------------------------------------------------------
# Test 4: k3b_submission_rows — end-to-end happy path with fake harness pieces
# ---------------------------------------------------------------------------

class FakeFusion:
    """Returns a fixed pool of candidates for any query."""

    def fuse(self, queries, topk, topk_internal=None, batch_context=None, user_ids=None,
             per_channel_queries=None):
        pool = _make_candidates(["t1", "t2", "t3", "t4", "t5"])
        return [pool for _ in queries]


class FakeFilter:
    """Returns the first topk track_ids from the ranked list."""

    def apply(self, ranked: RankedList) -> list[str]:
        return [c.track_id for c in ranked.items[:20]]


def test_k3b_submission_rows_returns_rows():
    catalog = FakeCatalog()
    qb = FakeQueryBuilder()
    k2 = FakeK2Reranker()
    fake_score = make_fake_score_fn(["t3", "t2", "t1"])

    chain = build_finetuned_k3_chain(
        catalog,
        qb,
        k2,
        adapter=ADAPTER_PATH,
        cross_encoder_k=CROSS_ENCODER_K,
        score_fn=fake_score,
    )

    turns = [_make_turn("sess1", 1), _make_turn("sess2", 1)]
    rows = k3b_submission_rows(qb, FakeFusion(), FakeFilter(), chain, turns, topk=20)

    assert len(rows) == 2, f"Expected 2 rows, got {len(rows)}"
    # Each row should have reranked ids; t3 should be first (highest fake score)
    for row in rows:
        assert row.predicted_track_ids[0] == "t3", (
            f"Expected t3 first, got {row.predicted_track_ids[0]}"
        )


# ---------------------------------------------------------------------------
# Test 5: score_fn=None path — build_cross_encoder_score_fn called with
#         correct lora_adapter / lora_revision kwargs (no GPU load)
# ---------------------------------------------------------------------------

def test_score_fn_none_calls_build_cross_encoder_score_fn_with_correct_kwargs():
    """Monkeypatch build_cross_encoder_score_fn to assert it receives the right
    lora_adapter / lora_revision kwargs -- no GPU, no model load.

    `build_cross_encoder_score_fn` is lazy-imported inside the function body as
    `from mcrs.rerank.cross_encoder import build_cross_encoder_score_fn`, so we patch
    it at the source module where Python resolves the name at import time.
    """
    catalog = FakeCatalog()
    qb = FakeQueryBuilder()
    k2 = FakeK2Reranker()
    fake_score = make_fake_score_fn(["t1"])

    with patch(
        "mcrs.rerank.cross_encoder.build_cross_encoder_score_fn",
        return_value=fake_score,
    ) as mock_build:
        chain = build_finetuned_k3_chain(
            catalog,
            qb,
            k2,
            adapter=ADAPTER_PATH,
            adapter_revision=ADAPTER_REV,
            cross_encoder_k=CROSS_ENCODER_K,
            # score_fn=None is the default -- triggers the real build path
        )

    mock_build.assert_called_once()
    _args, kwargs = mock_build.call_args
    assert kwargs.get("lora_adapter") == ADAPTER_PATH, (
        f"lora_adapter should be adapter path; got {kwargs.get('lora_adapter')!r}"
    )
    assert kwargs.get("lora_revision") == ADAPTER_REV, (
        f"lora_revision should be adapter_revision; got {kwargs.get('lora_revision')!r}"
    )
    assert isinstance(chain, ChainReranker)


def test_score_fn_none_no_revision_passes_none_lora_revision():
    """When adapter_revision=None, lora_revision=None is forwarded (HEAD load)."""
    catalog = FakeCatalog()
    qb = FakeQueryBuilder()
    k2 = FakeK2Reranker()
    fake_score = make_fake_score_fn(["t1"])

    with patch(
        "mcrs.rerank.cross_encoder.build_cross_encoder_score_fn",
        return_value=fake_score,
    ) as mock_build:
        build_finetuned_k3_chain(
            catalog,
            qb,
            k2,
            adapter=ADAPTER_PATH,
            # adapter_revision omitted -> None
            cross_encoder_k=CROSS_ENCODER_K,
        )

    _args, kwargs = mock_build.call_args
    assert kwargs.get("lora_adapter") == ADAPTER_PATH
    assert kwargs.get("lora_revision") is None, (
        f"lora_revision should be None when no adapter_revision; got {kwargs.get('lora_revision')!r}"
    )
