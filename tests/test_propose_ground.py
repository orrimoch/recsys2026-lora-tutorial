"""Tier-1 #3.5: RAG propose-then-ground channel (pure core + wiring).

The LLM call is integration (Colab); these test proposal parsing, channel wiring
(fake generator + fake inner), union gating, and the prompt leakage guard.
"""
from mcrs.query_rewriters.propose_ground import parse_proposals
from mcrs.retrieval_modules.propose_ground_channel import ProposeGroundRetriever
from mcrs.retrieval_modules import _wrrf_union_v1_specs


# ---- parse_proposals ----

def test_parse_numbered_list():
    txt = "1. Radiohead - Creep\n2. Muse - Bliss\n3) Placebo - Pure Morning"
    out = parse_proposals(txt)
    assert out == ["Radiohead - Creep", "Muse - Bliss", "Placebo - Pure Morning"]


def test_parse_json_list():
    out = parse_proposals('["A - x", "B - y"]')
    assert out == ["A - x", "B - y"]


def test_parse_strips_fences_and_prose():
    txt = "Here are picks:\n```\n1. The Cure - Lullaby\n```\nHope it helps!"
    assert parse_proposals(txt) == ["The Cure - Lullaby"]


def test_parse_malformed_returns_empty():
    assert parse_proposals("sorry, I cannot help") == []
    assert parse_proposals("") == []


# ---- channel wiring ----

class _FakeGen:
    def generate_batch(self, queries):
        # 2 proposals for q0, 1 for q1
        return [{"proposals": ["A - x", "B - y"]}, {"proposals": ["C - z"]}]


class _FakeInner:
    def __init__(self):
        self.seen = None

    def batch_text_to_item_retrieval(self, queries, topk, user_ids=None):
        self.seen = list(queries)
        # each proposal grounds to a distinct track id
        return [[f"t::{q}"] for q in queries]


def test_channel_grounds_proposals_and_fuses_per_query():
    inner = _FakeInner()
    chan = ProposeGroundRetriever(_FakeGen(), inner, topk_per_proposal=5)
    out = chan.batch_text_to_item_retrieval(["q0", "q1"], topk=10)
    # inner saw the flattened proposals across both queries
    assert inner.seen == ["A - x", "B - y", "C - z"]
    # q0 fuses its 2 grounded tracks; q1 has its 1
    assert set(out[0]) == {"t::A - x", "t::B - y"}
    assert out[1] == ["t::C - z"]


def test_channel_empty_proposals_yield_empty_list():
    class _EmptyGen:
        def generate_batch(self, queries):
            return [{"proposals": []} for _ in queries]
    chan = ProposeGroundRetriever(_EmptyGen(), _FakeInner(), topk_per_proposal=5)
    assert chan.batch_text_to_item_retrieval(["q"], topk=10) == [[]]


# ---- union gating ----

def test_propose_ground_off_by_default():
    assert "propose_ground" not in [s["type"] for s in _wrrf_union_v1_specs({})]


def test_use_propose_ground_appends_channel():
    specs = _wrrf_union_v1_specs({"use_propose_ground": True, "w_propose_ground": 0.6})
    pg = [s for s in specs if s["type"] == "propose_ground"]
    assert len(pg) == 1 and pg[0]["weight"] == 0.6


# ---- leakage guard ----

def test_prompt_excludes_leaky_fields():
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "music-crs-baselines",
                        "mcrs", "system_prompts", "propose_tracks.txt")
    text = open(path).read().lower()
    assert "thought" not in text and "goal_progress" not in text
