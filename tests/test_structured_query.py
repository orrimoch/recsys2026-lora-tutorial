"""Tier-1 #3.1b: LLM structured-query extraction channel (pure core + wiring).

The LLM call is integration (Colab); these test the deterministic core with no
model: JSON parsing, query assembly, channel wiring (fake extractor + fake inner),
union gating, and the leakage guard on the prompt.
"""
from mcrs.query_rewriters.structured_query import (
    parse_structured_json, assemble_query,
)
from mcrs.retrieval_modules.structured_query_channel import StructuredQueryRetriever
from mcrs.retrieval_modules import _wrrf_union_v1_specs


# ---- parse_structured_json ----

def test_parse_valid_json():
    f = parse_structured_json(
        '{"genres":["alt rock"],"moods":["energetic"],"era":"2000s",'
        '"culture":"Anglo","intent":"discover new bands","wants_new_artist":true}')
    assert f["genres"] == ["alt rock"]
    assert f["wants_new_artist"] is True
    assert f["era"] == "2000s"


def test_parse_json_embedded_in_prose():
    txt = 'Here is the result:\n{"genres":["punk"],"intent":"x"}\nThanks!'
    f = parse_structured_json(txt)
    assert f["genres"] == ["punk"]
    assert f["intent"] == "x"


def test_parse_malformed_returns_safe_defaults():
    f = parse_structured_json("not json at all")
    assert f["genres"] == [] and f["moods"] == []
    assert f["era"] == "" and f["culture"] == "" and f["intent"] == ""
    assert f["wants_new_artist"] is False


def test_parse_coerces_types():
    # genres as a string -> wrapped in a list; wants_new_artist truthy -> bool
    f = parse_structured_json('{"genres":"rock","wants_new_artist":"yes"}')
    assert isinstance(f["genres"], list) and f["genres"] == ["rock"]
    assert f["wants_new_artist"] is True


# ---- assemble_query ----

def test_assemble_omits_empty_fields():
    q = assemble_query({"genres": ["alt rock"], "moods": [], "era": "",
                        "culture": "Anglo", "intent": "new bands",
                        "wants_new_artist": True})
    assert "genres: alt rock" in q
    assert "culture: Anglo" in q
    assert "intent: new bands" in q
    assert "moods:" not in q and "era:" not in q


def test_assemble_is_content_only_no_label_noise_when_all_empty():
    assert assemble_query(parse_structured_json("garbage")) == ""


# ---- channel wiring (fake extractor + fake inner) ----

class _FakeExtractor:
    def extract_batch(self, queries, session_ids=None, turn_numbers=None):
        # turn each query into a deterministic synthetic query
        return [f"SQ::{q}" for q in queries]


class _FakeInner:
    def __init__(self):
        self.seen = None

    def batch_text_to_item_retrieval(self, queries, topk, user_ids=None):
        self.seen = list(queries)
        return [[f"t-{q}-{i}" for i in range(topk)] for q in queries]


def test_channel_retrieves_on_the_synthetic_query():
    inner = _FakeInner()
    chan = StructuredQueryRetriever(_FakeExtractor(), inner)
    out = chan.batch_text_to_item_retrieval(["play jazz", "more punk"], topk=3)
    # inner must have received the EXTRACTOR's synthetic queries, not the raw ones
    assert inner.seen == ["SQ::play jazz", "SQ::more punk"]
    assert len(out) == 2 and len(out[0]) == 3


def test_channel_single_query_helper():
    chan = StructuredQueryRetriever(_FakeExtractor(), _FakeInner())
    out = chan.text_to_item_retrieval("hello", topk=2)
    assert len(out) == 2


# ---- union gating ----

def test_structured_query_channel_off_by_default():
    assert "structured_query" not in [s["type"] for s in _wrrf_union_v1_specs({})]


def test_use_structured_query_appends_channel():
    specs = _wrrf_union_v1_specs({"use_structured_query": True, "w_structured_query": 0.6})
    sq = [s for s in specs if s["type"] == "structured_query"]
    assert len(sq) == 1 and sq[0]["weight"] == 0.6


# ---- leakage guard on the prompt ----

def test_prompt_excludes_leaky_fields():
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "music-crs-baselines",
                        "mcrs", "system_prompts", "structured_query.txt")
    text = open(path).read().lower()
    assert "thought" not in text
    assert "goal_progress" not in text
