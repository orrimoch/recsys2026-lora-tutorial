"""Track A — LLM listwise reranker (SASRec_Improved_Plan / plan: drastically
improve nDCG@20). The real Gemini call is integration (Colab); these test the pure
logic with a FAKE client (dependency-injected): candidate render (no UUID), prompt
build carrying all 3 streams + goal/culture, robust ranking parse, full-permutation
merge with graceful degrade, head-reorder/tail-preserve/topk-truncate, API-error
degrade, and on-disk caching. None need google-genai or an API key.
"""
import pytest

from mcrs.rerankers.llm_listwise_rerank import (
    LLMListwiseReranker,
    build_listwise_prompt,
    merge_order,
    parse_ranking,
    render_candidate,
)

# A tiny meta lookup mirroring the raw catalog row shape (list-valued fields).
META = {
    "uuid-aaa": {"track_id": "uuid-aaa", "track_name": ["Creep"],
                 "artist_name": ["Radiohead"], "album_name": ["Pablo Honey"],
                 "tag_list": ["alt rock", "90s"]},
    "uuid-bbb": {"track_id": "uuid-bbb", "track_name": ["Bliss"],
                 "artist_name": ["Muse"], "album_name": ["Origin of Symmetry"],
                 "tag_list": ["rock"]},
    "uuid-ccc": {"track_id": "uuid-ccc", "track_name": ["Teardrop"],
                 "artist_name": ["Massive Attack"], "album_name": ["Mezzanine"],
                 "tag_list": []},
}


# ---- render_candidate --------------------------------------------------------
def test_render_candidate_compact_no_uuid():
    s = render_candidate(1, "uuid-aaa", META)
    assert s.startswith("[1] ")
    assert "Radiohead" in s and "Creep" in s
    assert "uuid-aaa" not in s  # the UUID is wasted context — must be dropped


def test_render_candidate_missing_meta_falls_back_to_tid():
    s = render_candidate(3, "uuid-zzz", META)
    assert s == "[3] uuid-zzz"


# ---- enriched candidate rendering: release-year + more tags (EXP-014) ---------
import re as _re

RICH_META = {
    "uuid-rich": {"track_id": "uuid-rich", "track_name": ["Roygbiv"],
                  "artist_name": ["Boards of Canada"],
                  "album_name": ["Music Has the Right to Children"],
                  "release_date": "1998-04-20",
                  "tag_list": ["electronic", "idm", "ambient", "downtempo",
                               "chillout", "90s", "instrumental", "scottish"]},
}


def test_render_candidate_default_is_unchanged_no_year_5_tags():
    # backward-compat: default keeps current behavior (no year, first 5 tags).
    s = render_candidate(1, "uuid-rich", RICH_META)
    assert not _re.search(r"\(\d{4}\)", s)     # no release year
    assert s.count(",") == 4                    # 5 tags -> 4 commas
    assert "scottish" not in s                  # 8th tag dropped at default cap


def test_render_candidate_include_year_adds_release_year_after_album():
    s = render_candidate(1, "uuid-rich", RICH_META, include_year=True)
    assert "(1998)" in s
    assert s.index("(1998)") < s.index("electronic")   # year before the tags


def test_render_candidate_max_tags_controls_count():
    s = render_candidate(1, "uuid-rich", RICH_META, max_tags=8)
    assert "scottish" in s and s.count(",") == 7        # all 8 tags


def test_render_candidate_include_year_missing_release_date_no_crash():
    s = render_candidate(1, "uuid-aaa", META, include_year=True)  # META has no release_date
    assert not _re.search(r"\(\d{4}\)", s) and "Creep" in s


def test_prompt_rich_candidates_adds_year_and_more_tags():
    _, user = build_listwise_prompt(query="q", tids=["uuid-rich"], meta_lookup=RICH_META,
                                    k=1, system_prompt="SYS", rich_candidates=True)
    assert "(1998)" in user and "scottish" in user


def test_prompt_default_is_lean_no_year(tmp_path=None):
    _, user = build_listwise_prompt(query="q", tids=["uuid-rich"], meta_lookup=RICH_META,
                                    k=1, system_prompt="SYS")
    assert "(1998)" not in user and "scottish" not in user


def test_prompt_rich_renders_goal_category():
    # goal_category is accepted today but NEVER rendered; rich mode surfaces it.
    _, user = build_listwise_prompt(query="q", tids=["uuid-rich"], meta_lookup=RICH_META,
                                    k=1, system_prompt="SYS", rich_candidates=True,
                                    goal_category="discover new chill electronic")
    assert "discover new chill electronic" in user


def test_prompt_default_ignores_goal_category_backward_compat():
    _, user = build_listwise_prompt(query="q", tids=["uuid-rich"], meta_lookup=RICH_META,
                                    k=1, system_prompt="SYS",
                                    goal_category="discover new chill electronic")
    assert "discover new chill electronic" not in user


def test_reranker_rich_candidates_threads_through_to_prompt(tmp_path):
    captured = {}

    class _Cap:
        def generate(self, system, user):
            captured["user"] = user
            return "1"

    rr = LLMListwiseReranker(meta_lookup=RICH_META, cache_dir=str(tmp_path),
                             client=_Cap(), k=1, rich_candidates=True)
    rr.rerank(["q"], [["uuid-rich"]], topk=1)
    assert "(1998)" in captured["user"] and "scottish" in captured["user"]


# ---- build_listwise_prompt (all 3 streams) -----------------------------------
def test_prompt_carries_goal_culture_and_candidates_no_uuid():
    system, user = build_listwise_prompt(
        query="user: something sad and slow\ngoal: find one specific 90s trip-hop song",
        tids=["uuid-aaa", "uuid-ccc"],
        meta_lookup=META,
        k=2,
        system_prompt="SYS",
        profile={"preferred_musical_culture": "Anglo-American Rock",
                 "age_group": "30s", "country_name": "Mexico", "gender": "male"},
        goal_specificity="HH",
    )
    assert system == "SYS"
    # Text Query + Chat History (the goal line is the sharpest target cue)
    assert "find one specific 90s trip-hop song" in user
    # UserID->DB taste stream
    assert "Anglo-American Rock" in user
    # candidate block, index-keyed, no UUIDs leaked
    assert "[1]" in user and "[2]" in user
    assert "uuid-aaa" not in user and "uuid-ccc" not in user
    assert "Massive Attack" in user


def test_prompt_truncates_to_k():
    _, user = build_listwise_prompt(
        query="q", tids=["uuid-aaa", "uuid-bbb", "uuid-ccc"], meta_lookup=META,
        k=2, system_prompt="SYS")
    # only the first k candidates are rendered into the prompt
    assert "[1]" in user and "[2]" in user and "[3]" not in user


# ---- parse_ranking -----------------------------------------------------------
def test_parse_ranking_comma_list_1indexed_to_0indexed():
    assert parse_ranking("3, 1, 2", 3) == [2, 0, 1]


def test_parse_ranking_json_array():
    assert parse_ranking("[2, 1]", 2) == [1, 0]


def test_parse_ranking_dedupes_and_drops_out_of_range():
    # 5 and 0 are out of [1..3]; duplicate 2 dropped (keep first)
    assert parse_ranking("2, 5, 2, 0, 3", 3) == [1, 2]


def test_parse_ranking_prose_and_empty():
    assert parse_ranking("Ranking: 1 then 2.", 2) == [0, 1]
    assert parse_ranking("", 3) == []
    assert parse_ranking("no numbers here", 3) == []


# ---- merge_order (full permutation, graceful fill) ---------------------------
def test_merge_order_appends_missing_in_original_pool_order():
    # parsed only ranked index 2; 0 and 1 missing -> appended in original order
    assert merge_order([2], 4) == [2, 0, 1, 3]


def test_merge_order_empty_is_identity():
    assert merge_order([], 3) == [0, 1, 2]


# ---- LLMListwiseReranker.rerank (fake client) --------------------------------
class _FakeClient:
    """Duck-typed Gemini client: .generate(system, user) -> str."""

    def __init__(self, response="2, 1", raises=False):
        self.response = response
        self.raises = raises
        self.calls = []

    def generate(self, system_instruction, user_content):
        self.calls.append((system_instruction, user_content))
        if self.raises:
            raise RuntimeError("api error")
        return self.response


def _reranker(tmp_path, client, k=2):
    return LLMListwiseReranker(
        meta_lookup=META, cache_dir=str(tmp_path), client=client, k=k,
        system_prompt="rank them")


def test_rerank_reorders_head_preserves_tail_truncates(tmp_path):
    client = _FakeClient("2, 1")  # reverse the 2-item head
    rr = _reranker(tmp_path, client, k=2)
    out = rr.rerank(["q"], [["uuid-aaa", "uuid-bbb", "uuid-ccc"]], topk=3)
    # head [aaa,bbb] -> [bbb,aaa]; tail [ccc] preserved
    assert out == [["uuid-bbb", "uuid-aaa", "uuid-ccc"]]
    out2 = rr.rerank(["q"], [["uuid-aaa", "uuid-bbb", "uuid-ccc"]], topk=2)
    assert out2 == [["uuid-bbb", "uuid-aaa"]]


def test_rerank_api_error_degrades_to_original_order(tmp_path):
    client = _FakeClient(raises=True)
    rr = LLMListwiseReranker(meta_lookup=META, cache_dir=str(tmp_path), client=client,
                             k=3, system_prompt="rank", max_retries=2)
    out = rr.rerank(["q"], [["uuid-aaa", "uuid-bbb", "uuid-ccc"]], topk=3)
    assert out == [["uuid-aaa", "uuid-bbb", "uuid-ccc"]]  # never crashes


def test_rerank_caches_no_second_call(tmp_path):
    client = _FakeClient("2, 1")
    rr = _reranker(tmp_path, client, k=2)
    pool = [["uuid-aaa", "uuid-bbb"]]
    rr.rerank(["q"], pool, topk=2)
    n_after_first = len(client.calls)
    rr.rerank(["q"], pool, topk=2)  # same (query, head) -> cache hit, no call
    assert len(client.calls) == n_after_first


def test_rerank_exposes_valid_index_diagnostics(tmp_path):
    # model ranks only 1 of 3 -> n_parsed=1, head_len=3 (valid-index fraction 1/3)
    client = _FakeClient("3")
    rr = _reranker(tmp_path, client, k=3)
    rr.rerank(["q"], [["uuid-aaa", "uuid-bbb", "uuid-ccc"]], topk=3)
    assert rr.diagnostics["n_parsed"] == [1]
    assert rr.diagnostics["head_len"] == [3]


def test_failed_generation_is_not_cached_then_retried(tmp_path):
    # API error -> empty parse -> must NOT be cached (else permanent passthrough).
    # A later run with a working client must actually call + reorder.
    err = _FakeClient(raises=True)
    rr = LLMListwiseReranker(meta_lookup=META, cache_dir=str(tmp_path), client=err,
                             k=2, system_prompt="rank", max_retries=1)
    pool = [["uuid-aaa", "uuid-bbb"]]
    assert rr.rerank(["q"], pool, topk=2) == [["uuid-aaa", "uuid-bbb"]]  # passthrough
    ok = _FakeClient("2, 1")
    rr.client = ok
    out = rr.rerank(["q"], pool, topk=2)
    assert ok.calls, "failed generation was cached -> retry never happened"
    assert out == [["uuid-bbb", "uuid-aaa"]]  # now actually reordered


def test_rerank_partial_ranking_fills_missing(tmp_path):
    # model returns only the last index; the rest must fill in pool order
    client = _FakeClient("3")
    rr = _reranker(tmp_path, client, k=3)
    out = rr.rerank(["q"], [["uuid-aaa", "uuid-bbb", "uuid-ccc"]], topk=3)
    assert out == [["uuid-ccc", "uuid-aaa", "uuid-bbb"]]
