"""Tests for the retrieval-only / frozen-response-reuse path (EXP-016).

batch_chat(generate_response=False) must compute predicted_track_ids exactly as
the full path (retrieval + rerank) but SKIP the responder LM (Stage 2), emitting a
schema-valid "ok" stub. This is what run_inference_blindset.py --retrieval_only
uses to get fresh track_ids cheaply when reusing the frozen 0.50 responses, instead
of paying the ~35-65 min responder run.
"""
import inspect

from mcrs.crs_baseline import CRS_BASELINE


# ---- behavioral: build a minimal mock-self and exercise batch_chat directly ----

def _make_crs(captured):
    """An un-__init__'d CRS_BASELINE with instance attrs/methods stubbed so
    batch_chat's retrieval+rerank path runs without loading any model."""
    crs = object.__new__(CRS_BASELINE)
    # serve flags off — keep the path to the bare retrieval+rerank+respond spine
    crs.query_preprocessing_mode = "raw"
    crs.use_intent_state = False
    crs.intent_state_rewriter = None
    crs.use_state_tracker = False
    crs.state_tracker = None
    crs.use_cmqr = False
    crs.cmqr = None
    crs.sasrec_context_use_goal = False
    crs.retrieval_topk = 100
    crs.top_n_for_prompt = 1
    crs.response_prompt_name = "response_generation_cot_user_state"
    crs.response_max_new_tokens = 64
    crs.response_reranker = None
    crs._relevance_scorer = None

    # instance attrs shadow the class methods (looked up on the instance, so no
    # implicit self binding — these take the same args batch_chat passes).
    crs._get_system_prompt = lambda user_id, goal_text=None: "sys"
    crs._played_tids_for = lambda prior_history: []
    crs._sasrec_dialog_turns = lambda prior_history, q: []
    crs._finalize_topk = lambda items, pool, played_set: list(items)[:20]

    class _Retrieval:
        def batch_text_to_item_retrieval(self, inputs, topk=100, user_ids=None,
                                         batch_context=None):
            # one deterministic pool per query
            return [["t1", "t2", "t3"] for _ in inputs]

    class _Reranker:
        features = []
        def rerank(self, inputs, items, topk=20, **kw):
            # reverse to prove the reranked order (not raw retrieval) is returned
            return [list(reversed(it))[:topk] for it in items]

    class _ItemDB:
        def id_to_metadata(self, tid):
            return f"meta:{tid}"

    class _LM:
        def batch_response_generation(self, *a, **k):
            captured.append("lm_called")
            return ["GENERATED"] * len(a[0])

    crs.retrieval = _Retrieval()
    crs.reranker = _Reranker()
    crs.item_db = _ItemDB()
    crs.lm = _LM()
    return crs


def _batch():
    return [{"user_query": "play something upbeat", "user_id": "u1",
             "session_memory": []}]


def test_retrieval_only_skips_lm_and_stubs_response():
    captured = []
    crs = _make_crs(captured)
    out = crs.batch_chat(_batch(), generate_response=False)
    assert captured == [], "responder LM must NOT be called in retrieval-only mode"
    assert out[0]["response"] == "ok", "retrieval-only must emit the 'ok' stub"


def test_retrieval_only_still_returns_reranked_track_ids():
    crs = _make_crs([])
    out = crs.batch_chat(_batch(), generate_response=False)
    # reranker reverses the pool ["t1","t2","t3"] -> ["t3","t2","t1"]; the stub
    # path must still return the reranked ids (the nDCG axis is unaffected).
    assert out[0]["retrieval_items"] == ["t3", "t2", "t1"]


def test_default_still_calls_responder():
    captured = []
    crs = _make_crs(captured)
    out = crs.batch_chat(_batch())  # generate_response defaults True
    assert captured == ["lm_called"], "default path must invoke the responder LM"
    assert out[0]["response"] != "ok", "default path must return a real response"


# ---- source-level guards (mirror the repo's batch_chat test style) ----

def test_batch_chat_has_generate_response_param():
    sig = inspect.signature(CRS_BASELINE.batch_chat)
    assert "generate_response" in sig.parameters, \
        "batch_chat must accept generate_response (retrieval-only path)"
    assert sig.parameters["generate_response"].default is True, \
        "generate_response must default True so existing callers are unchanged"


def test_blindset_script_wires_retrieval_only():
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    src = (root / "music-crs-baselines" / "run_inference_blindset.py").read_text()
    assert "--retrieval_only" in src, "blindset script must expose --retrieval_only"
    assert "generate_response=not args.retrieval_only" in src, \
        "blindset script must pass generate_response through to batch_chat"
