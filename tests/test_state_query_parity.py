"""Serve-safe [STATE] intent block in the bge_m3_structured query.

The raw `thought` field is leakage-unsafe (empty in Blind-A); the StateTracker
user_state (6 keys) is LM-extracted at BOTH train and serve, so it carries no
schema mismatch. The block is emitted ONLY when a state dict is passed, so the
existing bge_m3_ft v1 format (state=None) is unchanged. Train (format_query_text)
and serve (build_retrieval_query) must produce byte-identical [STATE].
"""
from mcrs.crs_baseline import build_retrieval_query
from mcrs.retrieval_modules.bge_m3_format import format_query_text

_FULL = {"mood": "calm", "intent": "explore", "energy": "low",
         "sonic_pref": "dreamy folk", "era_pref": "2010s", "familiarity": "new"}


def test_state_none_omits_block_v1_backward_compat():
    q = build_retrieval_query([{"role": "user", "content": "play jazz"}],
                              mode="bge_m3_structured", state=None)
    assert "[STATE]" not in q


def test_state_block_renders_six_keys_in_order():
    q = build_retrieval_query([{"role": "user", "content": "play jazz"}],
                              mode="bge_m3_structured", state=_FULL)
    assert ("[STATE]: mood=calm intent=explore energy=low sonic_pref=dreamy folk "
            "era_pref=2010s familiarity=new") in q


def test_missing_and_empty_keys_render_unknown():
    q = build_retrieval_query([{"role": "user", "content": "x"}],
                              mode="bge_m3_structured", state={"mood": "calm", "intent": ""})
    assert ("[STATE]: mood=calm intent=unknown energy=unknown sonic_pref=unknown "
            "era_pref=unknown familiarity=unknown") in q


def test_empty_state_dict_all_unknown_positionally_stable():
    q = build_retrieval_query([{"role": "user", "content": "x"}],
                              mode="bge_m3_structured", state={})
    assert "[STATE]: mood=unknown intent=unknown energy=unknown" in q


def test_train_serve_state_byte_parity():
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey"}]
    cur = "play something dreamy"
    profile = {"age_group": "25-34", "country_code": "US", "gender": "f"}
    goal = {"listener_goal": "wind down"}
    q_train = format_query_text(history, cur, user_profile=profile,
                                conversation_goal=goal, mode="bge_m3_structured", state=_FULL)
    sm = history + [{"role": "user", "content": cur}]
    q_serve = build_retrieval_query(sm, mode="bge_m3_structured",
                                    goal_text=goal["listener_goal"], user_profile=profile, state=_FULL)
    assert q_train == q_serve
    assert "[STATE]:" in q_train
