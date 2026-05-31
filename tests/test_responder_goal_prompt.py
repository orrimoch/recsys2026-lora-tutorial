"""TDD for responder goal injection: the listener_goal text is appended to the
responder's system prompt ONLY when responder_use_goal=True (config-gated,
default off so the shipped config-194 prompt is bit-identical).

_get_system_prompt is exercised on a bare instance (no __init__ -> no model
load) with a fake role_prompt + user_db, so this is a pure string-logic test.
"""
from mcrs.crs_baseline import CRS_BASELINE


class _FakeUserDB:
    def id_to_profile_str(self, uid):
        return f"PROFILE_FOR_{uid}"


def _bare(responder_use_goal=None):
    obj = CRS_BASELINE.__new__(CRS_BASELINE)  # skip __init__ (avoids model load)
    obj.role_prompt = {
        "role_play": "ROLE ",
        "response_generation": "RESPGEN ",
        "personalization": "PERS ",
    }
    obj.user_db = _FakeUserDB()
    if responder_use_goal is not None:
        obj.responder_use_goal = responder_use_goal
    return obj


def test_goal_appended_when_flag_on():
    obj = _bare(responder_use_goal=True)
    s = CRS_BASELINE._get_system_prompt(obj, user_id=None,
                                        goal_text="explore new artists")
    assert "[SESSION GOAL]" in s
    assert "explore new artists" in s


def test_goal_absent_when_flag_off():
    obj = _bare(responder_use_goal=False)
    s = CRS_BASELINE._get_system_prompt(obj, user_id=None,
                                        goal_text="explore new artists")
    assert "[SESSION GOAL]" not in s
    assert "explore new artists" not in s
    # bit-identical to the no-goal prompt
    assert s == "ROLE RESPGEN "


def test_goal_noop_when_text_empty():
    obj = _bare(responder_use_goal=True)
    assert "[SESSION GOAL]" not in CRS_BASELINE._get_system_prompt(
        obj, user_id=None, goal_text=None)
    assert "[SESSION GOAL]" not in CRS_BASELINE._get_system_prompt(
        obj, user_id=None, goal_text="")


def test_default_off_when_attr_missing():
    # Back-compat: an instance built before this flag existed has no attribute;
    # getattr default must treat it as off.
    obj = _bare(responder_use_goal=None)  # no attr set
    s = CRS_BASELINE._get_system_prompt(obj, user_id=None, goal_text="x")
    assert "[SESSION GOAL]" not in s


def test_goal_comes_after_profile_block():
    obj = _bare(responder_use_goal=True)
    s = CRS_BASELINE._get_system_prompt(obj, user_id="u1", goal_text="g")
    assert "PROFILE_FOR_u1" in s
    assert s.index("[SESSION GOAL]") > s.index("PROFILE_FOR_u1")
