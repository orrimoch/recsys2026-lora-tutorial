"""Bug #1 fix: the serve path must recover played track_ids from the parser
output (music turns rewritten to role=="assistant"), not the old role=="music"
filter that always returned [] at serve.

Two fixes are covered:
  1. CRS_BASELINE._played_tids_for reuses played_tids_from_context (role in
     music/assistant + track_id fallback), so history_tids matches what the
     same_artist / session_cf channels see.
  2. run_inference_devset.chat_history_parser attaches track_id on music turns
     (mirrors run_inference_blindset), so dev-serve recovers them too.
"""


# ---- fix #1: CRS_BASELINE._played_tids_for ----

def _bare_crs(valid_catalog):
    from mcrs.crs_baseline import CRS_BASELINE
    obj = CRS_BASELINE.__new__(CRS_BASELINE)  # skip __init__ (no model load)
    obj._valid_catalog = valid_catalog
    return obj


def test_played_tids_for_recovers_assistant_turns_with_track_id():
    # This is the production case the bug broke: parser rewrote music -> assistant
    # and carried track_id; the old role=="music" filter missed it.
    obj = _bare_crs({"track-abc", "track-def"})
    ch = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "META:track-abc", "track_id": "track-abc"},
        {"role": "assistant", "content": "just a reply"},  # no track_id -> ignored
    ]
    assert obj._played_tids_for(ch) == ["track-abc"]


def test_played_tids_for_old_role_music_filter_would_have_returned_empty():
    # Demonstrate the bug: with the assistant-rewritten turns, a role=="music"
    # filter yields nothing. The fix must NOT.
    ch = [{"role": "assistant", "content": "META", "track_id": "track-abc"}]
    old = [t for t in ch if t.get("role") == "music"]
    assert old == []                                  # the bug
    assert _bare_crs({"track-abc"})._played_tids_for(ch) == ["track-abc"]  # the fix


def test_played_tids_for_legacy_raw_id_music_turn_preserved():
    # Eval-harness style: role=="music" with a raw-id content must still work.
    obj = _bare_crs({"track-xyz"})
    ch = [{"role": "music", "content": "track-xyz"}]
    assert obj._played_tids_for(ch) == ["track-xyz"]


def test_played_tids_for_none_catalog_graceful_fallback():
    obj = _bare_crs(None)  # db lacked metadata_dict
    ch = [{"role": "assistant", "content": "META", "track_id": "t1"},
          {"role": "user", "content": "hi"}]
    assert obj._played_tids_for(ch) == ["t1"]


# ---- fix #2: dev parser attaches track_id ----

class _StubItemDB:
    def id_to_metadata(self, tid):
        return f"META:{tid}"


class _StubCRS:
    item_db = _StubItemDB()


def test_dev_parser_attaches_track_id_on_music_turns():
    from run_inference_devset import chat_history_parser
    conversations = [
        {"turn_number": 1, "role": "user", "content": "play something"},
        {"turn_number": 2, "role": "music", "content": "track-abc"},
        {"turn_number": 3, "role": "user", "content": "more like that"},
    ]
    ch, q = chat_history_parser(conversations, _StubCRS(), target_turn_number=3)
    music = [t for t in ch if t.get("track_id")]
    assert len(music) == 1
    assert music[0]["track_id"] == "track-abc"
    assert music[0]["role"] == "assistant"
    assert music[0]["content"] == "META:track-abc"  # LM still gets expanded text
    assert q == "more like that"


def test_dev_parser_output_now_recovers_via_helper():
    # End-to-end of the two fixes: dev parser output -> _played_tids_for recovers.
    from run_inference_devset import chat_history_parser
    conversations = [
        {"turn_number": 1, "role": "music", "content": "track-1"},
        {"turn_number": 2, "role": "music", "content": "track-2"},
        {"turn_number": 3, "role": "user", "content": "go"},
    ]
    ch, _ = chat_history_parser(conversations, _StubCRS(), target_turn_number=3)
    obj = _bare_crs({"track-1", "track-2"})
    assert obj._played_tids_for(ch) == ["track-1", "track-2"]
