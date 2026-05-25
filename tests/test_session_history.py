from mcrs.retrieval_modules.session_history import played_tids_from_context

def test_prefers_explicit_history_tids():
    ctx = {"history_tids": ["a", "b"], "chat_history": []}
    assert played_tids_from_context(ctx, catalog_tids={"a", "b", "c"}) == ["a", "b"]

def test_falls_back_to_chat_history_music_turns():
    ctx = {"chat_history": [
        {"role": "user", "content": "play rock"},
        {"role": "music", "content": "a"},
        {"role": "assistant", "content": "here you go"},
        {"role": "music", "content": "zzz-not-in-catalog"},
    ]}
    assert played_tids_from_context(ctx, catalog_tids={"a", "b"}) == ["a"]

def test_empty_context_returns_empty():
    assert played_tids_from_context(None, catalog_tids={"a"}) == []
    assert played_tids_from_context({}, catalog_tids={"a"}) == []
