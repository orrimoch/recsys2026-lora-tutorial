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

def test_prefers_track_id_over_expanded_content_for_music_turns():
    # The inference parser expands music content to metadata TEXT but carries
    # the raw id in track_id; the fallback must use track_id, not content.
    ctx = {"chat_history": [
        {"role": "assistant", "content": "Song X by Artist Y (album Z)", "track_id": "a"},
        {"role": "assistant", "content": "Other Song", "track_id": "b"},
    ]}
    assert played_tids_from_context(ctx, catalog_tids={"a", "b", "c"}) == ["a", "b"]

def test_track_id_not_in_catalog_falls_back_to_content():
    # track_id absent/out-of-catalog -> use content if it is a valid id.
    ctx = {"chat_history": [
        {"role": "music", "content": "a"},                       # raw id in content
        {"role": "assistant", "content": "expanded", "track_id": "not-in-catalog"},
    ]}
    assert played_tids_from_context(ctx, catalog_tids={"a", "b"}) == ["a"]

def test_empty_context_returns_empty():
    assert played_tids_from_context(None, catalog_tids={"a"}) == []
    assert played_tids_from_context({}, catalog_tids={"a"}) == []
