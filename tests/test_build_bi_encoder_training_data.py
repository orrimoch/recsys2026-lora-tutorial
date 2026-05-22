"""Tests for build_bi_encoder_training_data.py orchestration."""
import pytest


def test_build_triples_for_row_writes_query_pos_neg():
    """Each output row is a dict with 'query', 'pos' (list), 'neg' (list)."""
    from scripts.build_bi_encoder_training_data import build_triples_for_row

    row = {
        "session_id": "s1",
        "user_id": "u1",
        "turn_number": 1,
        "current_user_query": "play me 70s rock",
        "chat_history": [],
        "user_profile_raw": {"age": 30, "country_code": "US"},
        "conversation_goal": {"listener_goal": "discover music"},
        "track_id": "t_gold",
    }
    track_text_map = {
        "t_gold": "track_name: gold | artist: a | album: b | release_date: 2023 | tag_list: rock",
        "t_neg1": "track_name: neg1 | ...",
        "t_neg2": "track_name: neg2 | ...",
    }
    triple = build_triples_for_row(
        row, gold_track_id="t_gold", neg_track_ids=["t_neg1", "t_neg2"],
        track_text_map=track_text_map,
    )
    assert "query" in triple
    assert "pos" in triple and isinstance(triple["pos"], list) and len(triple["pos"]) == 1
    assert triple["pos"][0] == track_text_map["t_gold"]
    assert "neg" in triple and len(triple["neg"]) == 2
    assert triple["neg"][0] == track_text_map["t_neg1"]


def test_build_triples_for_row_drops_negs_not_in_map():
    """Negatives whose track_id isn't in the text map are silently dropped (catalog drift)."""
    from scripts.build_bi_encoder_training_data import build_triples_for_row
    row = {"session_id": "s", "current_user_query": "q", "chat_history": [],
           "user_profile_raw": None, "conversation_goal": None, "track_id": "t1"}
    track_text_map = {"t1": "gold", "t_neg1": "n1"}  # t_neg2 missing
    triple = build_triples_for_row(
        row, gold_track_id="t1", neg_track_ids=["t_neg1", "t_neg2"],
        track_text_map=track_text_map,
    )
    assert len(triple["neg"]) == 1
    assert triple["neg"][0] == "n1"


def test_build_triples_for_row_default_mode_is_bge_m3_structured():
    """B-1 regression: training-time query format MUST match production
    config 180's query_preprocessing_mode (bge_m3_structured). If this
    drifts, the fine-tune is optimized for a distribution the runtime
    never sees → near-zero or negative leaderboard lift after ~10 GPU-hr."""
    from scripts.build_bi_encoder_training_data import build_triples_for_row
    row = {
        "chat_history": [{"role": "user", "content": "I like 70s rock"}],
        "current_user_query": "play me upbeat",
        "user_profile_raw": {"age_group": "25-34", "country_code": "US"},
        "conversation_goal": {"listener_goal": "discover"},
        "track_id": "t_gold",
    }
    triple = build_triples_for_row(
        row, "t_gold", ["t_neg1"], {"t_gold": "G", "t_neg1": "N1"},
    )
    # Default mode must emit the structured 4-block format.
    assert "[USER]:" in triple["query"], \
        f"default mode is not bge_m3_structured: {triple['query']!r}"
    assert "[GOAL]:" in triple["query"]
    assert "[HISTORY]:" in triple["query"]
    assert "[QUERY]:" in triple["query"]


def test_build_triples_for_row_explicit_raw_mode_still_works():
    """Back-compat: callers can opt into the old raw mode via query_mode='raw'."""
    from scripts.build_bi_encoder_training_data import build_triples_for_row
    row = {
        "chat_history": [], "current_user_query": "q",
        "user_profile_raw": None, "conversation_goal": None, "track_id": "t1",
    }
    triple = build_triples_for_row(
        row, "t1", ["n1"], {"t1": "G", "n1": "N"}, query_mode="raw",
    )
    assert "[USER]:" not in triple["query"]
    assert "user: q" in triple["query"]


def test_iter_conversation_turns_drops_empty_user_query():
    """I-2 regression: a 'user' turn with empty content followed by a 'music'
    turn must NOT produce a degenerate training row (would teach the model
    to map a blank query to a specific track)."""
    from scripts.build_bi_encoder_training_data import _iter_conversation_turns

    sessions = [{
        "session_id": "s1",
        "user_profile": None,
        "conversation_goal": None,
        "conversations": [
            {"role": "user", "content": ""},          # blank — must be skipped
            {"role": "music", "content": "track_X"},  # gold
            {"role": "user", "content": "real query"},
            {"role": "music", "content": "track_Y"},
        ],
    }]
    rows = _iter_conversation_turns(sessions)
    # Only ONE row should emit (the real query → track_Y); the blank→track_X
    # is dropped because its user content was empty.
    assert len(rows) == 1
    assert rows[0]["current_user_query"] == "real query"
    assert rows[0]["track_id"] == "track_Y"


def test_build_triples_for_row_emits_session_id():
    """Sub 2 fix: emit session_id so TripleJsonlDataset can split val
    SESSION-disjoint. Without this, row-shuffle val splits create 95%+
    session overlap with train (Sub 1 leak)."""
    from scripts.build_bi_encoder_training_data import build_triples_for_row
    row = {
        "session_id": "session_42",
        "chat_history": [{"role": "user", "content": "hi"}],
        "current_user_query": "play rock",
        "user_profile_raw": None,
        "conversation_goal": None,
        "track_id": "t_gold",
    }
    triple = build_triples_for_row(
        row, "t_gold", ["n1"], {"t_gold": "G", "n1": "N"},
    )
    assert "session_id" in triple, "build_triples_for_row must emit session_id"
    assert triple["session_id"] == "session_42"


def test_builder_uses_sentence_transformers_not_bgem3_specific():
    """The build script must use SentenceTransformer (not BGEM3FlagModel) so it
    works for any retriever backbone (BGE-M3, bge-base-en-v1.5, bge-large-en-v1.5).
    BGEM3FlagModel is M3-specific and would crash on bge-base."""
    import inspect
    from scripts import build_bi_encoder_training_data as mod
    main_src = inspect.getsource(mod.main)
    # Must NOT instantiate BGEM3FlagModel (that's the M3-specific path).
    assert "BGEM3FlagModel(" not in main_src, \
        "main() still instantiates BGEM3FlagModel — won't work for non-M3 backbones"
    # Must use SentenceTransformer instead.
    assert "SentenceTransformer(" in main_src, \
        "main() must use SentenceTransformer for encoder loading (generic across backbones)"


def test_builder_calls_encoder_with_normalize_embeddings_true():
    """The new encoding path uses normalize_embeddings=True (matches production's
    DENSE_LOCAL._encode_queries contract). Without this, embedding norms diverge
    from production and similarity scores are off."""
    import inspect
    from scripts import build_bi_encoder_training_data as mod
    main_src = inspect.getsource(mod.main)
    assert "normalize_embeddings=True" in main_src, \
        "encoder calls must pass normalize_embeddings=True"


def test_builder_uses_fp16_on_gpu():
    """FP16 inference on GPU for ~2× speed on Blackwell — matches old
    BGEM3FlagModel(use_fp16=True). Catalog encoding of 47K tracks is otherwise slow."""
    import inspect
    from scripts import build_bi_encoder_training_data as mod
    main_src = inspect.getsource(mod.main)
    # `.half()` is the sentence-transformers idiom for FP16 weights.
    assert "model.half()" in main_src or "model = model.half()" in main_src, \
        "missing FP16 conversion on GPU (model.half())"


def test_builder_imports_batch_mine_negatives():
    """After the vectorization patch, main() must use batch_mine_negatives
    (one GPU matmul per batch) instead of the per-query mine_negatives_for_query.
    Source-level check to prevent regression."""
    import inspect
    from scripts import build_bi_encoder_training_data as mod
    src = inspect.getsource(mod)
    assert "batch_mine_negatives" in src, "main() must import + use batch_mine_negatives"
    # The per-query function may still be imported for back-compat, but the
    # mining LOOP must call the batched function.
    main_src = inspect.getsource(mod.main)
    assert "batch_mine_negatives(" in main_src, \
        "main()'s mining loop must call batch_mine_negatives(...) not the per-query version"


def test_builder_pins_catalog_to_gpu_once():
    """Vectorized mining pins the catalog tensor once outside the batch loop
    so we don't re-upload to GPU on every batch (saves ~380GB of transfers
    on a full 120K-query mine)."""
    import inspect
    from scripts import build_bi_encoder_training_data as mod
    main_src = inspect.getsource(mod.main)
    # Look for the GPU pin happening before the mining loop.
    pin_idx = main_src.find("track_embs_dev")
    loop_idx = main_src.find("tqdm(range(0, len(train_rows)")
    assert pin_idx > 0 and loop_idx > 0, "expected both track_embs_dev and the mining loop"
    assert pin_idx < loop_idx, "catalog must be pinned to GPU BEFORE the batch loop"


def test_pos_neg_format_matches_history_music_turn_format():
    """Patch 4 (2026-05-22): pos/neg track text MUST use the same
    id_to_metadata-aligned format as [HISTORY] music-turn references and
    as the catalog vectors at inference. Previously pos/neg used
    format_track_text (5 fields, pipe-separated) while [HISTORY] used
    id_to_metadata (4 fields, comma-separated, lowercased) — forcing the
    encoder to learn two representations of every track. Aligning them
    closes the BlindA nDCG regression gap.

    The contract: when track_text_map is built via _format_history_music_turn,
    build_triples_for_row's pos/neg fields contain id_to_metadata-format text.
    """
    from scripts.build_bi_encoder_training_data import (
        build_triples_for_row, _format_history_music_turn,
    )

    metadata_dict = {
        "tk_gold": {"track_id": "tk_gold",
                    "track_name": ["Hotel California"],
                    "artist_name": ["Eagles"],
                    "album_name": ["Hotel California"]},
        "tk_n1":   {"track_id": "tk_n1",
                    "track_name": ["Stairway To Heaven"],
                    "artist_name": ["Led Zeppelin"],
                    "album_name": ["Led Zeppelin IV"]},
    }
    corpus_types = ["track_name", "artist_name", "album_name"]
    # Build track_text_map exactly the way main() now does.
    track_text_map = {
        tid: _format_history_music_turn(tid, metadata_dict, corpus_types)
        for tid in metadata_dict
    }
    row = {
        "session_id": "s1", "user_id": "u1",
        "chat_history": [], "current_user_query": "play rock",
        "user_profile_raw": None, "conversation_goal": None,
        "track_id": "tk_gold",
    }
    triple = build_triples_for_row(row, "tk_gold", ["tk_n1"], track_text_map)
    # pos field MUST be id_to_metadata format (track_id + 3 fields, comma, lowercased).
    assert triple["pos"][0].startswith("track_id: tk_gold,"), \
        f"pos not in id_to_metadata format: {triple['pos'][0]!r}"
    assert "track_name: hotel california" in triple["pos"][0]
    assert "artist_name: eagles" in triple["pos"][0]
    # neg field same format.
    assert triple["neg"][0].startswith("track_id: tk_n1,")
    assert "track_name: stairway to heaven" in triple["neg"][0]
    # Crucial: no pipe separators (would indicate the OLD format_track_text path).
    assert " | " not in triple["pos"][0]
    assert " | " not in triple["neg"][0]


def test_iter_conversation_turns_emits_user_id():
    """Δ1: rows must carry user_id from session['user_id']. Required for
    user-disjoint train/val split downstream (TripleJsonlDataset.split_key='user_id').
    Without this propagation, all sessions of one user can leak across train/val."""
    from scripts.build_bi_encoder_training_data import _iter_conversation_turns

    sessions = [{
        "session_id": "s_alpha",
        "user_id": "user_42",
        "user_profile": None,
        "conversation_goal": None,
        "conversations": [
            {"role": "user", "content": "play rock"},
            {"role": "music", "content": "track_A"},
        ],
    }]
    rows = _iter_conversation_turns(sessions)
    assert len(rows) == 1
    assert "user_id" in rows[0], "rows must carry user_id"
    assert rows[0]["user_id"] == "user_42"


def test_build_triples_for_row_emits_user_id():
    """Δ1: the JSONL triple carries user_id alongside session_id."""
    from scripts.build_bi_encoder_training_data import build_triples_for_row
    row = {
        "session_id": "session_42",
        "user_id": "user_99",
        "chat_history": [{"role": "user", "content": "hi"}],
        "current_user_query": "play rock",
        "user_profile_raw": None,
        "conversation_goal": None,
        "track_id": "t_gold",
    }
    triple = build_triples_for_row(
        row, "t_gold", ["n1"], {"t_gold": "G", "n1": "N"},
    )
    assert "user_id" in triple, "build_triples_for_row must emit user_id"
    assert triple["user_id"] == "user_99"


def test_build_triples_for_row_emits_neg_tids_in_order():
    """Δ3 issue C (extension per RocketQAv2 / BGE-M3 §3.3): emit neg_tids
    parallel to neg texts so the in-batch loss can mask the
    cross-pos-into-neg-slot collision (this query's gold appearing as a
    hard negative for another query in the batch)."""
    from scripts.build_bi_encoder_training_data import build_triples_for_row
    row = {
        "session_id": "s1", "user_id": "u1",
        "chat_history": [], "current_user_query": "q",
        "user_profile_raw": None, "conversation_goal": None,
        "track_id": "t_gold",
    }
    text_map = {"t_gold": "G", "t_n1": "N1", "t_n2": "N2", "t_n3": "N3"}
    triple = build_triples_for_row(
        row, "t_gold", ["t_n1", "t_n2", "t_n3"], text_map,
    )
    assert "neg_tids" in triple, "must emit neg_tids alongside neg texts"
    # Order MUST match `neg` exactly so loss-time masking can map slot k → tid.
    assert triple["neg_tids"] == ["t_n1", "t_n2", "t_n3"]
    assert triple["neg"] == ["N1", "N2", "N3"]


def test_build_triples_for_row_neg_tids_filter_matches_neg_filter():
    """When a neg's tid is absent from track_text_map (catalog drift), it's
    dropped from BOTH `neg` and `neg_tids` so their lengths stay equal."""
    from scripts.build_bi_encoder_training_data import build_triples_for_row
    row = {
        "session_id": "s", "user_id": "u",
        "chat_history": [], "current_user_query": "q",
        "user_profile_raw": None, "conversation_goal": None,
        "track_id": "t1",
    }
    # t_missing absent from map → must be dropped from BOTH lists.
    text_map = {"t1": "G", "t_n1": "N1"}
    triple = build_triples_for_row(
        row, "t1", ["t_n1", "t_missing"], text_map,
    )
    assert len(triple["neg"]) == 1
    assert len(triple["neg_tids"]) == 1
    assert triple["neg_tids"][0] == "t_n1"


def test_format_history_music_turn_matches_id_to_metadata():
    """The builder's history expander must produce BYTE-IDENTICAL output to
    `mcrs.db_item.music_catalog.MusicCatalogDB.id_to_metadata` so training-
    time [HISTORY] music turns match Blind-A / devset inference exactly."""
    from scripts.build_bi_encoder_training_data import _format_history_music_turn

    metadata_dict = {
        "tk_a14b7": {
            "track_id": "tk_a14b7",
            "track_name": ["Hotel California"],
            "artist_name": ["Eagles"],
            "album_name": ["Hotel California"],
        }
    }
    corpus_types = ["track_name", "artist_name", "album_name"]
    out = _format_history_music_turn("tk_a14b7", metadata_dict, corpus_types)
    # Matches id_to_metadata exactly: track_id prefix, comma+space separator,
    # lowercased values, fields in corpus_types order.
    assert out == (
        "track_id: tk_a14b7, "
        "track_name: hotel california, "
        "artist_name: eagles, "
        "album_name: hotel california"
    )


def test_format_history_music_turn_falls_back_to_raw_id_when_missing():
    """Catalog drift: a track_id that's not in metadata_dict gets emitted
    as-is so the row doesn't disappear silently."""
    from scripts.build_bi_encoder_training_data import _format_history_music_turn

    metadata_dict = {}
    out = _format_history_music_turn("tk_unknown", metadata_dict,
                                      ["track_name", "artist_name", "album_name"])
    assert out == "tk_unknown"


def test_iter_conversation_turns_expands_music_history_when_metadata_provided():
    """Training-history music turns get expanded to id_to_metadata format
    when metadata_dict is passed — matches Blind-A inference exactly."""
    from scripts.build_bi_encoder_training_data import _iter_conversation_turns

    sessions = [{
        "session_id": "s1", "user_id": "u1",
        "user_profile": None, "conversation_goal": None,
        "conversations": [
            {"role": "user",  "content": "play rock"},
            {"role": "music", "content": "tk_A"},
            {"role": "user",  "content": "more drums"},
            {"role": "music", "content": "tk_B"},
        ],
    }]
    metadata_dict = {
        "tk_A": {"track_id": "tk_A",
                 "track_name": ["Hotel California"],
                 "artist_name": ["Eagles"],
                 "album_name": ["Hotel California"]},
        "tk_B": {"track_id": "tk_B",
                 "track_name": ["Free Bird"],
                 "artist_name": ["Lynyrd Skynyrd"],
                 "album_name": ["Pronounced Leh-Nerd Skin-Nerd"]},
    }
    corpus_types = ["track_name", "artist_name", "album_name"]
    rows = _iter_conversation_turns(
        sessions,
        metadata_dict=metadata_dict,
        corpus_types=corpus_types,
    )
    # Row 0 emits the gold track for the first user→music pair (gold=tk_A,
    # chat_history empty at the time of emission).
    # Row 1 emits gold=tk_B with chat_history containing the previous user
    # turn and the previous music turn (expanded).
    assert len(rows) == 2
    # Row 1's chat_history MUST have tk_A expanded to id_to_metadata format.
    hist = rows[1]["chat_history"]
    music_turn_content = hist[1]["content"]  # second entry is the music turn
    assert music_turn_content.startswith("track_id: tk_A"), \
        f"music turn not expanded: {music_turn_content!r}"
    assert "track_name: hotel california" in music_turn_content
    assert "artist_name: eagles" in music_turn_content


def test_iter_conversation_turns_keeps_raw_ids_when_no_metadata_provided():
    """Back-compat: callers (e.g. existing dev-eval cells) that don't pass
    metadata_dict get the historical behavior (raw track_id in history)."""
    from scripts.build_bi_encoder_training_data import _iter_conversation_turns

    sessions = [{
        "session_id": "s1", "user_id": "u1",
        "user_profile": None, "conversation_goal": None,
        "conversations": [
            {"role": "user",  "content": "play rock"},
            {"role": "music", "content": "tk_A"},
            {"role": "user",  "content": "more drums"},
            {"role": "music", "content": "tk_B"},
        ],
    }]
    rows = _iter_conversation_turns(sessions)
    assert len(rows) == 2
    # No metadata → raw IDs stay (legacy behavior).
    assert rows[1]["chat_history"][1]["content"] == "tk_A"


def test_iter_conversation_turns_expansion_falls_back_per_id():
    """A single unknown track_id falls back to the raw ID without breaking
    the rest of the row."""
    from scripts.build_bi_encoder_training_data import _iter_conversation_turns

    sessions = [{
        "session_id": "s1", "user_id": "u1",
        "user_profile": None, "conversation_goal": None,
        "conversations": [
            {"role": "user",  "content": "q1"},
            {"role": "music", "content": "tk_KNOWN"},
            {"role": "user",  "content": "q2"},
            {"role": "music", "content": "tk_UNKNOWN_TO_CATALOG"},
            {"role": "user",  "content": "q3"},
            {"role": "music", "content": "tk_KNOWN2"},
        ],
    }]
    metadata_dict = {
        "tk_KNOWN": {"track_id": "tk_KNOWN", "track_name": ["A"], "artist_name": ["x"], "album_name": ["m"]},
        "tk_KNOWN2": {"track_id": "tk_KNOWN2", "track_name": ["B"], "artist_name": ["y"], "album_name": ["n"]},
    }
    rows = _iter_conversation_turns(
        sessions, metadata_dict=metadata_dict,
        corpus_types=["track_name", "artist_name", "album_name"],
    )
    # Row 2 has all three music turns in history (gold = tk_KNOWN2).
    hist = rows[2]["chat_history"]
    # Known IDs expanded.
    assert hist[1]["content"].startswith("track_id: tk_KNOWN,")
    # Unknown ID kept as-is.
    assert hist[3]["content"] == "tk_UNKNOWN_TO_CATALOG"


def test_build_triples_for_row_query_is_non_trivial():
    """Regression: ensure the query field carries the user's actual content.

    Previously the script silently produced empty queries because the row dict
    didn't have the expected component keys. This test asserts the contract:
    when components are present, the query string carries them.
    """
    from scripts.build_bi_encoder_training_data import build_triples_for_row
    row = {
        "chat_history": [{"role": "user", "content": "I like 70s rock"}],
        "current_user_query": "play me something upbeat",
        "user_profile_raw": {"age": 30},
        "conversation_goal": {"listener_goal": "discover"},
        "track_id": "t_gold",
    }
    triple = build_triples_for_row(
        row, "t_gold", ["t_neg1"], {"t_gold": "G", "t_neg1": "N1"},
    )
    # The actual user content MUST appear in the query.
    assert "play me something upbeat" in triple["query"], \
        f"query is missing the current user content: {triple['query']!r}"
    assert len(triple["query"]) > 20, \
        f"query is suspiciously short (possible schema bug): {triple['query']!r}"
