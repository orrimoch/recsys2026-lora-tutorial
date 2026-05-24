"""Tests for cross-encoder triple-builder."""
import pytest


def test_build_ce_triple_shapes():
    """Cross-encoder triple has query, pos list (len 1), neg list (len 7)."""
    from scripts.build_cross_encoder_training_data import build_ce_triple
    triple = build_ce_triple(
        query="play me jazz",
        gold_track_text="track_name: So What | ...",
        neg_track_texts=["track" + str(i) for i in range(7)],
    )
    assert triple["query"] == "play me jazz"
    assert triple["pos"] == ["track_name: So What | ..."]
    assert len(triple["neg"]) == 7


def test_build_ce_triple_preserves_neg_order():
    """Neg list order is preserved (matters for downstream pair construction)."""
    from scripts.build_cross_encoder_training_data import build_ce_triple
    triple = build_ce_triple(query="q", gold_track_text="g", neg_track_texts=["n1", "n2", "n3"])
    assert triple["neg"] == ["n1", "n2", "n3"]


# ---------------------------------------------------------------------------
# Phase 6a: multi-modal, multi-positive Stage B triples.
# select_positives_and_negatives + build_mm_ce_triple are the pure functions
# under test (no GPU / no model download). Fixtures are tiny synthetic maps in
# the style of tests/test_build_bi_encoder_training_data.py.
# ---------------------------------------------------------------------------


def _mm_fixture():
    """Synthetic catalog: gold + 4 Stage-A-retrieved candidates."""
    track_text_map = {
        "t_gold": "track_id: t_gold, track_name: gold",
        "t_c1": "track_id: t_c1, track_name: cand1",   # near-gold (promoted)
        "t_c2": "track_id: t_c2, track_name: cand2",   # below threshold (neg)
        "t_c3": "track_id: t_c3, track_name: cand3",   # below threshold (neg)
        "t_c4": "track_id: t_c4, track_name: cand4",   # below threshold (neg)
    }
    metadata_dict = {
        "t_gold": {"track_id": "t_gold", "tag_list": ["jazz", "smooth"]},
        "t_c1": {"track_id": "t_c1", "tag_list": ["jazz"]},
        "t_c2": {"track_id": "t_c2", "tag_list": ["rock"]},
        "t_c3": {"track_id": "t_c3", "tag_list": []},
        "t_c4": {"track_id": "t_c4", "tag_list": ["pop"]},
    }
    tag_vocab = {"jazz": 1, "smooth": 2, "rock": 3, "pop": 4}
    release_year_lookup = {"t_gold": 2001, "t_c1": 1999, "t_c2": 2010, "t_c4": 2020}
    return track_text_map, metadata_dict, tag_vocab, release_year_lookup


def test_multipositive_promotion_promotes_near_gold_candidates():
    """Any candidate scored >= threshold * gold_score becomes a positive.

    gold_score = 10.0, threshold = 0.85 -> cutoff = 8.5.
    t_c1 (9.0) >= 8.5 -> promoted positive.
    t_c2 (5.0), t_c3 (2.0), t_c4 (1.0) -> negatives.
    """
    from scripts.build_cross_encoder_training_data import select_positives_and_negatives
    teacher = {
        "pos_score": 10.0,
        "neg_tid_to_score": {"t_c1": 9.0, "t_c2": 5.0, "t_c3": 2.0, "t_c4": 1.0},
    }
    sel = select_positives_and_negatives(
        gold_tid="t_gold",
        candidate_tids=["t_c1", "t_c2", "t_c3", "t_c4"],
        teacher=teacher,
        multipositive_threshold=0.85,
    )
    assert sel.pos_tids == ["t_gold", "t_c1"]
    assert sel.neg_tids == ["t_c2", "t_c3", "t_c4"]
    # Teacher scores carried for the chosen ids.
    assert sel.pos_teacher_scores == [10.0, 9.0]
    assert sel.neg_teacher_scores == [5.0, 2.0, 1.0]


def test_multipositive_promotion_no_promotions_when_below_threshold():
    """When no candidate clears the cutoff, only the gold is positive and ALL
    candidates become negatives."""
    from scripts.build_cross_encoder_training_data import select_positives_and_negatives
    teacher = {
        "pos_score": 10.0,
        "neg_tid_to_score": {"t_c1": 4.0, "t_c2": 5.0, "t_c3": 2.0, "t_c4": 1.0},
    }
    sel = select_positives_and_negatives(
        gold_tid="t_gold",
        candidate_tids=["t_c1", "t_c2", "t_c3", "t_c4"],
        teacher=teacher,
        multipositive_threshold=0.85,
    )
    assert sel.pos_tids == ["t_gold"]
    assert sel.neg_tids == ["t_c1", "t_c2", "t_c3", "t_c4"]


def test_negatives_exclude_any_positive_tid():
    """Negatives must NOT contain the gold or any promoted positive (the gold
    can appear in the Stage A top-100 candidate pool — it must be filtered out)."""
    from scripts.build_cross_encoder_training_data import select_positives_and_negatives
    teacher = {
        "pos_score": 10.0,
        "neg_tid_to_score": {"t_gold": 10.0, "t_c1": 9.5, "t_c2": 3.0},
    }
    # gold appears IN the candidate pool (as Stage A retrieved it at rank 0).
    sel = select_positives_and_negatives(
        gold_tid="t_gold",
        candidate_tids=["t_gold", "t_c1", "t_c2"],
        teacher=teacher,
        multipositive_threshold=0.85,
    )
    assert "t_gold" not in sel.neg_tids, "gold leaked into negatives"
    assert "t_c1" not in sel.neg_tids, "promoted positive leaked into negatives"
    assert set(sel.neg_tids).isdisjoint(set(sel.pos_tids))
    assert sel.neg_tids == ["t_c2"]


def test_multipositive_promotion_without_teacher_is_single_positive():
    """No teacher entry -> single-positive (gold only), all candidates negative."""
    from scripts.build_cross_encoder_training_data import select_positives_and_negatives
    sel = select_positives_and_negatives(
        gold_tid="t_gold",
        candidate_tids=["t_c1", "t_c2"],
        teacher=None,
        multipositive_threshold=0.85,
    )
    assert sel.pos_tids == ["t_gold"]
    assert sel.neg_tids == ["t_c1", "t_c2"]
    assert sel.pos_teacher_scores is None
    assert sel.neg_teacher_scores is None


def test_build_mm_ce_triple_has_all_expected_schema_fields():
    """An emitted multi-modal triple carries EXACTLY the Stage A schema fields
    (mirrors build_bi_encoder_training_data.py) plus the Stage B teacher-score
    fields, and uses track-IDs + modality IDs only (no inlined CLAP/CF tensors)."""
    from scripts.build_cross_encoder_training_data import (
        select_positives_and_negatives, build_mm_ce_triple,
    )
    track_text_map, metadata_dict, tag_vocab, release_year_lookup = _mm_fixture()
    teacher = {
        "pos_score": 10.0,
        "neg_tid_to_score": {"t_c1": 9.0, "t_c2": 5.0, "t_c3": 2.0, "t_c4": 1.0},
    }
    sel = select_positives_and_negatives(
        gold_tid="t_gold",
        candidate_tids=["t_c1", "t_c2", "t_c3", "t_c4"],
        teacher=teacher,
        multipositive_threshold=0.85,
    )
    row = {"user_id": "u1", "session_id": "s1"}
    triple = build_mm_ce_triple(
        query="play smooth jazz",
        selection=sel,
        track_text_map=track_text_map,
        row=row,
        metadata_dict=metadata_dict,
        tag_vocab=tag_vocab,
        release_year_lookup=release_year_lookup,
        max_tags=20,
    )
    expected = {
        "query", "pos", "neg", "pos_tid", "pos_tids", "neg_tids",
        "user_id", "session_id",
        "tag_ids_pos", "tag_ids_neg", "release_year_pos", "release_year_neg",
        "pos_teacher_score", "pos_teacher_scores", "neg_teacher_scores",
    }
    assert set(triple.keys()) == expected, (
        f"schema mismatch: missing={expected - set(triple.keys())}, "
        f"extra={set(triple.keys()) - expected}"
    )
    # No inlined dense modality tensors (dataloader joins memmapped arrays).
    for forbidden in ("pos_clap", "neg_clap", "pos_cf_track", "neg_cf_track", "user_cf"):
        assert forbidden not in triple, f"{forbidden} must NOT be inlined"


def test_build_mm_ce_triple_field_alignment_and_values():
    """pos/neg text + parallel modality lists are index-aligned with the
    chosen positive / negative track-IDs; pos_tid is the gold (the teacher
    key), pos_tids is the full multi-positive set."""
    from scripts.build_cross_encoder_training_data import (
        select_positives_and_negatives, build_mm_ce_triple,
    )
    track_text_map, metadata_dict, tag_vocab, release_year_lookup = _mm_fixture()
    teacher = {
        "pos_score": 10.0,
        "neg_tid_to_score": {"t_c1": 9.0, "t_c2": 5.0, "t_c3": 2.0, "t_c4": 1.0},
    }
    sel = select_positives_and_negatives(
        gold_tid="t_gold",
        candidate_tids=["t_c1", "t_c2", "t_c3", "t_c4"],
        teacher=teacher,
        multipositive_threshold=0.85,
    )
    triple = build_mm_ce_triple(
        query="play smooth jazz",
        selection=sel,
        track_text_map=track_text_map,
        row={"user_id": "u1", "session_id": "s1"},
        metadata_dict=metadata_dict,
        tag_vocab=tag_vocab,
        release_year_lookup=release_year_lookup,
        max_tags=20,
    )
    # pos_tid is the gold (key into teacher_scores.parquet); pos_tids is the
    # full promoted set in order (gold first).
    assert triple["pos_tid"] == "t_gold"
    assert triple["pos_tids"] == ["t_gold", "t_c1"]
    # pos / neg text index-aligned with the selected tids.
    assert triple["pos"] == [track_text_map["t_gold"], track_text_map["t_c1"]]
    assert triple["neg_tids"] == ["t_c2", "t_c3", "t_c4"]
    assert triple["neg"] == [track_text_map["t_c2"], track_text_map["t_c3"], track_text_map["t_c4"]]
    # Modality lists parallel to pos / neg.
    assert len(triple["tag_ids_pos"]) == 2
    assert len(triple["tag_ids_neg"]) == 3
    assert len(triple["release_year_pos"]) == 2
    assert len(triple["release_year_neg"]) == 3
    # Tag IDs derived from metadata via tag_vocab (gold -> jazz=1, smooth=2).
    assert triple["tag_ids_pos"][0] == [1, 2]
    # Release years from lookup; missing -> -1 (t_c3 absent from lookup).
    assert triple["release_year_pos"] == [2001, 1999]
    assert triple["release_year_neg"] == [2010, -1, 2020]
    # Teacher score fields.
    assert triple["pos_teacher_score"] == 10.0
    assert triple["pos_teacher_scores"] == [10.0, 9.0]
    assert triple["neg_teacher_scores"] == [5.0, 2.0, 1.0]


def test_build_mm_ce_triple_drops_candidate_not_in_text_map():
    """Catalog drift: a candidate tid absent from track_text_map is dropped
    from BOTH neg + neg_tids so their lengths stay aligned (mirrors Stage A)."""
    from scripts.build_cross_encoder_training_data import (
        select_positives_and_negatives, build_mm_ce_triple,
    )
    track_text_map, metadata_dict, tag_vocab, release_year_lookup = _mm_fixture()
    # t_missing is selected as a negative but absent from the text map.
    teacher = {
        "pos_score": 10.0,
        "neg_tid_to_score": {"t_c2": 5.0, "t_missing": 4.0},
    }
    sel = select_positives_and_negatives(
        gold_tid="t_gold",
        candidate_tids=["t_c2", "t_missing"],
        teacher=teacher,
        multipositive_threshold=0.85,
    )
    triple = build_mm_ce_triple(
        query="q",
        selection=sel,
        track_text_map=track_text_map,
        row={"user_id": "u1", "session_id": "s1"},
        metadata_dict=metadata_dict,
        tag_vocab=tag_vocab,
        release_year_lookup=release_year_lookup,
        max_tags=20,
    )
    assert triple["neg_tids"] == ["t_c2"]
    assert len(triple["neg"]) == 1
    assert len(triple["tag_ids_neg"]) == 1
    assert len(triple["release_year_neg"]) == 1
    assert len(triple["neg_teacher_scores"]) == 1


def test_builder_uses_multimodal_stage_a_retriever_not_text_only_bge_m3():
    """Candidate generation must use the trained multi-modal Stage A retriever
    (DENSE_MULTIMODAL_LOCAL), NOT the old text-only bge-m3 SentenceTransformer.
    Source-level guard to prevent regression to the text-only pool."""
    import inspect
    from scripts import build_cross_encoder_training_data as mod
    src = inspect.getsource(mod)
    assert "DENSE_MULTIMODAL_LOCAL" in src, \
        "candidate generation must use the multi-modal Stage A retriever"
    main_src = inspect.getsource(mod.main)
    # The text-only SentenceTransformer catalog-encode path must be gone.
    assert "SentenceTransformer(" not in main_src, \
        "main() must not encode the catalog with a text-only SentenceTransformer"


def test_builder_exposes_stage_a_and_teacher_cli_flags():
    """CLI must expose the Stage A artifact + teacher-score knobs consistent
    with nb 70/71 CONFIG."""
    import inspect
    from scripts import build_cross_encoder_training_data as mod
    main_src = inspect.getsource(mod.main)
    for flag in (
        "--stage-a-hub-repo", "--multimodal-artifacts",
        "--teacher-scores-path", "--multipositive-threshold", "--pool-size",
    ):
        assert flag in main_src, f"missing CLI flag: {flag}"


def test_load_teacher_scores_missing_or_empty_path_returns_empty(tmp_path):
    """A missing or empty teacher-scores path must skip gracefully (return {}),
    NOT raise. v1 is single-positive and the parquet is dormant/absent; crashing
    after the full ~121K-row walk (nb 71 cell 3) on a missing file is the bug we
    are fixing. An empty {} maps to single-positive downstream (every key misses)."""
    from scripts.build_cross_encoder_training_data import _load_teacher_scores
    assert _load_teacher_scores("") == {}
    assert _load_teacher_scores(str(tmp_path / "does_not_exist.parquet")) == {}
