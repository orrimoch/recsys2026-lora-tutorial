"""TDD tests for the sasrec_rank_inv feature in build_lgbm_features.py.

Written BEFORE the implementation — they must fail on the un-patched code.
"""
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Test 1: extract_features emits sasrec_rank_inv when candidate has sasrec_rank
# ---------------------------------------------------------------------------

def _minimal_session_info():
    return {
        "session_id": "sess001",
        "user_id": "u1",
        "turn_number": 1,
        "conversation_goal": {"category": "discovery", "specificity": "vague"},
        "goal_progress_assessments": [],
        "prior_track_count": 0,
        "query_drift_score": 1.0,
    }


def _minimal_track_meta():
    return {
        "t1": {
            "artist_name": "ArtistA",
            "album_name": "AlbumA",
            "tag_list": ["rock"],
            "popularity": 100.0,
            "release_date": "2010-01-01",
        },
        "t2": {
            "artist_name": "ArtistB",
            "album_name": "AlbumB",
            "tag_list": [],
            "popularity": 50.0,
            "release_date": "2015-06-15",
        },
    }


def test_extract_features_emits_sasrec_rank_inv_when_present():
    """When candidate dict carries sasrec_rank=3, row must have sasrec_rank_inv == 1/3."""
    from scripts.build_lgbm_features import extract_features

    candidates = [{"tid": "t1", "wrrf_rank": 1, "sasrec_rank": 3}]
    rows = extract_features(
        query="some query",
        candidates=candidates,
        gold_tid="t1",
        session_info=_minimal_session_info(),
        user_info={"age_group": "20s", "country_code": "US", "gender": "male"},
        track_meta=_minimal_track_meta(),
        cfbpr_tid_to_idx={},
        cfbpr_track_mat=np.zeros((0, 128)),
        cfbpr_user_embs={},
        query_tokens={"some", "query"},
        pop_rank_pct=None,
        played_meta=[],
    )
    assert len(rows) == 1
    row = rows[0]
    assert "sasrec_rank_inv" in row, "sasrec_rank_inv should be present when candidate has sasrec_rank"
    assert abs(row["sasrec_rank_inv"] - 1.0 / 3) < 1e-9, f"expected 1/3, got {row['sasrec_rank_inv']}"


def test_extract_features_omits_sasrec_rank_inv_when_absent():
    """When candidate dict has NO sasrec_rank, row must NOT have sasrec_rank_inv (schema unchanged)."""
    from scripts.build_lgbm_features import extract_features

    candidates = [{"tid": "t2", "wrrf_rank": 2}]
    rows = extract_features(
        query="some query",
        candidates=candidates,
        gold_tid="t1",
        session_info=_minimal_session_info(),
        user_info={"age_group": "20s", "country_code": "US", "gender": "male"},
        track_meta=_minimal_track_meta(),
        cfbpr_tid_to_idx={},
        cfbpr_track_mat=np.zeros((0, 128)),
        cfbpr_user_embs={},
        query_tokens={"some", "query"},
        pop_rank_pct=None,
        played_meta=[],
    )
    assert len(rows) == 1
    row = rows[0]
    assert "sasrec_rank_inv" not in row, "sasrec_rank_inv must be absent when candidate lacks sasrec_rank"


def test_extract_features_sasrec_sentinel_value():
    """Sentinel sasrec_rank=10000 -> sasrec_rank_inv == 1/10000."""
    from scripts.build_lgbm_features import extract_features

    candidates = [{"tid": "t1", "wrrf_rank": 1, "sasrec_rank": 10000}]
    rows = extract_features(
        query="some query",
        candidates=candidates,
        gold_tid="t1",
        session_info=_minimal_session_info(),
        user_info={},
        track_meta=_minimal_track_meta(),
        cfbpr_tid_to_idx={},
        cfbpr_track_mat=np.zeros((0, 128)),
        cfbpr_user_embs={},
        query_tokens=set(),
        pop_rank_pct=None,
        played_meta=[],
    )
    row = rows[0]
    assert "sasrec_rank_inv" in row
    assert abs(row["sasrec_rank_inv"] - 1.0 / 10000) < 1e-12


# ---------------------------------------------------------------------------
# Test 2: WRRFRunner.run sasrec path — no real models loaded
# ---------------------------------------------------------------------------

class _FakeRRFModel:
    """Minimal stand-in for the RRF_MODEL object, so we don't load any files."""

    def __init__(self):
        # Three subs: bm25, dense, sasrec_seq.
        self.subs = [
            {"weight": 1.0, "label": "bm25"},
            {"weight": 1.0, "label": "dense"},
            {"weight": 1.0, "label": "sasrec_seq"},
        ]
        self.k = 60

    def batch_per_sub_rankings(self, queries, user_ids=None, batch_context=None):
        """Return canned rankings for two queries.

        per_sub[s][q] = list of tids ranked by sub s for query q.

        Fused topk (computed by the real fuse_per_sub) must include 't_not_in_sasrec'
        because bm25 ranks it first — but sasrec doesn't rank it at all.
        """
        n = len(queries)
        # bm25 sub ranks: t_not_in_sasrec, t_common
        bm25 = [["t_not_in_sasrec", "t_common"]] * n
        # dense sub ranks: t_common, t_not_in_sasrec
        dense = [["t_common", "t_not_in_sasrec"]] * n
        # sasrec sub ranks only: t_sasrec_top, t_common (NOT t_not_in_sasrec)
        sasrec = [["t_sasrec_top", "t_common"]] * n
        per_sub = [bm25, dense, sasrec]
        labels = ["bm25", "dense", "sasrec_seq"]
        return per_sub, labels


def test_wrrfrunner_run_sasrec_path_correct_ranks():
    """WRRFRunner.run with use_sasrec=True returns sasrec_rank for each candidate.

    * candidates that the sasrec sub ranked → correct 1-indexed sasrec_rank
    * candidates the sasrec sub did NOT rank → sasrec_rank == 10000 (SENTINEL)
    """
    from scripts.build_lgbm_features import WRRFRunner
    from mcrs.retrieval_modules.rrf import RRF_MODEL

    runner = WRRFRunner.__new__(WRRFRunner)
    runner.use_sasrec = True
    runner.wrrf = _FakeRRFModel()

    results = runner.run(
        queries=["query one"],
        topk=10,
        batch_context=[{"history_tids": []}],
        user_ids=["u1"],
    )
    assert len(results) == 1
    cands = results[0]

    tid_to_cand = {c["tid"]: c for c in cands}

    # t_common is in the sasrec ranking at position 2 (0-indexed 1 → 1-indexed 2)
    assert "t_common" in tid_to_cand, "t_common should be fused into candidates"
    assert tid_to_cand["t_common"]["sasrec_rank"] == 2, (
        f"t_common is 2nd in sasrec, expected sasrec_rank=2, got {tid_to_cand['t_common']['sasrec_rank']}"
    )

    # t_not_in_sasrec is ranked by bm25 but NOT by sasrec → sentinel
    assert "t_not_in_sasrec" in tid_to_cand, "t_not_in_sasrec should appear via bm25"
    assert tid_to_cand["t_not_in_sasrec"]["sasrec_rank"] == 10000, (
        f"expected sentinel 10000 for t_not_in_sasrec, got {tid_to_cand['t_not_in_sasrec']['sasrec_rank']}"
    )

    # All returned dicts must also carry wrrf_rank
    for c in cands:
        assert "wrrf_rank" in c, f"wrrf_rank missing from {c}"
        assert "sasrec_rank" in c, f"sasrec_rank missing from {c}"


def test_wrrfrunner_run_sasrec_path_wrrf_rank_is_1indexed():
    """wrrf_rank for the top fused candidate must be 1."""
    from scripts.build_lgbm_features import WRRFRunner

    runner = WRRFRunner.__new__(WRRFRunner)
    runner.use_sasrec = True
    runner.wrrf = _FakeRRFModel()

    results = runner.run(queries=["q"], topk=5, batch_context=None, user_ids=None)
    cands = results[0]
    wrrf_ranks = [c["wrrf_rank"] for c in cands]
    assert wrrf_ranks[0] == 1, f"top candidate wrrf_rank must be 1, got {wrrf_ranks[0]}"
    assert wrrf_ranks == sorted(wrrf_ranks), "wrrf_ranks must be ascending"


def test_wrrfrunner_run_non_sasrec_path_no_sasrec_rank_key():
    """When use_sasrec=False, candidates must NOT have a sasrec_rank key (schema unchanged)."""
    from scripts.build_lgbm_features import WRRFRunner

    class _FakeSimpleRRF:
        def batch_text_to_item_retrieval(self, queries, topk, user_ids=None, batch_context=None):
            return [["t1", "t2"]] * len(queries)

    runner = WRRFRunner.__new__(WRRFRunner)
    runner.use_sasrec = False
    runner.wrrf = _FakeSimpleRRF()

    results = runner.run(queries=["q"], topk=5, batch_context=None, user_ids=None)
    cands = results[0]
    for c in cands:
        assert "sasrec_rank" not in c, f"sasrec_rank must be absent in non-sasrec path, got {c}"


# ---------------------------------------------------------------------------
# n_channels_hit (Lever 2): extract_features emits it when the candidate dict
# carries it; omits it otherwise. (TDD: written before implementation.)
# ---------------------------------------------------------------------------

def test_extract_features_emits_n_channels_hit_when_present():
    from scripts.build_lgbm_features import extract_features

    candidates = [{"tid": "t1", "wrrf_rank": 1, "n_channels_hit": 3}]
    rows = extract_features(
        query="some query",
        candidates=candidates,
        gold_tid="t1",
        session_info=_minimal_session_info(),
        user_info={"age_group": "20s", "country_code": "US", "gender": "male"},
        track_meta=_minimal_track_meta(),
        cfbpr_tid_to_idx={},
        cfbpr_track_mat=np.zeros((0, 128)),
        cfbpr_user_embs={},
        query_tokens={"some", "query"},
        pop_rank_pct=None,
    )
    assert rows[0]["n_channels_hit"] == 3


def test_extract_features_omits_n_channels_hit_when_absent():
    from scripts.build_lgbm_features import extract_features

    candidates = [{"tid": "t1", "wrrf_rank": 1}]
    rows = extract_features(
        query="some query",
        candidates=candidates,
        gold_tid="t1",
        session_info=_minimal_session_info(),
        user_info={"age_group": "20s", "country_code": "US", "gender": "male"},
        track_meta=_minimal_track_meta(),
        cfbpr_tid_to_idx={},
        cfbpr_track_mat=np.zeros((0, 128)),
        cfbpr_user_embs={},
        query_tokens={"some", "query"},
        pop_rank_pct=None,
    )
    assert "n_channels_hit" not in rows[0]


# --- CLAP audio-similarity feature (clap_session_sim) train-side emit ---

def test_extract_features_emits_clap_session_sim_when_lookup_given():
    import numpy as np
    from scripts.build_lgbm_features import extract_features
    clap = {"t1": np.array([1.0, 0.0], dtype=np.float32),
            "p1": np.array([1.0, 0.0], dtype=np.float32)}
    rows = extract_features(
        query="q", candidates=[{"tid": "t1", "wrrf_rank": 1}], gold_tid="t1",
        session_info=_minimal_session_info(),
        user_info={"age_group": "20s", "country_code": "US", "gender": "male"},
        track_meta=_minimal_track_meta(),
        cfbpr_tid_to_idx={}, cfbpr_track_mat=np.zeros((0, 128)), cfbpr_user_embs={},
        query_tokens={"q"}, pop_rank_pct=None,
        clap_lookup=clap, played_tids=["p1"],
    )
    assert abs(rows[0]["clap_session_sim"] - 1.0) < 1e-6


def test_extract_features_omits_clap_when_no_lookup():
    import numpy as np
    from scripts.build_lgbm_features import extract_features
    rows = extract_features(
        query="q", candidates=[{"tid": "t1", "wrrf_rank": 1}], gold_tid="t1",
        session_info=_minimal_session_info(),
        user_info={"age_group": "20s", "country_code": "US", "gender": "male"},
        track_meta=_minimal_track_meta(),
        cfbpr_tid_to_idx={}, cfbpr_track_mat=np.zeros((0, 128)), cfbpr_user_embs={},
        query_tokens={"q"}, pop_rank_pct=None,
    )
    assert "clap_session_sim" not in rows[0]
