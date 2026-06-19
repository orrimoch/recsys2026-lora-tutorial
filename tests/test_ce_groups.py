# tests/test_ce_groups.py
from mcrs.contracts import Candidate
from mcrs.training.ce_data import build_ce_training_groups, build_doc, GP_WEIGHTS, _stable_seed


def test_stable_seed_is_process_stable_and_distinct():
    # md5-based -> a fixed constant regardless of PYTHONHASHSEED (cross-process reproducible OOF).
    assert _stable_seed(0, "s", 1) == 554319870
    assert _stable_seed(0, "s", 2) != _stable_seed(0, "s", 1)   # distinct per turn
    assert _stable_seed(1, "s", 1) != _stable_seed(0, "s", 1)   # distinct per pipeline seed

class FakeQB:
    def build(self, ctx):
        from mcrs.contracts import Query
        return Query(text=f"q{ctx.turn_number}")

class FakeFusion:
    def __init__(self, pools): self.pools = pools   # list[list[Candidate]] aligned to turns
    def fuse(self, queries, topk, **kw): return self.pools

class FakeCat:
    def __init__(self):
        self._meta = {t: {"artist_name": f"a{t}", "track_name": f"n{t}"} for t in ["g","x","y","z","w"]}
        self.enr = set(self._meta)
    def is_enriched(self, t): return t in self.enr
    def id_to_metadata(self, t, enriched=False): return f"doc-{t}"
    def metadata(self, t): return self._meta[t]

def _turn(tn, gp=None, session_id="s"):
    from mcrs.contracts import TurnContext, UserProfile
    return TurnContext(session_id, "u", tn, ["u"]*tn, "goal", UserProfile("u",None,None,None,[]), [], "cold")

def test_gold_in_pool_only_and_group_shape():
    cat = FakeCat()
    pool = [Candidate(track_id=t, rrf_score=1.0/i, channel_ranks={"c": i})
            for i, t in enumerate(["g","x","y","z","w"], start=1)]
    fusion = FakeFusion([pool, [Candidate("x"), Candidate("y")]])  # turn2 pool lacks gold 'g'
    turns = [_turn(1, "MOVES_TOWARD_GOAL"), _turn(2, "MOVES_TOWARD_GOAL")]
    gold_fn = lambda t: "g"
    gp_fn = lambda t: "MOVES_TOWARD_GOAL"
    groups = build_ce_training_groups(FakeQB(), fusion, turns, gold_fn, catalog=cat,
                                      cross_encoder_k=5, n_negatives=3, k_min=1, seed=0, gp_fn=gp_fn)
    assert len(groups) == 1                         # turn2 dropped (gold not in pool)
    q, docs, gw = groups[0]
    assert q == "q1"
    assert docs[0] == "doc-g"                       # positive at index 0
    assert 1 <= len(docs) - 1 <= 3                  # negatives
    assert gw == GP_WEIGHTS["MOVES_TOWARD_GOAL"]

def test_goal_progress_group_weight():
    cat = FakeCat()
    pool = [Candidate(track_id=t, channel_ranks={"c": i}) for i, t in enumerate(["g","x","y","z","w"],1)]
    fusion = FakeFusion([pool])
    groups = build_ce_training_groups(FakeQB(), fusion, [_turn(1,"x")], lambda t:"g", catalog=cat,
                 cross_encoder_k=5, n_negatives=2, k_min=1, seed=0,
                 gp_fn=lambda t: "DOES_NOT_MOVE_TOWARD_GOAL")
    assert groups[0][2] == GP_WEIGHTS["DOES_NOT_MOVE_TOWARD_GOAL"]


# ---------------------------------------------------------------------------
# Hardening tests
# ---------------------------------------------------------------------------

def _pool_with_gold(gold_id="g", other_ids=("x","y","z","w")):
    """Make a pool where gold_id is at rank 1, others follow."""
    ids = [gold_id] + list(other_ids)
    return [Candidate(track_id=t, channel_ranks={"c": i}) for i, t in enumerate(ids, 1)]


def test_gold_absent_from_pool_increments_dropped_no_gold():
    """A turn whose gold is NOT in the pool must be dropped and counted in dropped_no_gold."""
    cat = FakeCat()
    pool_no_gold = [Candidate(track_id=t, channel_ranks={"c": i}) for i, t in enumerate(["x","y","z","w","v"],1)]
    # 'g' is the gold but is not in pool_no_gold
    # We need FakeCat to also have 'v'
    cat._meta["v"] = {"artist_name": "av", "track_name": "nv"}
    cat.enr.add("v")
    report = {}
    groups = build_ce_training_groups(FakeQB(), FakeFusion([pool_no_gold]), [_turn(1)],
                                      lambda t: "g", catalog=cat,
                                      cross_encoder_k=5, n_negatives=3, k_min=1, seed=0,
                                      report=report)
    assert len(groups) == 0
    assert report["dropped_no_gold"] == 1
    assert report["dropped_few_neg"] == 0


def test_too_few_negs_after_denoise_increments_dropped_few_neg():
    """A turn left with < k_min negatives after near-dup filtering must be dropped and counted."""
    # Build a catalog where gold title is 'song', and all other tracks have near-dup titles
    from mcrs.contracts import TurnContext, UserProfile
    # Use distinct track_ids not in the standard FakeCat
    cat = FakeCat()
    # Override metadata: gold track_name='song', others all near-dup 'song (live)', 'song (remastered)'
    for t, name in [("g", "song"), ("x", "song (live)"), ("y", "song (remastered)"),
                    ("z", "song (acoustic)"), ("w", "song (2023)")]:
        cat._meta[t] = {"artist_name": f"art_{t}", "track_name": name}
    pool = _pool_with_gold("g", ["x","y","z","w"])
    report = {}
    groups = build_ce_training_groups(FakeQB(), FakeFusion([pool]), [_turn(1)],
                                      lambda t: "g", catalog=cat,
                                      cross_encoder_k=5, n_negatives=5, k_min=2, seed=0,
                                      report=report)
    assert len(groups) == 0
    assert report["dropped_few_neg"] == 1
    assert report["dropped_no_gold"] == 0


def test_gp_fn_none_gives_weight_1():
    """gp_fn=None (no goal-progress function) must produce group weight 1.0."""
    cat = FakeCat()
    pool = _pool_with_gold()
    groups = build_ce_training_groups(FakeQB(), FakeFusion([pool]), [_turn(1)],
                                      lambda t: "g", catalog=cat,
                                      cross_encoder_k=5, n_negatives=3, k_min=1, seed=0,
                                      gp_fn=None)
    assert groups[0][2] == 1.0


def test_gp_fn_returning_none_gives_weight_1():
    """gp_fn that returns None for the turn must map to weight 1.0 (None key in GP_WEIGHTS)."""
    cat = FakeCat()
    pool = _pool_with_gold()
    groups = build_ce_training_groups(FakeQB(), FakeFusion([pool]), [_turn(1)],
                                      lambda t: "g", catalog=cat,
                                      cross_encoder_k=5, n_negatives=3, k_min=1, seed=0,
                                      gp_fn=lambda t: None)
    assert groups[0][2] == 1.0


def test_gp_fn_does_not_move_uses_w_low():
    """gp_fn returning DOES_NOT_MOVE_TOWARD_GOAL must apply the w_low parameter, not the constant."""
    cat = FakeCat()
    pool = _pool_with_gold()
    groups = build_ce_training_groups(FakeQB(), FakeFusion([pool]), [_turn(1)],
                                      lambda t: "g", catalog=cat,
                                      cross_encoder_k=5, n_negatives=3, k_min=1, seed=0,
                                      gp_fn=lambda t: "DOES_NOT_MOVE_TOWARD_GOAL",
                                      w_low=0.15)
    # w_low=0.15 overrides the GP_WEIGHTS constant 0.3 at call time
    assert groups[0][2] == 0.15


def test_positive_doc_at_index_0_equals_build_doc_gold():
    """The positive document (index 0 in docs) must equal build_doc(catalog, gold)."""
    from mcrs.data.catalog import Catalog
    row = {"track_id": "g", "track_name": "Holocene", "artist_name": "Bon Iver",
           "album_name": "Bon Iver", "release_date": "2011", "tag_list": []}
    extra_rows = [{"track_id": t, "track_name": t, "artist_name": t,
                   "album_name": t, "release_date": "2011", "tag_list": []}
                  for t in ["x","y","z","w"]]
    enriched = {t: f"enriched-doc-{t}" for t in ["g","x","y","z","w"]}
    cat = Catalog([row] + extra_rows, enriched_docs=enriched)
    pool = _pool_with_gold()

    class RealCatQB:
        def build(self, ctx):
            from mcrs.contracts import Query
            return Query(text="query")

    groups = build_ce_training_groups(RealCatQB(), FakeFusion([pool]), [_turn(1)],
                                      lambda t: "g", catalog=cat,
                                      cross_encoder_k=5, n_negatives=3, k_min=1, seed=0)
    q, docs, gw = groups[0]
    assert docs[0] == build_doc(cat, "g")        # positive at index 0 is build_doc(gold)
    assert docs[0] == "enriched-doc-g"[:2000]    # char-capped enriched doc


def test_report_keeps_count_is_number_of_groups():
    """report['kept'] must equal the number of returned groups."""
    cat = FakeCat()
    pool = _pool_with_gold()
    pool_no_gold = [Candidate(track_id=t, channel_ranks={"c": i}) for i, t in enumerate(["x","y","z","w"],1)]
    report = {}
    groups = build_ce_training_groups(FakeQB(), FakeFusion([pool, pool_no_gold]), [_turn(1), _turn(2)],
                                      lambda t: "g", catalog=cat,
                                      cross_encoder_k=5, n_negatives=3, k_min=1, seed=0,
                                      report=report)
    assert report["kept"] == len(groups)


def test_report_exposes_kept_keys_aligned_to_groups():
    # kept_keys must align 1:1 with groups (used to map fold ids -> groups), even when a turn is dropped.
    cat = FakeCat()
    pool = [Candidate(track_id=t, rrf_score=1.0/i, channel_ranks={"c": i})
            for i, t in enumerate(["g", "x", "y", "z", "w"], start=1)]
    no_gold = [Candidate("x"), Candidate("y")]                       # turn 2 gold not in pool -> dropped
    fusion = FakeFusion([pool, no_gold])
    turns = [_turn(1, session_id="sA"), _turn(2, session_id="sB")]
    rep = {}
    groups = build_ce_training_groups(FakeQB(), fusion, turns, lambda t: "g", catalog=cat,
                                      cross_encoder_k=5, n_negatives=2, k_min=1, seed=0, report=rep)
    assert rep["kept_keys"] == [("sA", 1)]                           # only the kept turn, in order
    assert len(rep["kept_keys"]) == len(groups)


def test_fusion_query_builder_separate_from_ce_query():
    # The CE query (query_builder) and the retrieval/fusion query (fusion_query_builder) can differ;
    # the group's query_text is the CE one. FakeFusion ignores the text, so we just assert the CE query wins.
    cat = FakeCat()
    pool = [Candidate(track_id=t, channel_ranks={"c": i}) for i, t in enumerate(["g", "x", "y"], 1)]
    class PlainQB:
        def build(self, ctx):
            from mcrs.contracts import Query
            return Query(text="PLAIN")
    groups = build_ce_training_groups(FakeQB(), FakeFusion([pool]), [_turn(1)], lambda t: "g", catalog=cat,
                                      cross_encoder_k=5, n_negatives=1, k_min=1, seed=0,
                                      fusion_query_builder=PlainQB())
    assert groups[0][0] == "q1"                                      # CE query (FakeQB), not "PLAIN"


def test_fusion_chunking_matches_single_call():
    # Chunked fusion (for progress + bounded memory) must yield identical groups to one batched call.
    cat = FakeCat()
    pool_tpl = [Candidate(track_id=t, channel_ranks={"c": i}) for i, t in enumerate(["g", "x", "y", "z", "w"], 1)]

    class ChunkAwareFusion:                                          # one pool per query (honors chunk slices)
        def fuse(self, queries, topk, **kw):
            return [list(pool_tpl) for _ in queries]

    turns = [_turn(1, session_id="s1"), _turn(2, session_id="s2"), _turn(3, session_id="s3")]
    kw = dict(catalog=cat, cross_encoder_k=5, n_negatives=2, k_min=1, seed=0)
    g_single = build_ce_training_groups(FakeQB(), ChunkAwareFusion(), turns, lambda t: "g", **kw)
    g_chunked = build_ce_training_groups(FakeQB(), ChunkAwareFusion(), turns, lambda t: "g", fusion_chunk=2, **kw)
    assert len(g_single) == len(g_chunked) == 3
    assert [docs for _, docs, _ in g_single] == [docs for _, docs, _ in g_chunked]   # identical, chunk-invariant


def test_list_valued_metadata_is_coerced_not_crashed():
    # Real Track-Metadata has LIST-valued track_name/artist_name; they must be coerced to str for
    # the denoise helpers (normalize_title/.lower()), not passed through raw (regression: AttributeError).
    cat = FakeCat()
    cat._meta = {t: {"artist_name": [f"a{t}"], "track_name": [f"n{t}", "(alt)"]}
                 for t in ["g", "x", "y", "z", "w"]}
    pool = [Candidate(track_id=t, channel_ranks={"c": i}) for i, t in enumerate(["g", "x", "y", "z", "w"], 1)]
    groups = build_ce_training_groups(FakeQB(), FakeFusion([pool]), [_turn(1)], lambda t: "g", catalog=cat,
                                      cross_encoder_k=5, n_negatives=2, k_min=1, seed=0)
    assert len(groups) == 1                                          # built, no crash on list fields


def test_teacher_score_fn_drops_false_negative_from_group():
    # T1.3 wiring: a frozen-CE teacher scores 'x' (rank-1 negative) as highly relevant -> it must be
    # filtered out of the group's negatives; with the lever off (default) it stays.
    cat = FakeCat()
    pool = [Candidate(track_id=t, channel_ranks={"c": i}) for i, t in enumerate(["g","x","y","z","w"], 1)]
    turns = [_turn(1)]
    teacher = lambda q, tids: [9.0 if t == "x" else 0.0 for t in tids]   # 'x' looks like a positive

    on = build_ce_training_groups(FakeQB(), FakeFusion([pool]), turns, lambda t: "g", catalog=cat,
                                  cross_encoder_k=5, n_negatives=4, k_min=1, seed=0,
                                  teacher_score_fn=teacher, fp_quantile=0.25)
    off = build_ce_training_groups(FakeQB(), FakeFusion([pool]), turns, lambda t: "g", catalog=cat,
                                   cross_encoder_k=5, n_negatives=4, k_min=1, seed=0)
    assert "doc-x" not in on[0][1]      # teacher false-negative dropped
    assert "doc-x" in off[0][1]         # default keeps it -> the filter (not sampling) removed it
