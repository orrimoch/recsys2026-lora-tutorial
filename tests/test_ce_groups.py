# tests/test_ce_groups.py
from mcrs.contracts import Candidate
from mcrs.training.ce_data import build_ce_training_groups, GP_WEIGHTS

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

def _turn(tn, gp):
    from mcrs.contracts import TurnContext, UserProfile
    return TurnContext("s", "u", tn, ["u"]*tn, "goal", UserProfile("u",None,None,None,[]), [], "cold")

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
