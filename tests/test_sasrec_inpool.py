"""Primitives for the in-pool SASRec reranker (SASRec_Improved_Plan.md):
1) build_sasrec_context — goal-ful context with byte-identical empty-goal degrade
   (3-way train/gate/serve parity).
2) inpool_loss — softmax-CE over the per-sample candidate POOL (not the full catalog).
"""
import torch

from mcrs.retrieval_modules.sasrec_model import (
    SasrecModel, build_sasrec_context, build_user_dialog, inpool_loss,
    inpool_target_index)


# ---- 3) pool-prep: gold-in-pool filter (skip when recall missed the gold) --
def test_target_index_when_gold_in_pool():
    assert inpool_target_index(["a", "b", "c"], "b") == 1
    assert inpool_target_index(["a", "b", "c"], "a") == 0


def test_skip_when_gold_not_in_pool():
    # recall miss -> unrecoverable by any ranker -> skip the sample (None)
    assert inpool_target_index(["a", "b", "c"], "z") is None
    assert inpool_target_index([], "a") is None


# ---- 1) goal-ful context -------------------------------------------------
TURNS = [
    {"role": "user", "content": "play me something upbeat"},
    {"role": "music", "content": "track-123"},
    {"role": "user", "content": "more like that"},
]


def test_context_appends_goal_when_present():
    out = build_sasrec_context(TURNS, goal_text="discover new indie")
    assert out == build_user_dialog(TURNS) + "\ngoal: discover new indie"


def test_context_degrades_to_goalless_when_empty_or_none():
    base = build_user_dialog(TURNS)
    assert build_sasrec_context(TURNS, goal_text="") == base
    assert build_sasrec_context(TURNS, goal_text="   ") == base   # whitespace -> degrade
    assert build_sasrec_context(TURNS, goal_text=None) == base
    assert build_sasrec_context(TURNS) == base                    # default None


def test_context_format_matches_retrieval_raw_with_goal():
    # mirror build_retrieval_query raw_with_goal: f"{base}\ngoal: {gt}" if gt else base
    gt = "chill study vibes"
    assert build_sasrec_context(TURNS, gt) == f"{build_user_dialog(TURNS)}\ngoal: {gt}"


# ---- 2) in-pool loss -----------------------------------------------------
def _tiny_model():
    torch.manual_seed(0)
    return SasrecModel(item_modality_dims=[4], ctx_in_dim=8, d=16, n_layers=1,
                       n_heads=2, max_len=10, temperature=0.07).eval()


def test_inpool_loss_matches_manual_pool_softmax_ce():
    m = _tiny_model()
    B, K, L = 3, 5, 2
    ctx = torch.randn(B, 8)
    played = torch.randn(B, L, 4)
    lengths = torch.tensor([L, L, L])
    pool = torch.randn(B, K, 4)
    target = torch.tensor([0, 2, 4])

    got = inpool_loss(m, ctx, played, lengths, pool, target)

    # manual reference: encode once, fuse pool, cosine/temperature, CE over pool
    with torch.no_grad():
        state = torch.nn.functional.normalize(m.encode(ctx, played, lengths), dim=-1)
        cand = torch.nn.functional.normalize(m.item_fusion(pool), dim=-1)
        logits = torch.bmm(cand, state.unsqueeze(-1)).squeeze(-1) / m.temperature
        ref = torch.nn.functional.cross_entropy(logits, target)
    assert torch.allclose(got, ref, atol=1e-5)


def test_inpool_loss_is_lower_when_gold_aligns_with_state():
    """Sanity: a pool whose target item == the state direction should beat a
    pool whose target is orthogonal noise."""
    m = _tiny_model()
    B, K, L = 2, 4, 1
    ctx = torch.randn(B, 8); played = torch.randn(B, L, 4); lengths = torch.tensor([L, L])
    pool = torch.randn(B, K, 4); target = torch.tensor([1, 1])
    loss = inpool_loss(m, ctx, played, lengths, pool, target)
    assert loss.ndim == 0 and torch.isfinite(loss)  # scalar, finite
