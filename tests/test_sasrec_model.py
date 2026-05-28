import torch
from mcrs.retrieval_modules.sasrec_model import build_session_examples, ItemFusion


def test_build_session_examples_includes_empty_prefix_first_turn():
    ex = build_session_examples([[10, 11, 12]], max_len=50)
    assert ex == [([], 10), ([10], 11), ([10, 11], 12)]


def test_build_session_examples_left_truncates_to_max_len():
    ex = build_session_examples([[1, 2, 3, 4]], max_len=2)
    assert ([2, 3], 4) in ex
    assert ([1, 2], 3) in ex
    assert build_session_examples([[7]], max_len=50) == [([], 7)]


def test_item_fusion_projects_to_d_and_is_deterministic():
    fusion = ItemFusion(modality_dims=[16], d=8).eval()
    feats = torch.randn(5, 16)
    out1 = fusion(feats)
    out2 = fusion(feats)
    assert out1.shape == (5, 8)
    assert torch.allclose(out1, out2)


def test_item_fusion_handles_extra_leading_dims():
    fusion = ItemFusion(modality_dims=[16], d=8).eval()
    out = fusion(torch.randn(3, 4, 16))
    assert out.shape == (3, 4, 8)


from mcrs.retrieval_modules.sasrec_model import SasrecModel, next_item_loss


def _tiny_model():
    return SasrecModel(item_modality_dims=[16], ctx_in_dim=12, d=8, n_layers=1,
                       n_heads=2, max_len=5).eval()


def test_encode_returns_session_state_shape():
    m = _tiny_model()
    B, L = 4, 3
    ctx = torch.randn(B, 12)
    items = torch.randn(B, L, 16)
    lengths = torch.tensor([3, 2, 1, 0])  # row 3 = empty prefix (turn 1)
    state = m.encode(ctx, items, lengths)
    assert state.shape == (B, 8)
    assert torch.isfinite(state).all()


def test_score_shape_against_item_matrix():
    m = _tiny_model()
    state = torch.randn(4, 8)
    item_matrix = torch.randn(20, 8)
    logits = m.score(state, item_matrix)
    assert logits.shape == (4, 20)


def test_next_item_loss_decreases_on_overfit_batch():
    torch.manual_seed(0)
    m = SasrecModel(item_modality_dims=[16], ctx_in_dim=12, d=8, n_layers=1, n_heads=2, max_len=5)
    all_item_feats = torch.randn(20, 16)
    ctx = torch.randn(4, 12)
    items = torch.randn(4, 3, 16)
    lengths = torch.tensor([3, 3, 3, 3])
    target = torch.tensor([1, 5, 9, 13])
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    losses = []
    for _ in range(40):
        opt.zero_grad()
        item_matrix = m.item_fusion(all_item_feats)
        loss = next_item_loss(m, ctx, items, lengths, target, item_matrix)
        loss.backward(); opt.step()
        losses.append(float(loss))
    assert losses[-1] < losses[0] - 0.5


from mcrs.retrieval_modules.sasrec_model import build_user_dialog


def test_build_user_dialog_keeps_only_user_turns():
    turns = [{"role": "user", "content": "play something upbeat"},
             {"role": "music", "content": "track-uuid-1"},
             {"role": "user", "content": "more mellow please"}]
    assert build_user_dialog(turns) == "play something upbeat\nmore mellow please"
    assert build_user_dialog([{"role": "music", "content": "x"}]) == ""
