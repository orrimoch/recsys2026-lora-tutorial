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
    fusion = ItemFusion(in_dim=16, d=8).eval()
    feats = torch.randn(5, 16)
    out1 = fusion(feats)
    out2 = fusion(feats)
    assert out1.shape == (5, 8)
    assert torch.allclose(out1, out2)


def test_item_fusion_handles_extra_leading_dims():
    fusion = ItemFusion(in_dim=16, d=8).eval()
    out = fusion(torch.randn(3, 4, 16))
    assert out.shape == (3, 4, 8)
