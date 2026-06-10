import numpy as np
import pytest
import torch
from mcrs.retrieval_modules.sasrec_model import (
    build_session_examples, ItemFusion, apply_item_feats_mode)


def test_feats_mode_content_is_identity():
    feats = np.arange(12, dtype=np.float32).reshape(2, 6)
    out, dims = apply_item_feats_mode(feats, [4, 2], mode="content")
    assert np.array_equal(out, feats)
    assert dims == [4, 2]


def test_feats_mode_metadata_keeps_first_modality():
    feats = np.arange(12, dtype=np.float32).reshape(2, 6)
    out, dims = apply_item_feats_mode(feats, [4, 2], mode="metadata")
    assert out.shape == (2, 4)
    assert np.array_equal(out, feats[:, :4])
    assert dims == [4]


def test_feats_mode_audio_keeps_trailing_modality():
    feats = np.arange(12, dtype=np.float32).reshape(2, 6)
    out, dims = apply_item_feats_mode(feats, [4, 2], mode="audio")
    assert out.shape == (2, 2)
    assert np.array_equal(out, feats[:, 4:])
    assert dims == [2]


def test_feats_mode_random_is_id_proxy_same_shape_deterministic():
    feats = np.zeros((5, 6), dtype=np.float32)
    a, dims_a = apply_item_feats_mode(feats, [4, 2], mode="random", seed=42)
    b, _ = apply_item_feats_mode(feats, [4, 2], mode="random", seed=42)
    c, _ = apply_item_feats_mode(feats, [4, 2], mode="random", seed=7)
    assert a.shape == feats.shape          # architecture identical to content
    assert dims_a == [4, 2]                 # modality dims preserved
    assert np.array_equal(a, b)             # deterministic given the seed
    assert not np.array_equal(a, c)         # different seed -> different vectors
    assert not np.allclose(a, 0.0)          # not the (zeroed) content
    assert not np.array_equal(a[0], a[1])   # unique per item (a per-item id)


def test_feats_mode_invalid_raises():
    feats = np.zeros((2, 6), dtype=np.float32)
    with pytest.raises(ValueError):
        apply_item_feats_mode(feats, [4, 2], mode="bogus")


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
        # Pin label_smoothing=0 so the smoothing floor doesn't fight the
        # overfit signal this test measures.
        loss = next_item_loss(m, ctx, items, lengths, target, item_matrix,
                              label_smoothing=0.0)
        loss.backward(); opt.step()
        losses.append(float(loss))
    assert losses[-1] < losses[0] - 0.5


def test_next_item_loss_label_smoothing_changes_value():
    """label_smoothing=0 vs 0.05 must produce meaningfully different losses on
    the same forward pass — confirms the kwarg actually wires through."""
    torch.manual_seed(0)
    m = SasrecModel(item_modality_dims=[16], ctx_in_dim=12, d=8, n_layers=1,
                    n_heads=2, max_len=5).eval()
    all_item_feats = torch.randn(20, 16)
    ctx = torch.randn(4, 12)
    items = torch.randn(4, 3, 16)
    lengths = torch.tensor([3, 3, 3, 3])
    target = torch.tensor([1, 5, 9, 13])
    item_matrix = m.item_fusion(all_item_feats)
    loss_hard = float(next_item_loss(m, ctx, items, lengths, target, item_matrix,
                                     label_smoothing=0.0))
    loss_smooth = float(next_item_loss(m, ctx, items, lengths, target, item_matrix,
                                       label_smoothing=0.05))
    # Should not be equal and should be at least somewhat different (well
    # above float-noise). The actual values depend on init, but smoothing
    # adds the uniform-target floor so the two must differ.
    assert abs(loss_hard - loss_smooth) > 1e-3, (loss_hard, loss_smooth)


def test_item_fusion_default_dropout_is_0_3():
    """Regularization bump: default ItemFusion dropout went 0.2 -> 0.3."""
    fusion = ItemFusion(modality_dims=[16], d=8)
    # The MLP has a single Dropout layer; pull its p.
    dropouts = [m.p for m in fusion.net if isinstance(m, torch.nn.Dropout)]
    assert dropouts == [0.3], dropouts


def test_sasrec_model_default_d_is_256_and_item_fusion_default_hidden_is_1536():
    """Item-side widen: SasrecModel default d went 192 -> 256, and ItemFusion
    default hidden went 1024 -> 1536. n_layers/n_heads/max_len unchanged."""
    # SasrecModel default d.
    m = SasrecModel(item_modality_dims=[16])
    assert m.d == 256
    # n_layers and n_heads must NOT have moved as part of this widen.
    assert len(m.encoder.layers) == 2
    assert m.encoder.layers[0].self_attn.num_heads == 2
    assert m.max_len == 50
    # ItemFusion default hidden: inspect the first Linear's out_features.
    fusion = ItemFusion(modality_dims=[16])
    first_linear = next(layer for layer in fusion.net if isinstance(layer, torch.nn.Linear))
    assert first_linear.out_features == 1536
    # And the fusion default d is 256.
    last_linear = [layer for layer in fusion.net if isinstance(layer, torch.nn.Linear)][-1]
    assert last_linear.out_features == 256


from mcrs.retrieval_modules.sasrec_model import build_user_dialog


def test_build_user_dialog_keeps_only_user_turns():
    turns = [{"role": "user", "content": "play something upbeat"},
             {"role": "music", "content": "track-uuid-1"},
             {"role": "user", "content": "more mellow please"}]
    assert build_user_dialog(turns) == "play something upbeat\nmore mellow please"
    assert build_user_dialog([{"role": "music", "content": "x"}]) == ""


def test_new_format_checkpoint_loads_under_weights_only_true(tmp_path):
    """A checkpoint saved with item_feats as a torch tensor (and track_ids as
    a list[str]) must load under torch.load(..., weights_only=True). This is
    the format scripts/train_sasrec.py emits after the save-side cleanup."""
    model = SasrecModel(item_modality_dims=[16], ctx_in_dim=12, d=8, n_layers=1,
                        n_heads=2, max_len=5).eval()
    item_feats = torch.randn(20, 16, dtype=torch.float32)
    track_ids = [f"t{i}" for i in range(20)]
    ckpt_path = tmp_path / "sasrec.pt"
    model_kwargs = {"item_modality_dims": [16], "ctx_in_dim": 12,
                    "d": 8, "n_layers": 1, "n_heads": 2, "max_len": 5}
    torch.save({
        "state_dict": model.state_dict(),
        "model_kwargs": model_kwargs,
        "item_feats": item_feats,
        "track_ids": track_ids,
    }, ckpt_path)
    # The point of this test: weights_only=True must succeed on this layout.
    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert isinstance(loaded["item_feats"], torch.Tensor)
    assert loaded["item_feats"].dtype == torch.float32
    assert loaded["item_feats"].shape == (20, 16)
    assert loaded["track_ids"] == track_ids
    assert loaded["model_kwargs"]["d"] == 8
    # And the state dict actually reconstructs into a SasrecModel.
    m2 = SasrecModel(**loaded["model_kwargs"])
    m2.load_state_dict(loaded["state_dict"])


# ---- encode(): masked padding must not leak into the session state ----------

def test_encode_state_invariant_to_trailing_masked_padding():
    """encode() reads the hidden at position `lengths`, with every later slot
    masked (key_padding = idx > lengths). So appending extra item slots — even
    with arbitrary content — must NOT change the state. This is exactly why
    training at max_seq=50 and gating at model.max_len yield identical states:
    sessions have far fewer played tracks than either cap, and the surplus slots
    are masked. Guards the train/gate sequence-length parity claim."""
    torch.manual_seed(0)
    m = SasrecModel(item_modality_dims=[16], ctx_in_dim=12, d=8,
                    n_layers=2, n_heads=2, max_len=50).eval()
    B, item_in = 4, 16
    ctx = torch.randn(B, 12)
    lengths = torch.tensor([3, 2, 1, 0])           # incl. an empty-prefix turn-1 row
    real = torch.randn(B, 3, item_in)              # exactly max(lengths) real slots
    padded = torch.cat([real, torch.randn(B, 7, item_in)], dim=1)  # +7 masked slots
    with torch.no_grad():
        s_real = m.encode(ctx, real, lengths)
        s_padded = m.encode(ctx, padded, lengths)
    assert torch.allclose(s_real, s_padded, atol=1e-5), \
        "content in masked padding slots leaked into the session state"
