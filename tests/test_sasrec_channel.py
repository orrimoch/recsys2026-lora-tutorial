import numpy as np
import torch
from mcrs.retrieval_modules.sasrec_model import SasrecModel
from mcrs.retrieval_modules.sasrec_seq import SasrecRetriever


def _build(text_encode=None):
    torch.manual_seed(0)
    model = SasrecModel(item_in_dim=16, ctx_in_dim=12, d=8, n_layers=1,
                        n_heads=2, max_len=5).eval()
    track_ids = [f"t{i}" for i in range(20)]
    item_feats = torch.randn(20, 16)
    item_repr = model.item_fusion(item_feats).detach()
    if text_encode is None:
        def text_encode(qs):
            return np.ones((len(qs), 12), dtype=np.float32)
    return SasrecRetriever(model, item_repr, track_ids, item_feats, text_encode,
                           max_len=5)


def test_channel_returns_topk_per_query():
    r = _build()
    out = r.batch_text_to_item_retrieval(
        ["q1", "q2"], topk=5,
        batch_context=[{"history_tids": ["t3", "t4"]}, {"history_tids": []}])
    assert len(out) == 2
    assert all(len(lst) == 5 for lst in out)
    assert all(t in {f"t{i}" for i in range(20)} for t in out[0])


def test_channel_handles_empty_history_turn_one():
    r = _build()
    out = r.batch_text_to_item_retrieval(
        ["q"], topk=3, batch_context=[{"history_tids": []}])
    assert len(out[0]) == 3


def test_channel_ignores_unknown_history_tids():
    r = _build()
    out = r.batch_text_to_item_retrieval(
        ["q"], topk=3, batch_context=[{"history_tids": ["UNKNOWN", "t2"]}])
    assert len(out[0]) == 3


def test_channel_uses_user_dialog_from_context():
    seen = {}
    def spy(qs):
        seen["texts"] = list(qs)
        return np.ones((len(qs), 12), dtype=np.float32)
    r = _build(text_encode=spy)
    r.batch_text_to_item_retrieval(
        ["RAW_QUERY"], topk=3,
        batch_context=[{"history_tids": [], "user_dialog": "USER_TURNS_ONLY"}])
    # the channel must encode the user_dialog text, not the raw query
    assert seen["texts"] == ["USER_TURNS_ONLY"]
