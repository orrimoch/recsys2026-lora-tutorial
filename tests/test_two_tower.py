"""Tier-1 #3.3: intent->content two-tower recall channel (CPU core + wiring).

Training is Colab; these test the model math, the InfoNCE objective, the channel
wiring (fake model), and the union gating — all on CPU with tiny dims.
"""
import torch

from mcrs.retrieval_modules.two_tower_model import TwoTowerModel, info_nce_loss
from mcrs.retrieval_modules.two_tower_channel import TwoTowerRetriever
from mcrs.retrieval_modules import _wrrf_union_v1_specs


MODS = [8, 4]   # two tiny frozen modalities (sum=12)
QDIM = 6


def _model():
    torch.manual_seed(0)
    return TwoTowerModel(item_modality_dims=MODS, q_in_dim=QDIM, d=5)


def test_encode_item_and_query_are_l2_normalized():
    m = _model()
    it = m.encode_item(torch.randn(3, sum(MODS)))
    q = m.encode_query(torch.randn(3, QDIM))
    assert it.shape == (3, 5) and q.shape == (3, 5)
    assert torch.allclose(it.norm(dim=1), torch.ones(3), atol=1e-5)
    assert torch.allclose(q.norm(dim=1), torch.ones(3), atol=1e-5)


def test_info_nce_loss_decreases_on_overfit_batch():
    m = _model()
    torch.manual_seed(1)
    B = 8
    q = torch.randn(B, QDIM)
    pos = torch.randn(B, sum(MODS))
    opt = torch.optim.Adam(m.parameters(), lr=0.05)
    first = None
    for step in range(60):
        opt.zero_grad()
        loss = info_nce_loss(m, q, pos)
        loss.backward()
        opt.step()
        if step == 0:
            first = float(loss)
    assert float(loss) < first  # the tower learns to align each query to its item


def test_retriever_ranks_by_cosine():
    # fake model: query_encode -> a fixed query vector; item_repr has a clear best.
    item_repr = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]])
    track_ids = ["a", "b", "c"]

    class _FakeModel:
        def encode_query(self, q_emb):
            return torch.tensor([[1.0, 0.0]])  # closest to track "a", then "c"

    def query_encode(queries):
        return torch.zeros(len(queries), 2)  # content ignored by the fake model

    r = TwoTowerRetriever(_FakeModel(), item_repr, track_ids, query_encode)
    out = r.batch_text_to_item_retrieval(["anything"], topk=2)
    assert out[0][0] == "a"          # highest cosine
    assert out[0][1] == "c"          # second


def test_two_tower_channel_off_by_default():
    assert "two_tower" not in [s["type"] for s in _wrrf_union_v1_specs({})]


def test_use_two_tower_appends_channel():
    specs = _wrrf_union_v1_specs({"use_two_tower": True, "w_two_tower": 0.7})
    tt = [s for s in specs if s["type"] == "two_tower"]
    assert len(tt) == 1 and tt[0]["weight"] == 0.7
