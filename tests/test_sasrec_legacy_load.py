"""Legacy checkpoint fallback: the sasrec_seq factory branch must still load
older checkpoints that bundle a numpy item_feats matrix (which trips PyTorch
2.6's weights_only=True unpickler). The loader catches pickle.UnpicklingError,
emits a UserWarning, and re-loads with weights_only=False.

Touches the factory entry point in mcrs.retrieval_modules.__init__ to make
sure the try/except wires correctly end-to-end. SentenceTransformer is
monkeypatched so this test does not require network access.
"""
import os
import sys
import warnings

import numpy as np
import torch

import mcrs.retrieval_modules as retrieval_modules
from mcrs.retrieval_modules.sasrec_model import SasrecModel


class _FakeST:
    def __init__(self, *_, **__):
        self.max_seq_length = 512

        class _Tok:
            truncation_side = "right"

        self.tokenizer = _Tok()

    def encode(self, texts, **__):
        return np.zeros((len(list(texts)), 12), dtype=np.float32)


def _write_legacy_checkpoint(model_dir):
    """Save a checkpoint in the OLD format: item_feats is a numpy ndarray.
    This is what tripped torch.load(weights_only=True) in PyTorch 2.6+."""
    os.makedirs(model_dir, exist_ok=True)
    model = SasrecModel(item_modality_dims=[16], ctx_in_dim=12, d=8, n_layers=1,
                        n_heads=2, max_len=5).eval()
    item_feats_np = np.random.RandomState(0).randn(20, 16).astype(np.float32)
    track_ids = [f"t{i}" for i in range(20)]
    torch.save({
        "state_dict": model.state_dict(),
        "model_kwargs": {"item_modality_dims": [16], "ctx_in_dim": 12,
                         "d": 8, "n_layers": 1, "n_heads": 2, "max_len": 5},
        "item_feats": item_feats_np,  # numpy on purpose — the legacy case.
        "track_ids": track_ids,
    }, os.path.join(model_dir, "sasrec.pt"))


def test_legacy_numpy_checkpoint_loads_via_factory_with_warning(tmp_path, monkeypatch):
    """The factory loader must warn AND still return a working module when
    given a legacy numpy-item_feats checkpoint."""
    cache_dir = tmp_path
    model_dir = cache_dir / "retrieval_v2" / "sasrec" / "sasrec_v1"
    _write_legacy_checkpoint(str(model_dir))

    # Avoid the network/model download that the factory does to build the
    # text encoder. The factory does `from sentence_transformers import
    # SentenceTransformer` lazily, so we shim the module entry it sees.
    fake_mod = type(sys)("sentence_transformers")
    fake_mod.SentenceTransformer = _FakeST
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_mod)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mod = retrieval_modules.load_retrieval_module(
            "sasrec_seq",
            dataset_name="dummy",
            track_split_types=[],
            cache_dir=str(cache_dir),
            extra_config={"model_dir": "sasrec_v1"},
        )
    # The fallback warning must have fired.
    msgs = [str(w.message) for w in caught]
    assert any("legacy SASRec checkpoint" in m for m in msgs), msgs
    # And the loader must have returned a usable retriever.
    assert mod is not None
    assert hasattr(mod, "batch_text_to_item_retrieval")
