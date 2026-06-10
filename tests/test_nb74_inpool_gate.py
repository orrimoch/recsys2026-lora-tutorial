"""Regression guards for the nb74 # 12c-inpool dev gate cell.

Two train/serve parity bugs were fixed in this cell and must not silently
regress (the gate cannot be unit-imported, so we assert on its source):

  1. The bge context encoder must normalize embeddings — train_sasrec_inpool /
     encode_dialogs_cached encode with normalize_embeddings=True. Omitting it at
     the gate let an un-normalized ctx token (|v|~10-25) dominate the played-track
     tokens inside SasrecModel.encode, muting the SASRec sequence signal.
  2. The union pool (use_sasrec=True) must receive user_dialog (build_user_dialog,
     user-turns-only) in batch_context, else the SASRec recall channel falls back
     to the raw query (sasrec_seq.py warns) and ranks a slightly-wrong pool.
"""
import json
import os

NB = os.path.join(os.path.dirname(__file__), "..", "colab",
                  "74_e2e_sasrec_union_lgbm_ndcg.ipynb")


def _cell_source(tag):
    """Concatenated source of the first code cell whose body starts with `tag`."""
    with open(NB, encoding="utf-8") as f:
        nb = json.load(f)
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if src.lstrip().startswith(tag):
            return src
    return None


def test_inpool_gate_cell_exists():
    assert _cell_source("# 12c-inpool") is not None, \
        "# 12c-inpool gate cell missing from nb74"


def test_inpool_gate_normalizes_context_embedding():
    src = _cell_source("# 12c-inpool")
    assert "normalize_embeddings=True" in src, \
        "gate ctx encode must use normalize_embeddings=True (train/serve parity)"


def test_inpool_gate_passes_user_dialog_to_recall_channel():
    src = _cell_source("# 12c-inpool")
    assert "build_user_dialog" in src and "'user_dialog'" in src, \
        "gate must pass user_dialog (build_user_dialog, user-turns-only) into the union batch_context"


def test_inpool_gate_uses_goalful_reranker_context():
    # The in-pool RERANKER context stays goal-ful (build_sasrec_context) — distinct
    # from the goal-less user_dialog fed to the recall channel. Guard we keep both.
    src = _cell_source("# 12c-inpool")
    assert "build_sasrec_context" in src, \
        "gate must build the reranker context via build_sasrec_context (goal-ful)"
