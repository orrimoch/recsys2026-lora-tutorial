"""Context-encoder truncation parity for the in-pool SASRec fine-tune.

The warm-start training (train_sasrec.py main) and the serve factory
(load_retrieval_module 'sasrec_seq') BOTH configure the bge context encoder as

    st.max_seq_length = 512
    st.tokenizer.truncation_side = "left"   # keep the most-recent turns

so contexts longer than 512 tokens keep the END of the dialog — the current
user request and the trailing "goal: ..." line. The in-pool fine-tune
(scripts/train_sasrec_inpool.py) and the nb74 # 12c-inpool gate cell encoded
with the tokenizer DEFAULTS (truncation_side='right'), silently dropping the
goal + the latest turns on every >512-token context (9.9% of train turns,
5.4% of dev turns) and breaking 3-way parity. These tests pin the fix at both
sites. NOTE: train_sasrec_inpool's cache_kw must claim the SAME config it
actually encodes with, or encode_dialogs_cached serves stale embeddings.
"""
import json
import os
import re

REPO = os.path.join(os.path.dirname(__file__), "..")
SCRIPT = os.path.join(REPO, "scripts", "train_sasrec_inpool.py")
NB = os.path.join(REPO, "colab", "74_e2e_sasrec_union_lgbm_ndcg.ipynb")


def _script_src():
    with open(SCRIPT, encoding="utf-8") as f:
        return f.read()


def _gate_cell_src():
    with open(NB, encoding="utf-8") as f:
        nb = json.load(f)
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            src = "".join(c["source"])
            if src.lstrip().startswith("# 12c-inpool"):
                return src
    raise AssertionError("# 12c-inpool gate cell missing from nb74")


def test_inpool_train_configures_encoder_truncation():
    src = _script_src()
    assert re.search(r"st\.max_seq_length\s*=\s*512", src), \
        "train_sasrec_inpool must set st.max_seq_length = 512 (warm-start/serve parity)"
    assert re.search(r'truncation_side\s*=\s*"left"', src), \
        "train_sasrec_inpool must set tokenizer.truncation_side = 'left' (keep latest turns + goal)"


def test_inpool_train_cache_key_matches_actual_encoder_config():
    # cache_kw claimed max_seq_length=256 while the encoder ran at its 512
    # default — the cache key must describe the real config or a future change
    # silently serves stale embeddings.
    src = _script_src()
    assert "max_seq_length=256" not in src, \
        "cache_kw claims max_seq_length=256 but the encoder is configured to 512"
    assert re.search(r"cache_kw\s*=\s*dict\([^)]*max_seq_length=512", src, re.S), \
        "cache_kw must claim max_seq_length=512 (the actual encoder config)"


def test_inpool_gate_cell_configures_encoder_truncation():
    src = _gate_cell_src()
    assert re.search(r"st\.max_seq_length\s*=\s*512", src), \
        "# 12c-inpool gate must set st.max_seq_length = 512 (train/serve parity)"
    assert re.search(r"truncation_side\s*=\s*'left'|truncation_side\s*=\s*\"left\"", src), \
        "# 12c-inpool gate must set tokenizer.truncation_side = 'left' (train/serve parity)"
