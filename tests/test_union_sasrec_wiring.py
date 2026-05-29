from mcrs.retrieval_modules import _wrrf_union_v1_specs


def test_union_specs_default_has_no_sasrec():
    specs = _wrrf_union_v1_specs({})
    assert all(s["type"] != "sasrec_seq" for s in specs)


def test_union_specs_add_sasrec_when_enabled():
    specs = _wrrf_union_v1_specs({"use_sasrec": True, "w_sasrec": 0.9})
    sas = next(s for s in specs if s["type"] == "sasrec_seq")
    assert sas["weight"] == 0.9
    assert sas["topk_internal"] == 100
    assert sas["extra_config"]["model_dir"] == "sasrec_v1"


# --- dense channel instruct fix (nb74 Stage 7: raw dense recall@100 0.0894 ->
#     instruct 0.1789, 2x). Qwen3-Embedding is asymmetric; the query side needs
#     the instruct prefix. Default to the fixed variant; keep ablatable. ---

def test_union_dense_uses_instruct_by_default():
    specs = _wrrf_union_v1_specs({})
    dense = [s for s in specs if "dense" in s["type"]]
    assert len(dense) == 1
    assert dense[0]["type"] == "dense_metadata_qwen3_instruct"


def test_union_dense_instruct_can_be_disabled_for_ablation():
    specs = _wrrf_union_v1_specs({"dense_instruct": False})
    dense = [s for s in specs if "dense" in s["type"]]
    assert len(dense) == 1
    assert dense[0]["type"] == "dense_metadata_qwen3"
