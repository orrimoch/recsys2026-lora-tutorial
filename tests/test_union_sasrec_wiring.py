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
