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


# --- lyrics content channel (A1, roadmap 2026-05-30): a 2nd content view
#     (lyrics-qwen3 embeddings, precomputed) to attack the new-artist wall.
#     Opt-in via use_lyrics so existing configs (incl. shipped 194) are unchanged. ---

def test_union_specs_default_has_no_lyrics():
    specs = _wrrf_union_v1_specs({})
    assert all(s["type"] != "dense_lyrics_qwen3_instruct" for s in specs)


def test_union_specs_add_lyrics_when_enabled():
    specs = _wrrf_union_v1_specs({"use_lyrics": True, "w_lyrics": 0.5})
    lyr = next(s for s in specs if s["type"] == "dense_lyrics_qwen3_instruct")
    assert lyr["weight"] == 0.5
    assert lyr["topk_internal"] == 100


def test_union_specs_lyrics_default_weight():
    specs = _wrrf_union_v1_specs({"use_lyrics": True})
    lyr = next(s for s in specs if s["type"] == "dense_lyrics_qwen3_instruct")
    assert lyr["weight"] == 0.4  # mirrors the metadata-dense default mass


# --- cf-bpr union channel (Lever 4): user x item affinity, orthogonal to the
#     session/content channels. Opt-in via use_cfbpr; low default weight (0.25)
#     since only ~43% of users are warm (cold -> empty list, RRF falls back). ---

def test_union_specs_default_has_no_cfbpr():
    specs = _wrrf_union_v1_specs({})
    assert all(s["type"] != "cf_bpr" for s in specs)


def test_union_specs_add_cfbpr_when_enabled():
    specs = _wrrf_union_v1_specs({"use_cfbpr": True, "w_cfbpr": 0.5})
    cf = next(s for s in specs if s["type"] == "cf_bpr")
    assert cf["weight"] == 0.5
    assert cf["topk_internal"] == 100


def test_union_specs_cfbpr_default_weight():
    specs = _wrrf_union_v1_specs({"use_cfbpr": True})
    cf = next(s for s in specs if s["type"] == "cf_bpr")
    assert cf["weight"] == 0.25  # low: ~43% warm-user coverage


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
