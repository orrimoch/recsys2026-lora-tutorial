"""P4 / W3.a: CLAP text->audio recall channel union gating (regression pin).

clap_text is a COLD-firable channel (query text -> CLAP audio space -> nearest
catalog tracks by sound) — it reaches new-artist wall golds by acoustics, orthogonal
to the text/metadata channels (ColBERT included). DISTINCT from clap_recall, which
mean-pools PLAYED tracks and is dead at turn-1/Blind. Opt-in via use_clap_text.
"""
from mcrs.retrieval_modules import _wrrf_union_v1_specs


def test_clap_text_off_by_default():
    assert "clap_text" not in [s["type"] for s in _wrrf_union_v1_specs({})]


def test_use_clap_text_appends_channel_with_default_weight():
    cb = [s for s in _wrrf_union_v1_specs({"use_clap_text": True}) if s["type"] == "clap_text"]
    assert len(cb) == 1
    assert cb[0]["weight"] == 1.0
    assert cb[0]["topk_internal"] == 100


def test_w_clap_text_overrides_weight():
    cb = [s for s in _wrrf_union_v1_specs({"use_clap_text": True, "w_clap_text": 0.5})
          if s["type"] == "clap_text"]
    assert cb[0]["weight"] == 0.5
