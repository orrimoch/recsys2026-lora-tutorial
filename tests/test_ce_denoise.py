from mcrs.training.ce_data import normalize_title, is_near_dup, is_same_artist

def test_normalize_title():
    assert normalize_title("Holocene (Remastered)") == "holocene"
    assert normalize_title("  Hello, World!  ") == "hello world"

def test_is_near_dup():
    assert is_near_dup("Holocene", "holocene (live)") is True
    assert is_near_dup("Holocene", "Sunburst") is False

def test_is_same_artist():
    meta = {"a": {"artist_name": "Bon Iver"}, "b": {"artist_name": "bon iver"}, "c": {"artist_name": "Tobu"}}
    label = lambda tid: meta[tid]["artist_name"]
    assert is_same_artist("a", "b", label) is True
    assert is_same_artist("a", "c", label) is False
