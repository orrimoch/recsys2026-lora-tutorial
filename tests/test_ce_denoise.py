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


# ---------------------------------------------------------------------------
# Hardening tests
# ---------------------------------------------------------------------------

def test_normalize_title_empty_string():
    """normalize_title on an empty string must return an empty string (no crash)."""
    assert normalize_title("") == ""
    assert normalize_title(None) == ""        # the `(t or "")` guard


def test_normalize_title_digits_outside_parens_preserved():
    """Digits outside parentheses are kept; digits inside parentheses are stripped with them."""
    assert normalize_title("track 2") == "track 2"
    assert normalize_title("track (2023)") == "track"


def test_is_near_dup_both_empty_is_not_dup():
    """Two empty titles (normalize to '') must NOT be called near-dups (bool(na) guard)."""
    assert is_near_dup("", "") is False
    assert is_near_dup("(live)", "(remastered)") is False   # both normalize to ''


def test_is_near_dup_symmetric():
    """is_near_dup must be symmetric."""
    assert is_near_dup("Holocene (Remastered)", "Holocene") == \
           is_near_dup("Holocene", "Holocene (Remastered)")


def test_is_same_artist_empty_artist_returns_false():
    """If either artist is empty/None, is_same_artist must return False (bool guards)."""
    assert is_same_artist("a", "b", lambda tid: None) is False
    assert is_same_artist("a", "b", lambda tid: "") is False
