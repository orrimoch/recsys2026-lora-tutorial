from mcrs.training.ce_data import assign_session_folds, drop_cross_fold_near_dups

def test_sessions_never_split_across_folds():
    sids = [f"s{i}" for i in range(10)]
    turns = [(f"{s}", i) for s in sids for i in range(3)]   # (session_id, turn)
    folds = assign_session_folds([s for s, _ in turns], k=3, seed=0)
    by_session = {}
    for (s, _), f in zip(turns, folds):
        by_session.setdefault(s, set()).add(f)
    assert all(len(fs) == 1 for fs in by_session.values())   # each session in exactly one fold

def test_drop_cross_fold_near_dups():
    # items: (key, fold); a (query,gold) key appearing in two folds must be dropped from all but one
    items = [("k1", 0), ("k1", 1), ("k2", 0), ("k3", 2)]
    keep = drop_cross_fold_near_dups(items)
    keys = [items[i][0] for i in keep]
    assert keys.count("k1") == 1 and "k2" in keys and "k3" in keys


# ---------------------------------------------------------------------------
# Hardening tests
# ---------------------------------------------------------------------------

def test_k_larger_than_sessions_all_sessions_still_assigned():
    """k > number of sessions: some folds empty, but every session lands in exactly one fold."""
    session_ids = ["s0", "s0", "s1", "s1"]   # 2 sessions, k=5
    folds = assign_session_folds(session_ids, k=5, seed=0)
    assert len(folds) == len(session_ids)
    # Both rows of s0 share the same fold, both rows of s1 share the same fold
    assert folds[0] == folds[1]               # s0's two rows are in the same fold
    assert folds[2] == folds[3]               # s1's two rows are in the same fold
    # All assigned folds are valid indices
    assert all(0 <= f < 5 for f in folds)
    # s0 and s1 may be in the same or different folds -- just not split
    assert len(set(folds[:2])) == 1           # s0 never split
    assert len(set(folds[2:])) == 1           # s1 never split


def test_single_session_all_rows_same_fold():
    """A single session with k=3 must put all rows in the same fold."""
    session_ids = ["s0", "s0", "s0"]
    folds = assign_session_folds(session_ids, k=3, seed=0)
    assert len(set(folds)) == 1               # all rows map to the same fold


def test_assign_session_folds_deterministic_same_seed():
    """Two calls with identical inputs and same seed must return identical fold assignments."""
    session_ids = ["a", "a", "b", "b", "c", "c", "d"]
    f1 = assign_session_folds(session_ids, k=3, seed=42)
    f2 = assign_session_folds(session_ids, k=3, seed=42)
    assert f1 == f2


def test_assign_session_folds_different_seeds_can_differ():
    """Different seeds on the same sessions should (usually) produce different assignments."""
    session_ids = [f"s{i}" for i in range(12)]   # 12 distinct sessions, ample permutation space
    session_ids_repeated = [s for s in session_ids for _ in range(2)]
    f1 = assign_session_folds(session_ids_repeated, k=3, seed=0)
    f2 = assign_session_folds(session_ids_repeated, k=3, seed=1)
    assert f1 != f2


def test_all_unique_keys_kept_by_dedup():
    """drop_cross_fold_near_dups with all-unique keys must keep every item."""
    items = [("k1", 0), ("k2", 1), ("k3", 2), ("k4", 0)]
    keep = drop_cross_fold_near_dups(items)
    assert keep == list(range(len(items)))    # all indices preserved


def test_all_identical_keys_keeps_only_first():
    """drop_cross_fold_near_dups with all identical keys must keep exactly the first occurrence."""
    items = [("k1", 0), ("k1", 1), ("k1", 2), ("k1", 0)]
    keep = drop_cross_fold_near_dups(items)
    assert keep == [0]                        # only index 0 survives
