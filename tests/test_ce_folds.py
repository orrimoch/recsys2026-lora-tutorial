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
