"""Tests for the LightGBM LambdaRank trainer."""
import pytest


def test_build_groups_preserves_dataframe_row_order():
    """build_groups must use sort=False — group sizes correspond to the row
    order of the DataFrame, NOT to the sorted group keys."""
    import pandas as pd
    from scripts.train_lgbm_ranker import build_groups
    # Out-of-order keys: the FIRST group encountered is (s2, 1).
    df = pd.DataFrame([
        {"session_id": "s2", "turn_number": 1},
        {"session_id": "s2", "turn_number": 1},
        {"session_id": "s1", "turn_number": 1},
        {"session_id": "s1", "turn_number": 1},
        {"session_id": "s1", "turn_number": 1},
    ])
    groups = build_groups(df)
    # Row-order: 2 rows for (s2,1), then 3 rows for (s1,1).
    assert groups == [2, 3], (
        f"build_groups returned {groups}; expected [2, 3]. "
        "If you see [3, 2], you used sort=True (default), which corrupts the alignment "
        "between (X, y) row order and group sizes."
    )


def test_build_groups_handles_already_sorted():
    import pandas as pd
    from scripts.train_lgbm_ranker import build_groups
    df = pd.DataFrame([
        {"session_id": "s1", "turn_number": 1},
        {"session_id": "s1", "turn_number": 1},
        {"session_id": "s1", "turn_number": 2},
        {"session_id": "s2", "turn_number": 1},
    ])
    groups = build_groups(df)
    assert groups == [2, 1, 1]


def test_write_metadata_json_emits_features_and_categorical_levels(tmp_path):
    """The metadata.json layout MUST match mcrs.rerankers.lgbm_rerank.LGBM_RERANKER's reader."""
    from scripts.train_lgbm_ranker import write_metadata_json
    out_dir = tmp_path / "lgbm_model"
    out_dir.mkdir()
    write_metadata_json(
        out_dir=str(out_dir),
        features=["wrrf_rank", "ce_score", "goal_category"],
        categorical_features=["goal_category"],
        categorical_levels={"goal_category": ["A", "B", "C"]},
        best_iteration=137,
        best_val_ndcg20=0.42,
    )
    import json
    meta = json.loads((out_dir / "metadata.json").read_text())
    assert meta["features"] == ["wrrf_rank", "ce_score", "goal_category"]
    assert meta["categorical_features"] == ["goal_category"]
    assert meta["categorical_levels"] == {"goal_category": ["A", "B", "C"]}
    assert meta["best_iteration"] == 137
    assert abs(meta["best_val_ndcg20"] - 0.42) < 1e-9
