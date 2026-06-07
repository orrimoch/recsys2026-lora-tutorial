"""LightGBM LambdaRank trainer for Stage C.

Writes `booster.txt` + `metadata.json` in the layout the existing
`mcrs.rerankers.lgbm_rerank.LGBM_RERANKER` already reads — so we plug into
the existing reranker path without changing inference code.

Usage:
  python scripts/train_lgbm_ranker.py \
    --train-features experiments/cache/retrieval_v2/lgbm/lgbm_train_split.parquet \
    --val-features   experiments/cache/retrieval_v2/lgbm/lgbm_val_split.parquet \
    --output-dir     experiments/cache/retrieval_v2/lgbm/lgbm_v1
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd


# Categorical columns (LGBM-native categorical handling).
CATEGORICAL_FEATURES = [
    "goal_category", "goal_specificity",
    "user_age_group", "user_country", "user_gender",
]

# Columns we do NOT pass to the model (ids + label).
NON_FEATURE_COLS = {"query_id", "session_id", "user_id", "turn_number", "candidate_tid", "label"}


def internal_val_warning() -> str:
    """Tier-0 #4 guard text, printed every run.

    The internal val parquet overlaps the train sample ~98.6% by session AND
    shares the full-train SASRec/cf-bpr pool, so its nDCG is leaky and has been
    anti-correlated with dev (the 'internal val trap'). It is valid ONLY for
    early stopping — never for model/config selection.
    """
    return (
        "[lgbm] NOTE: val_ndcg@20 is for EARLY STOPPING ONLY — do not select on "
        "it. The internal val is leaky (~98.6% session overlap with train + shared "
        "SASRec/cf-bpr pool) and anti-correlated with dev. Select configs on the "
        "temporal holdout (scripts/carve_temporal_selection_set.py) and confirm on "
        "split='test'."
    )


def build_groups(df: pd.DataFrame) -> list[int]:
    """Group sizes by (session_id, turn_number) — preserves DataFrame row order.

    CRITICAL: pandas' groupby default is sort=True, which would sort by group
    key and silently misalign with the (X, y) row order. We pass sort=False
    so the i-th group size corresponds to the i-th *block* of rows in df.
    """
    return df.groupby(["session_id", "turn_number"], sort=False).size().tolist()


def _encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Cast categorical columns to pandas 'category' dtype + capture level lists.

    Returns (df_encoded, {col: levels_list}) — levels_list is the .cat.categories
    in the order LightGBM saw them, so the inference-time encoder can map strings
    back to the same integer codes.
    """
    levels: dict[str, list[str]] = {}
    out = df.copy()
    for c in CATEGORICAL_FEATURES:
        if c not in out.columns:
            continue
        out[c] = out[c].astype("category")
        levels[c] = list(out[c].cat.categories)
    return out, levels


def write_metadata_json(
    out_dir: str,
    features: list[str],
    categorical_features: list[str],
    categorical_levels: dict[str, list[str]],
    best_iteration: int,
    best_val_ndcg20: float,
) -> None:
    """Layout matches LGBM_RERANKER.__init__."""
    meta = {
        "features": features,
        "categorical_features": categorical_features,
        "categorical_levels": categorical_levels,
        "best_iteration": int(best_iteration),
        "best_val_ndcg20": float(best_val_ndcg20),
    }
    (Path(out_dir) / "metadata.json").write_text(json.dumps(meta, indent=2))


def main():
    import lightgbm as lgb

    p = argparse.ArgumentParser()
    p.add_argument("--train-features", required=True)
    p.add_argument("--val-features", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-estimators", type=int, default=1000)
    p.add_argument("--early-stopping", type=int, default=50)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(args.train_features)
    val_df = pd.read_parquet(args.val_features)
    # Sort rows so build_groups returns aligned sizes.
    train_df = train_df.sort_values(["session_id", "turn_number"]).reset_index(drop=True)
    val_df = val_df.sort_values(["session_id", "turn_number"]).reset_index(drop=True)

    train_df, train_levels = _encode_categoricals(train_df)
    # Use train levels to encode val so codes align.
    for c in CATEGORICAL_FEATURES:
        if c not in val_df.columns:
            continue
        val_df[c] = pd.Categorical(val_df[c], categories=train_levels[c])

    feat_cols = [c for c in train_df.columns if c not in NON_FEATURE_COLS]
    train_X = train_df[feat_cols]
    train_y = train_df["label"].astype(int).values
    val_X = val_df[feat_cols]
    val_y = val_df["label"].astype(int).values
    train_groups = build_groups(train_df)
    val_groups = build_groups(val_df)

    cat_in_feats = [c for c in CATEGORICAL_FEATURES if c in feat_cols]

    train_ds = lgb.Dataset(train_X, label=train_y, group=train_groups,
                            categorical_feature=cat_in_feats, free_raw_data=False)
    val_ds = lgb.Dataset(val_X, label=val_y, group=val_groups,
                          categorical_feature=cat_in_feats, reference=train_ds, free_raw_data=False)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [20],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l2": 1.0,
        "verbosity": -1,
    }

    model = lgb.train(
        params, train_ds, num_boost_round=args.n_estimators,
        valid_sets=[val_ds], valid_names=["val"],
        callbacks=[lgb.early_stopping(args.early_stopping), lgb.log_evaluation(50)],
    )

    booster_path = out_dir / "booster.txt"
    model.save_model(str(booster_path))
    best_iter = int(model.best_iteration or 0)
    best_score = float(model.best_score.get("val", {}).get("ndcg@20", 0.0))

    write_metadata_json(
        out_dir=str(out_dir),
        features=feat_cols,
        categorical_features=cat_in_feats,
        categorical_levels={c: train_levels[c] for c in cat_in_feats},
        best_iteration=best_iter,
        best_val_ndcg20=best_score,
    )

    importance = sorted(
        zip(feat_cols, model.feature_importance(importance_type="gain")),
        key=lambda x: -x[1],
    )[:20]
    print(f"[lgbm] saved → {booster_path} (best_iter={best_iter}, "
          f"val_ndcg@20={best_score:.4f} [early-stopping only])")
    print(internal_val_warning())
    print("[lgbm] top-20 features by gain:")
    for name, gain in importance:
        print(f"  {name}: {gain:.2f}")


if __name__ == "__main__":
    main()
