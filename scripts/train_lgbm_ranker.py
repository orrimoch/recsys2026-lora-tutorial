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


def split_by_holdout(df: pd.DataFrame, holdout_ids):
    """Split a feature frame into (train, holdout) by session_id. The holdout =
    rows whose session_id is in holdout_ids (the latest-by-date sessions from
    carve_temporal_selection_set); train = the rest. Used to early-stop on a
    leak-free temporal holdout instead of the leaky internal val. Raises if the
    holdout is empty (the parquet must include the holdout sessions)."""
    ids = set(holdout_ids)
    mask = df["session_id"].isin(ids)
    hold = df[mask]
    if len(hold) == 0:
        raise ValueError(
            "holdout split is empty — the train parquet contains none of the "
            "holdout session_ids; build features over the holdout sessions too")
    return df[~mask], hold


def drop_all_negative_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Drop (session_id, turn_number) groups with no positive (label.sum()==0).

    With recall@100 ~0.5, roughly half the surfaced groups have zero gold rows;
    under lambdarank they contribute zero gradient anyway, so removing them is
    leak-free hygiene (faster/leaner, ~no model change). Preserves row order
    within the kept groups so build_groups stays aligned."""
    pos = df.groupby(["session_id", "turn_number"])["label"].transform("sum")
    return df[pos > 0].reset_index(drop=True)


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
    n_bag: int = 1,
) -> None:
    """Layout matches LGBM_RERANKER.__init__."""
    meta = {
        "features": features,
        "categorical_features": categorical_features,
        "categorical_levels": categorical_levels,
        "best_iteration": int(best_iteration),
        "best_val_ndcg20": float(best_val_ndcg20),
        "n_bag": int(n_bag),
    }
    (Path(out_dir) / "metadata.json").write_text(json.dumps(meta, indent=2))


def main():
    import lightgbm as lgb

    p = argparse.ArgumentParser()
    p.add_argument("--train-features", required=True)
    p.add_argument("--val-features", default=None,
                   help="Validation parquet for early stopping. The internal val "
                        "is LEAKY (anti-correlated with dev) — prefer --holdout-ids.")
    p.add_argument("--holdout-ids", default=None,
                   help="JSON from carve_temporal_selection_set.py. When set, "
                        "early-stop on the leak-free temporal holdout carved from "
                        "the train parquet (overrides --val-features).")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-estimators", type=int, default=1000)
    p.add_argument("--early-stopping", type=int, default=50)
    # Tier-2 #4.3 knobs.
    p.add_argument("--num-leaves", type=int, default=31)
    p.add_argument("--min-data-in-leaf", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-bag", type=int, default=1,
                   help="Multi-seed bagging: train N boosters (seeds seed..seed+N-1) "
                        "and average their scores at serve. 1 = single model (default).")
    p.add_argument("--drop-all-negative", action="store_true",
                   help="Drop train groups with no positive (leak-free hygiene).")
    args = p.parse_args()
    if not args.val_features and not args.holdout_ids:
        p.error("provide --holdout-ids (preferred, leak-free) or --val-features")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(args.train_features)
    using_holdout = bool(args.holdout_ids)
    if using_holdout:
        import json as _json
        hid = _json.loads(Path(args.holdout_ids).read_text())
        holdout_ids = hid.get("holdout_ids", hid) if isinstance(hid, dict) else hid
        train_df, val_df = split_by_holdout(train_df, holdout_ids)
        print(f"[lgbm] early-stopping on the temporal holdout (leak-free): "
              f"train={train_df['session_id'].nunique()} sessions, "
              f"holdout={val_df['session_id'].nunique()} sessions")
    else:
        val_df = pd.read_parquet(args.val_features)
    if args.drop_all_negative:
        before = len(train_df)
        train_df = drop_all_negative_groups(train_df)
        print(f"[lgbm] dropped all-negative train groups: {before} -> {len(train_df)} rows")
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
        # Tier-2 #4.3: align the lambdarank pair-truncation with the nDCG@20 metric
        # (LightGBM defaults to 30, which optimizes pairs beyond the cutoff we score).
        "lambdarank_truncation_level": 20,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "min_data_in_leaf": args.min_data_in_leaf,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l2": 1.0,
        "verbosity": -1,
    }

    # Tier-2 #4.3 multi-seed bagging: train n_bag boosters with different seeds and
    # average their scores at serve (variance reduction; adds no feature -> cannot
    # worsen the in-sample leak). n_bag=1 -> a single booster.txt (backward compat).
    n_bag = max(1, int(args.n_bag))
    best_iter, best_score = 0, 0.0
    for b in range(n_bag):
        seed = args.seed + b
        bag_params = {**params, "seed": seed, "bagging_seed": seed,
                      "feature_fraction_seed": seed}
        model = lgb.train(
            bag_params, train_ds, num_boost_round=args.n_estimators,
            valid_sets=[val_ds], valid_names=["val"],
            callbacks=[lgb.early_stopping(args.early_stopping), lgb.log_evaluation(50)],
        )
        fname = "booster.txt" if n_bag == 1 else f"booster_{b}.txt"
        model.save_model(str(out_dir / fname))
        best_iter = int(model.best_iteration or 0)
        best_score = float(model.best_score.get("val", {}).get("ndcg@20", 0.0))
        if n_bag > 1:
            print(f"[lgbm] bag {b+1}/{n_bag} (seed={seed}) "
                  f"best_iter={best_iter} val_ndcg@20={best_score:.4f}")
    booster_path = out_dir / ("booster.txt" if n_bag == 1 else "booster_0.txt")

    write_metadata_json(
        out_dir=str(out_dir),
        features=feat_cols,
        categorical_features=cat_in_feats,
        categorical_levels={c: train_levels[c] for c in cat_in_feats},
        best_iteration=best_iter,
        best_val_ndcg20=best_score,
        n_bag=n_bag,
    )

    importance = sorted(
        zip(feat_cols, model.feature_importance(importance_type="gain")),
        key=lambda x: -x[1],
    )[:20]
    if using_holdout:
        print(f"[lgbm] saved → {booster_path} (best_iter={best_iter}, "
              f"holdout_ndcg@20={best_score:.4f} [temporal holdout — leak-free, "
              f"selection-safe])")
    else:
        print(f"[lgbm] saved → {booster_path} (best_iter={best_iter}, "
              f"val_ndcg@20={best_score:.4f} [early-stopping only])")
        print(internal_val_warning())
    print("[lgbm] top-20 features by gain:")
    for name, gain in importance:
        print(f"  {name}: {gain:.2f}")


if __name__ == "__main__":
    main()
