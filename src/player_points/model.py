"""
Train and backtest the player-points projection model.

Outputs a mean prediction and an 80% prediction interval using three
XGBoost quantile regressors (q10, mean/q50, q90).

Split: train on 2023-24 games, test on 2024-25 games (or pass --test-seasons).
No player-game appears in both sets.

Metrics reported vs. a naive "season-average-to-date" baseline:
  - MAE            (lower = better)
  - 80% coverage   (should be ~0.80 for a well-calibrated interval)
  - Mean interval width (narrower = more useful)

Usage:
    python src/player_points/model.py --data data
    python src/player_points/model.py --data data --test-seasons 2024-25
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import sys
import numpy as np
import pandas as pd

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from features import build_features, FEATURES, TARGET

MIN_TRAIN_GAMES = 5   # drop players with fewer training samples
MIN_GAMES_FOR_ROLLING = 3   # drop rows where rolling features are still warming up


def _train_xgb(X, y, objective, alpha=None):
    from xgboost import XGBRegressor
    params = dict(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.8, min_child_weight=5,
        n_jobs=-1, tree_method="hist",
        objective=objective,
    )
    if alpha is not None:
        params["quantile_alpha"] = alpha
    clf = XGBRegressor(**params)
    clf.fit(X, y)
    return clf


def train_models(X_tr, y_tr):
    """Train mean + lower-bound + upper-bound models."""
    print("  Training mean model …")
    m_mean = _train_xgb(X_tr, y_tr, "reg:squarederror")
    print("  Training q10 model …")
    m_lo   = _train_xgb(X_tr, y_tr, "reg:quantileerror", alpha=0.10)
    print("  Training q90 model …")
    m_hi   = _train_xgb(X_tr, y_tr, "reg:quantileerror", alpha=0.90)
    return {"mean": m_mean, "lo": m_lo, "hi": m_hi}


def predict_distribution(models, X):
    """Return (mean, lo, hi) arrays with quantile-crossing fix."""
    y_mean = models["mean"].predict(X).clip(min=0)
    y_lo   = models["lo"].predict(X).clip(min=0)
    y_hi   = models["hi"].predict(X).clip(min=0)
    # Enforce ordering: lo <= mean <= hi
    y_lo  = np.minimum(y_lo, y_mean)
    y_hi  = np.maximum(y_hi, y_mean)
    return y_mean, y_lo, y_hi


def evaluate(name, y_true, y_mean, y_lo=None, y_hi=None):
    mae = float(np.abs(y_true - y_mean).mean())
    result = {"model": name, "mae": mae, "n": int(len(y_true))}
    if y_lo is not None and y_hi is not None:
        covered = ((y_true >= y_lo) & (y_true <= y_hi)).mean()
        width   = (y_hi - y_lo).mean()
        result["coverage_80"] = float(covered)
        result["interval_width"] = float(width)
    return result


def naive_baseline(train_df, test_df):
    """Season-average-to-date baseline: predict pts_season_avg (already in features)."""
    y_pred = test_df["pts_season_avg"].fillna(test_df[TARGET].mean()).to_numpy()
    return y_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--test-seasons", nargs="+", default=None)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--min-pts", type=float, default=0.0,
                    help="filter rows: player averaged at least this many pts "
                         "in the training season (removes DNP-heavy players)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    raw = pd.read_csv(os.path.join(args.data, "player_gamelogs.csv"))
    print(f"Loaded {len(raw):,} player-game rows, "
          f"{raw['player_id'].nunique():,} players, seasons {sorted(raw['season'].unique())}")

    print("Building features …")
    df = build_features(raw)
    print(f"Feature matrix: {df.shape}")

    seasons = sorted(df["season"].unique())
    test_seasons = args.test_seasons or [seasons[-1]]
    train_df = df[~df["season"].isin(test_seasons)].copy()
    test_df  = df[ df["season"].isin(test_seasons)].copy()

    # Drop cold-start rows (early-season: rolling windows not warm)
    train_df = (
        train_df.groupby("player_id")
        .filter(lambda g: len(g) >= MIN_TRAIN_GAMES)
    )

    # Optional: filter low-minute players
    if args.min_pts > 0:
        avg_pts = train_df.groupby("player_id")[TARGET].mean()
        keep = avg_pts[avg_pts >= args.min_pts].index
        train_df = train_df[train_df["player_id"].isin(keep)]
        test_df  = test_df[ test_df["player_id"].isin(keep)]

    print(f"\nTrain: {len(train_df):,} rows ({', '.join(s for s in seasons if s not in test_seasons)})")
    print(f"Test : {len(test_df):,} rows ({', '.join(test_seasons)})")

    X_tr = train_df[FEATURES].to_numpy(dtype=float)
    y_tr = train_df[TARGET].to_numpy(dtype=float)
    X_te = test_df[FEATURES].to_numpy(dtype=float)
    y_te = test_df[TARGET].to_numpy(dtype=float)

    print("\nTraining models …")
    models = train_models(X_tr, y_tr)
    y_mean, y_lo, y_hi = predict_distribution(models, X_te)

    # Naive baseline (season-avg-to-date from features)
    y_naive = naive_baseline(train_df, test_df)

    results = [
        evaluate("naive (season avg to date)", y_te, y_naive),
        evaluate("xgboost (mean + 80% interval)", y_te, y_mean, y_lo, y_hi),
    ]

    res_df = pd.DataFrame(results).set_index("model")
    print("\nResults:")
    print(res_df.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\n(coverage_80 should be ~0.80 for a well-calibrated interval)")

    # Feature importance
    imp = sorted(zip(FEATURES, models["mean"].feature_importances_),
                 key=lambda x: -x[1])
    print("\nFeature importance (mean model):")
    for f, v in imp:
        print(f"  {f:22s} {v:.3f}")

    # Save models
    model_path = os.path.join(args.out, "player_points_models.pkl")
    with open(model_path, "wb") as fh:
        pickle.dump(models, fh)
    print(f"\nSaved models to {model_path}")

    # Save summary
    summary = {
        "test_seasons": test_seasons,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_players_train": int(train_df["player_id"].nunique()),
        "n_players_test": int(test_df["player_id"].nunique()),
        "results": results,
        "feature_importance": {f: float(v) for f, v in imp},
    }
    summary_path = os.path.join(args.out, "player_points_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
