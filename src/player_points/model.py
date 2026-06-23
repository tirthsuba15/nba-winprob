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

try:
    from .features import build_features, FEATURES, TARGET
except ImportError:
    from features import build_features, FEATURES, TARGET

MIN_TRAIN_GAMES = 5   # drop players with fewer training samples
MIN_GAMES_FOR_ROLLING = 3   # drop rows where rolling features are still warming up


def _train_xgb(X, y, objective, alpha=None, sample_weight=None):
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
    clf.fit(X, y, sample_weight=sample_weight)
    return clf


def train_models(X_tr, y_tr, sample_weight=None):
    """Train mean + lower-bound + upper-bound models."""
    print("  Training mean model …")
    m_mean = _train_xgb(X_tr, y_tr, "reg:squarederror", sample_weight=sample_weight)
    print("  Training q10 model …")
    m_lo   = _train_xgb(X_tr, y_tr, "reg:quantileerror", alpha=0.10, sample_weight=sample_weight)
    print("  Training q90 model …")
    m_hi   = _train_xgb(X_tr, y_tr, "reg:quantileerror", alpha=0.90, sample_weight=sample_weight)
    return {"mean": m_mean, "lo": m_lo, "hi": m_hi}


def recency_weights(train_df, last_n=7, factor=3.0):
    """Weight each player's most recent `last_n` training games `factor`x.

    Training-only — does not touch features or test, so no leakage.
    """
    rank = train_df.groupby("player_id")["game_date"].rank(method="first", ascending=False)
    w = np.where(rank <= last_n, factor, 1.0)
    return w.astype(float)


def predict_distribution(models, X):
    """Return (mean, lo, hi) arrays with quantile-crossing fix."""
    y_mean = models["mean"].predict(X).clip(min=0)
    y_lo   = models["lo"].predict(X).clip(min=0)
    y_hi   = models["hi"].predict(X).clip(min=0)
    # Enforce ordering: lo <= mean <= hi
    y_lo  = np.minimum(y_lo, y_mean)
    y_hi  = np.maximum(y_hi, y_mean)
    return y_mean, y_lo, y_hi


# ── minutes sub-model + bundle ────────────────────────────────────────────────────
# Pre-game drivers of how many minutes a player will get tonight. The minutes
# model predicts minutes from these; pred_minutes is then a feature for points.
# LEAKAGE: pred_minutes uses ONLY pre-game inputs; the actual game's min_dec is
# never a points feature (it is only the minutes model's training label).
MINUTES_FEATURES = ["min_season_avg", "home", "days_rest", "b2b", "top2_teammate_out"]
POINTS_FEATURES = FEATURES + ["pred_minutes"]
REQUIRED_BUNDLE_KEYS = ("mean", "lo", "hi", "minutes", "features", "po_cal")


def validate_bundle(bundle):
    """Raise a clear ValueError if the model bundle is malformed / stale."""
    if not isinstance(bundle, dict):
        raise ValueError(
            "model bundle is not a dict (stale/corrupt pickle). Retrain: "
            "python src/player_points/model.py --data data")
    missing = [k for k in REQUIRED_BUNDLE_KEYS if k not in bundle]
    if missing:
        raise ValueError(
            f"model bundle missing required key(s) {missing}. Retrain: "
            "python src/player_points/model.py --data data")
    return bundle


def train_minutes(train_df, sample_weight=None):
    """Train an XGBoost minutes model (target = actual minutes that game)."""
    from xgboost import XGBRegressor
    X = train_df[MINUTES_FEATURES].to_numpy(dtype=float)
    y = train_df["min_dec"].to_numpy(dtype=float)
    m = XGBRegressor(n_estimators=250, max_depth=4, learning_rate=0.05,
                     subsample=0.9, colsample_bytree=0.8, min_child_weight=5,
                     n_jobs=-1, tree_method="hist", objective="reg:squarederror")
    m.fit(X, y, sample_weight=sample_weight)
    return m


def points_matrix(df, minutes_model):
    """Build the points feature matrix = FEATURES + pred_minutes (leakage-safe)."""
    pred_min = minutes_model.predict(df[MINUTES_FEATURES].to_numpy(dtype=float)).clip(min=0)
    return np.column_stack([df[FEATURES].to_numpy(dtype=float), pred_min])


def train_bundle(train_df):
    """Train minutes model + mean/q10/q90 points models + an isotonic P(over)
    calibrator (fit on train only). Returns a bundle dict.

    The calibrator is stored so the SERVING path (predict.py / app.py) returns
    the same calibrated probabilities the backtest reports — not the raw,
    overconfident ones.
    """
    weights = recency_weights(train_df)   # last 7 games per player weighted 3x
    print(f"  Recency weighting: {(weights > 1).sum():,} of {len(weights):,} "
          f"train rows at 3x (each player's last 7 games)")
    print("  Training minutes sub-model …")
    minutes = train_minutes(train_df, sample_weight=weights)
    Xtr = points_matrix(train_df, minutes)
    ytr = train_df[TARGET].to_numpy(dtype=float)
    models = train_models(Xtr, ytr, sample_weight=weights)
    bundle = {"mean": models["mean"], "lo": models["lo"], "hi": models["hi"],
              "minutes": minutes, "features": POINTS_FEATURES}

    # fit isotonic calibrator on train over/under (line = season-avg-to-date)
    try:
        from .odds import prob_over
    except ImportError:
        from odds import prob_over
    from sklearn.isotonic import IsotonicRegression
    m, lo, hi = predict_distribution(bundle, Xtr)
    line = train_df["pts_season_avg"].to_numpy(dtype=float)
    valid = ~np.isnan(line)
    if valid.sum() > 100:
        p_raw = np.array([prob_over(a, b, c, d) for a, b, c, d in
                          zip(m[valid], lo[valid], hi[valid], line[valid])])
        over = (ytr[valid] > line[valid]).astype(int)
        bundle["po_cal"] = IsotonicRegression(out_of_bounds="clip").fit(p_raw, over)
    return bundle


def predict_bundle(bundle, df):
    """(mean, lo, hi) for rows in df using the full bundle."""
    X = points_matrix(df, bundle["minutes"])
    return predict_distribution(bundle, X)


def calibrate_p(bundle, p_raw):
    """Apply the bundle's isotonic P(over) calibrator. Scalar in -> scalar out;
    array in -> array out. No-op if the bundle has no calibrator."""
    cal = bundle.get("po_cal")
    if cal is None:
        return p_raw
    arr = np.atleast_1d(np.asarray(p_raw, dtype=float))
    out = np.clip(cal.predict(arr), 0.0, 1.0)
    return float(out[0]) if np.ndim(p_raw) == 0 else out


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

    y_te = test_df[TARGET].to_numpy(dtype=float)

    print("\nTraining bundle (minutes sub-model + points models) …")
    bundle = train_bundle(train_df)
    y_mean, y_lo, y_hi = predict_bundle(bundle, test_df)

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

    # Feature importance (points mean model, over FEATURES + pred_minutes)
    imp = sorted(zip(POINTS_FEATURES, bundle["mean"].feature_importances_),
                 key=lambda x: -x[1])
    print("\nFeature importance (points mean model):")
    for f, v in imp:
        print(f"  {f:22s} {v:.3f}")

    # Save the full bundle (minutes model + points models + feature order)
    model_path = os.path.join(args.out, "player_points_models.pkl")
    with open(model_path, "wb") as fh:
        pickle.dump(bundle, fh)
    print(f"\nSaved bundle to {model_path}")

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
