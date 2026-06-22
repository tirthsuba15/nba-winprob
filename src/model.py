"""
Train + evaluate the win-probability model.

Headline metric is CALIBRATION, not accuracy: when the model says 70%, the home
team should win ~70% of the time. We report log-loss, Brier score and accuracy,
compare against three honest baselines, and draw a reliability diagram.

Baselines:
  naive     : always predict the base rate (home win %). The "do nothing" bar.
  elo_only  : pregame Elo probability -- ignores live game state entirely.
  logistic  : logistic regression on margin/time -- the classic simple model.
Model:
  xgboost   : gradient-boosted trees on all features (the nflfastR approach).

Usage:
  python model.py --data data            # uses data/moments.csv + data/games.csv
  python model.py --data data --test-seasons 2024-25
"""
from __future__ import annotations
import argparse
import json
import os
import numpy as np
import pandas as pd
from features import build_dataset, split_by_season, Xy, FEATURES, TARGET


def calibration_table(y_true, p_pred, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p_pred, bins) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        rows.append({
            "bin": f"{bins[b]:.0%}-{bins[b+1]:.0%}",
            "n": int(m.sum()),
            "predicted": float(p_pred[m].mean()),
            "actual": float(y_true[m].mean()),
        })
    return pd.DataFrame(rows)


def evaluate(name, y, p):
    from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "model": name,
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "accuracy": float(accuracy_score(y, (p >= 0.5).astype(int))),
    }


def reliability_plot(curves, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="#888", lw=1.5, label="perfect calibration")
    styles = {"xgboost (raw)": ("o--", "#d62728"), "xgboost (calibrated)": ("o-", "#2ca02c")}
    for label, (y, p) in curves.items():
        t = calibration_table(y, p, n_bins=10)
        fmt, color = styles.get(label, ("o-", None))
        kw = {"color": color} if color else {}
        ax.plot(t["predicted"], t["actual"], fmt, label=label, **kw)
    ax.set_xlabel("Predicted home win probability")
    ax.set_ylabel("Actual home win rate")
    ax.set_title("Reliability diagram (test set)")
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"  saved {path}")


def train_xgb(Xtr, ytr):
    from xgboost import XGBClassifier
    clf = XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, min_child_weight=5,
        eval_metric="logloss", n_jobs=-1, tree_method="hist",
    )
    clf.fit(Xtr, ytr)
    return clf


def calibrate_xgb(clf, Xtr, ytr):
    """Wrap raw XGBoost in isotonic calibration, fit on training data only."""
    from sklearn.calibration import CalibratedClassifierCV
    cal = CalibratedClassifierCV(clf, method="isotonic", cv=5)
    cal.fit(Xtr, ytr)
    return cal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--test-seasons", nargs="+", default=None,
                    help="seasons to hold out; default = the latest season present")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    moments = pd.read_csv(os.path.join(args.data, "moments.csv"))
    games = pd.read_csv(os.path.join(args.data, "games.csv"))
    df = build_dataset(moments, games)

    seasons = sorted(df["season"].unique())
    test_seasons = args.test_seasons or [seasons[-1]]
    train, test = split_by_season(df, test_seasons)
    print(f"Seasons: {seasons}")
    print(f"Train seasons: {[s for s in seasons if s not in test_seasons]} "
          f"({len(train):,} moments)")
    print(f"Test  seasons: {test_seasons} ({len(test):,} moments)\n")

    Xtr, ytr = Xy(train)
    Xte, yte = Xy(test)

    # --- baselines ---
    base_rate = ytr.mean()
    p_naive = np.full(len(yte), base_rate)
    p_elo = test["elo_prob_home"].fillna(base_rate).to_numpy()

    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression(max_iter=1000)
    simple_cols = ["score_margin", "frac_left", "margin_x_fracleft"]
    log.fit(train[simple_cols], ytr)
    p_log = log.predict_proba(test[simple_cols])[:, 1]

    # --- model ---
    clf = train_xgb(Xtr, ytr)
    p_xgb_raw = clf.predict_proba(Xte)[:, 1]

    print("Fitting calibration wrapper on training data (cv=5, isotonic) ...")
    clf_cal = calibrate_xgb(clf, Xtr, ytr)
    p_xgb_cal = clf_cal.predict_proba(Xte)[:, 1]

    results = [
        evaluate("naive (base rate)", yte, p_naive),
        evaluate("elo_only (pregame)", yte, p_elo),
        evaluate("logistic (margin+time)", yte, p_log),
        evaluate("xgboost (raw)", yte, p_xgb_raw),
        evaluate("xgboost (calibrated)", yte, p_xgb_cal),
    ]
    res_df = pd.DataFrame(results).set_index("model")
    print(res_df.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\nCalibration of the XGBoost (calibrated) model (test set):")
    cal_table = calibration_table(yte, p_xgb_cal)
    print(cal_table.to_string(index=False,
                               float_format=lambda x: f"{x:.3f}"))

    # importances
    imp = sorted(zip(FEATURES, clf.feature_importances_),
                 key=lambda x: -x[1])
    print("\nFeature importance:")
    for f, v in imp:
        print(f"  {f:18s} {v:.3f}")

    # --- save artifacts ---
    import pickle
    clf.save_model(os.path.join(args.out, "model.json"))
    with open(os.path.join(args.out, "model_calibrated.pkl"), "wb") as fh:
        pickle.dump(clf_cal, fh)
    res_df.to_csv(os.path.join(args.out, "metrics.csv"))
    cal_table.to_csv(os.path.join(args.out, "calibration.csv"), index=False)
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump({
            "test_seasons": test_seasons,
            "n_train_moments": int(len(train)),
            "n_test_moments": int(len(test)),
            "metrics": results,
            "feature_importance": {f: float(v) for f, v in imp},
        }, fh, indent=2)
    reliability_plot(
        {"elo_only": (yte, p_elo),
         "logistic": (yte, p_log),
         "xgboost (raw)": (yte, p_xgb_raw),
         "xgboost (calibrated)": (yte, p_xgb_cal)},
        os.path.join(args.out, "reliability_diagram.png"),
    )
    print(f"\nSaved model + metrics + reliability diagram to {args.out}/")


if __name__ == "__main__":
    main()
