"""
Backtest the over/under calls — CLI ONLY (no dashboard).

5-season, leakage-safe, expanding-train design:
  * Models train on the earlier seasons (2020-21 … 2023-24); every 2024-25
    player-game is a test call. Same test season as the original 1-season run,
    so the headline 0.542 is directly comparable.
  * The LINE is each player's season-average-to-date (shift(1).expanding().mean());
    first game of a season has no prior average and is dropped.
  * Call "Over" when P(over) > 0.5, then compare to actual points.

P(over) ISOTONIC CALIBRATION (item 4): the raw quantile interval is slightly too
narrow, so P(over) is overconfident. We fit IsotonicRegression on the TRAIN
calls only (train P(over) -> train actual-over) and apply it to the test calls.
No test data touches the fit.

Reports overall hit rate, Over-confidence buckets, calibration deciles, and
Brier/log-loss vs naive — side by side with the previous run.

Run:
    python src/player_points/backtest_overunder.py --data data
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from features import build_features, FEATURES, TARGET
from odds import prob_over
from model import train_models, predict_distribution, MIN_TRAIN_GAMES


def compute_lines(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["player_id", "game_date"]).copy()
    df["line"] = (
        df.groupby(["player_id", "season"])[TARGET]
        .transform(lambda s: s.shift(1).expanding().mean())
    )
    return df


def _p_over_vec(mean, lo, hi, lines):
    return np.array([prob_over(m, l, h, L) for m, l, h, L in zip(mean, lo, hi, lines)])


def _metrics(p, actual_over):
    from sklearn.metrics import brier_score_loss, log_loss
    n = len(p)
    call_over = p > 0.5
    overall = float((call_over == (actual_over == 1)).mean())
    p_clip = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "overall_hit_rate": overall,
        "brier": float(brier_score_loss(actual_over, p)),
        "log_loss": float(log_loss(actual_over, p_clip, labels=[0, 1])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--test-seasons", nargs="+", default=["2024-25"])
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    out_path = os.path.join(args.out, "overunder_backtest.json")
    old = json.load(open(out_path)) if os.path.exists(out_path) else None

    raw = pd.read_csv(os.path.join(args.data, "player_gamelogs.csv"))
    print("Building 5-season features (rolling + availability + opp team logs) …")
    df = build_features(raw)
    df = compute_lines(df)

    seasons = sorted(df["season"].unique())
    test_seasons = args.test_seasons
    train_df = df[~df["season"].isin(test_seasons)].copy()
    test_df = df[df["season"].isin(test_seasons)].dropna(subset=["line"]).copy()
    train_df = train_df.dropna(subset=["line"])
    train_df = train_df.groupby("player_id").filter(lambda g: len(g) >= MIN_TRAIN_GAMES)

    train_seasons = [s for s in seasons if s not in test_seasons]
    print(f"Train seasons: {train_seasons}  ({len(train_df):,} rows)")
    print(f"Test  seasons: {test_seasons}  ({len(test_df):,} player-games with a valid line)\n")

    X_tr = train_df[FEATURES].to_numpy(dtype=float)
    y_tr = train_df[TARGET].to_numpy(dtype=float)
    line_tr = train_df["line"].to_numpy(dtype=float)
    X_te = test_df[FEATURES].to_numpy(dtype=float)
    y_te = test_df[TARGET].to_numpy(dtype=float)
    line_te = test_df["line"].to_numpy(dtype=float)

    print("Training models on the train seasons …")
    models = train_models(X_tr, y_tr)

    m_tr, lo_tr, hi_tr = predict_distribution(models, X_tr)
    m_te, lo_te, hi_te = predict_distribution(models, X_te)

    p_tr = _p_over_vec(m_tr, lo_tr, hi_tr, line_tr)
    p_raw = _p_over_vec(m_te, lo_te, hi_te, line_te)
    over_tr = (y_tr > line_tr).astype(int)
    actual_over = (y_te > line_te).astype(int)

    # ── isotonic calibration of P(over), fit on TRAIN only ───────────────────────
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_tr, over_tr)
    p_cal = np.clip(iso.predict(p_raw), 0.0, 1.0)

    raw_m = _metrics(p_raw, actual_over)
    cal_m = _metrics(p_cal, actual_over)
    n = len(p_cal)
    always_over = float((actual_over == 1).mean())
    base = float(actual_over.mean())

    from sklearn.metrics import brier_score_loss, log_loss
    base_clip = np.clip(base, 1e-6, 1 - 1e-6)
    brier_naive = float(brier_score_loss(actual_over, np.full(n, base)))
    ll_naive = float(log_loss(actual_over, np.full(n, base_clip), labels=[0, 1]))

    # ── 1. overall, side by side ─────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("1. OVERALL  (NEW 5-season+calibrated  vs  OLD 1-season)")
    print("=" * 72)
    old_hit = old.get("overall_hit_rate") if old else None
    print(f"  {'metric':24} {'OLD':>10} {'NEW raw':>10} {'NEW cal':>10}")
    print(f"  {'calls':24} {old.get('n_calls','—') if old else '—':>10} "
          f"{n:>10,} {n:>10,}")
    print(f"  {'overall hit rate':24} {fmt(old_hit):>10} "
          f"{raw_m['overall_hit_rate']:>10.3f} {cal_m['overall_hit_rate']:>10.3f}")
    print(f"  {'naive always-Over':24} {fmt(old.get('naive_always_over') if old else None):>10} "
          f"{always_over:>10.3f} {always_over:>10.3f}")
    print(f"  {'naive coin flip':24} {'0.500':>10} {'0.500':>10} {'0.500':>10}")

    # ── 2. confidence buckets (calibrated, two-sided) ───────────────────────────
    print("\n" + "=" * 72)
    print("2. HIT RATE BY CONFIDENCE  (calibrated; |P-0.5|, the side we'd call)")
    print("=" * 72)
    print(f"  {'min confidence':>14} {'n':>8} {'hit_rate':>10}")
    conf = max(p_cal, 1 - p_cal) if False else np.maximum(p_cal, 1 - p_cal)
    call_over = p_cal > 0.5
    hit = (call_over == (actual_over == 1))
    conf_buckets = {}
    for t in (0.55, 0.60, 0.65, 0.70):
        m = conf > t
        cnt = int(m.sum())
        hr = float(hit[m].mean()) if cnt else float("nan")
        conf_buckets[f">{t:.2f}"] = {"n": cnt, "hit_rate": hr}
        print(f"  {('>'+format(t,'.2f')):>14} {cnt:>8,} {(f'{hr:.3f}' if cnt else '—'):>10}")

    # ── 3. calibration deciles (raw vs calibrated) ──────────────────────────────
    print("\n" + "=" * 72)
    print("3. P(over) CALIBRATION BY DECILE  (pred vs actual over-rate)")
    print("=" * 72)
    print(f"  {'bin':>10} | {'RAW n':>7} {'pred':>6} {'actual':>7} | {'CAL n':>7} {'pred':>6} {'actual':>7}")
    bins = np.linspace(0, 1, 11)
    calib = []
    for b in range(10):
        rr = _bin_stats(p_raw, actual_over, bins, b)
        cc = _bin_stats(p_cal, actual_over, bins, b)
        if rr["n"] == 0 and cc["n"] == 0:
            continue
        calib.append({"bin": f"{bins[b]:.1f}-{bins[b+1]:.1f}", "raw": rr, "cal": cc})
        print(f"  {bins[b]:.1f}-{bins[b+1]:.1f} | "
              f"{rr['n']:>7,} {fmt(rr['pred']):>6} {fmt(rr['actual']):>7} | "
              f"{cc['n']:>7,} {fmt(cc['pred']):>6} {fmt(cc['actual']):>7}")

    # ── 4. Brier + log-loss ──────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("4. PROBABILISTIC QUALITY  (binary over/under)")
    print("=" * 72)
    old_brier = old.get("brier") if old else None
    old_ll = old.get("log_loss") if old else None
    print(f"  {'':12} {'OLD':>10} {'NEW raw':>10} {'NEW cal':>10} {'naive':>10}")
    print(f"  {'Brier':12} {fmt(old_brier):>10} {raw_m['brier']:>10.4f} "
          f"{cal_m['brier']:>10.4f} {brier_naive:>10.4f}")
    print(f"  {'log-loss':12} {fmt(old_ll):>10} {raw_m['log_loss']:>10.4f} "
          f"{cal_m['log_loss']:>10.4f} {ll_naive:>10.4f}")
    print(f"\n  (naive = always predict the base over-rate {base:.3f})")

    # ── save (new becomes the reference; keep the old inline) ────────────────────
    summary = {
        "train_seasons": train_seasons,
        "test_seasons": test_seasons,
        "n_calls": n,
        "overall_hit_rate": cal_m["overall_hit_rate"],
        "overall_hit_rate_raw": raw_m["overall_hit_rate"],
        "naive_always_over": always_over,
        "naive_coin_flip": 0.5,
        "confidence_buckets": conf_buckets,
        "calibration_deciles": calib,
        "brier": cal_m["brier"],
        "brier_raw": raw_m["brier"],
        "brier_naive": brier_naive,
        "log_loss": cal_m["log_loss"],
        "log_loss_raw": raw_m["log_loss"],
        "log_loss_naive": ll_naive,
        "base_over_rate": base,
        "n_features": len(FEATURES),
        "features": FEATURES,
        "previous_run": {k: old.get(k) for k in
                         ("overall_hit_rate", "brier", "log_loss", "n_calls")} if old else None,
    }
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nSaved {out_path}")


def _bin_stats(p, y, bins, b):
    idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
    m = idx == b
    cnt = int(m.sum())
    if cnt == 0:
        return {"n": 0, "pred": None, "actual": None}
    return {"n": cnt, "pred": float(p[m].mean()), "actual": float(y[m].mean())}


def fmt(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) and x is not None else "—"


if __name__ == "__main__":
    main()
