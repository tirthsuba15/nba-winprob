"""
Backtest the over/under calls — CLI ONLY (no dashboard).

Leakage-safe design:
  * Models train on 2023-24 only; every 2024-25 player-game is a test call.
  * The LINE is each player's season-average-to-date, computed strictly from
    games BEFORE the current one (shift(1).expanding().mean()). The first game
    of a player's season has no prior average and is dropped — we never peek at
    the current or future games. (Note: this is computed independently of the
    model's `pts_season_avg` *feature*, which is NaN-filled for cold starts and
    would leak full-season info if reused as the line.)
  * Call "Over" when P(over) > 0.5, then compare to actual points.

Reports:
  1. Total calls, overall hit rate, naive baselines (always-Over, coin flip).
  2. Hit rate by Over-confidence buckets: P(over) > 0.55 / 0.60 / 0.65 / 0.70.
  3. P(over) calibration by decile (predicted vs actual over-rate).
  4. Brier + log-loss on the binary over/under vs a base-rate naive model.

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

SIGMA_DIVISOR = 2.563  # only used for the comment; prob_over owns the math


def compute_lines(df: pd.DataFrame) -> pd.DataFrame:
    """Strict season-average-to-date line per player-game (no leakage)."""
    df = df.sort_values(["player_id", "game_date"]).copy()
    df["line"] = (
        df.groupby(["player_id", "season"])[TARGET]
        .transform(lambda s: s.shift(1).expanding().mean())
    )
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--test-seasons", nargs="+", default=None)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    raw = pd.read_csv(os.path.join(args.data, "player_gamelogs.csv"))
    df = build_features(raw)
    df = compute_lines(df)

    seasons = sorted(df["season"].unique())
    test_seasons = args.test_seasons or [seasons[-1]]
    train_df = df[~df["season"].isin(test_seasons)].copy()
    test_df = df[df["season"].isin(test_seasons)].copy()

    # Mirror model.py: drop cold-start players from TRAIN only.
    train_df = train_df.groupby("player_id").filter(lambda g: len(g) >= MIN_TRAIN_GAMES)

    # Test set: every player-game with a valid (strictly prior) line.
    test_df = test_df.dropna(subset=["line"])

    train_seasons = [s for s in seasons if s not in test_seasons]
    print(f"Train seasons: {train_seasons}  ({len(train_df):,} rows)")
    print(f"Test  seasons: {test_seasons}  ({len(test_df):,} player-games with a valid line)\n")

    X_tr = train_df[FEATURES].to_numpy(dtype=float)
    y_tr = train_df[TARGET].to_numpy(dtype=float)
    X_te = test_df[FEATURES].to_numpy(dtype=float)
    y_te = test_df[TARGET].to_numpy(dtype=float)
    lines = test_df["line"].to_numpy(dtype=float)

    print("Training models on the train season(s) …")
    models = train_models(X_tr, y_tr)
    mean, lo, hi = predict_distribution(models, X_te)

    # P(over) via odds.py for each test call
    p_over = np.array([prob_over(m, l, h, L) for m, l, h, L in zip(mean, lo, hi, lines)])
    actual_over = (y_te > lines).astype(int)
    call_over = p_over > 0.5
    hit = (call_over == (actual_over == 1))

    n = len(p_over)
    overall_hit = float(hit.mean())
    always_over_hit = float((actual_over == 1).mean())   # if you always call Over
    coin_flip = 0.5

    # ── 1. overall ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("1. OVERALL")
    print("=" * 64)
    print(f"  Total calls          : {n:,}")
    print(f"  Overall hit rate     : {overall_hit:.3f}")
    print(f"  Naive always-Over    : {always_over_hit:.3f}  (= test over-rate)")
    print(f"  Naive coin flip      : {coin_flip:.3f}")

    # ── 2. confidence buckets (Over side) ───────────────────────────────────────
    print("\n" + "=" * 64)
    print("2. HIT RATE BY OVER-CONFIDENCE  (rows where P(over) > threshold)")
    print("=" * 64)
    print(f"  {'threshold':>10} {'n':>8} {'hit_rate':>10}")
    conf_buckets = {}
    for t in (0.55, 0.60, 0.65, 0.70):
        m = p_over > t
        cnt = int(m.sum())
        # these are all Over calls, so hit = actually went over
        hr = float((actual_over[m] == 1).mean()) if cnt else float("nan")
        conf_buckets[f">{t:.2f}"] = {"n": cnt, "hit_rate": hr}
        hr_str = f"{hr:.3f}" if cnt else "  —  "
        print(f"  {'P(over)>'+format(t,'.2f'):>10} {cnt:>8,} {hr_str:>10}")

    # ── 3. calibration deciles ───────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("3. P(over) CALIBRATION BY DECILE")
    print("=" * 64)
    print(f"  {'bin':>10} {'n':>8} {'pred_p':>9} {'actual':>9}")
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p_over, bins) - 1, 0, 9)
    calib = []
    for b in range(10):
        m = idx == b
        cnt = int(m.sum())
        if cnt == 0:
            continue
        pred = float(p_over[m].mean())
        act = float(actual_over[m].mean())
        calib.append({"bin": f"{bins[b]:.1f}-{bins[b+1]:.1f}", "n": cnt,
                      "pred_p": pred, "actual_over_rate": act})
        print(f"  {bins[b]:.1f}-{bins[b+1]:.1f}{'':>2} {cnt:>8,} {pred:>9.3f} {act:>9.3f}")

    # ── 4. Brier + log-loss vs naive ─────────────────────────────────────────────
    from sklearn.metrics import brier_score_loss, log_loss
    p_clip = np.clip(p_over, 1e-6, 1 - 1e-6)
    base = float(actual_over.mean())
    base_clip = np.clip(base, 1e-6, 1 - 1e-6)

    brier = float(brier_score_loss(actual_over, p_over))
    ll = float(log_loss(actual_over, p_clip, labels=[0, 1]))
    brier_naive = float(brier_score_loss(actual_over, np.full(n, base)))
    ll_naive = float(log_loss(actual_over, np.full(n, base_clip), labels=[0, 1]))

    print("\n" + "=" * 64)
    print("4. PROBABILISTIC QUALITY (binary over/under)")
    print("=" * 64)
    print(f"  {'':12} {'model':>10} {'naive':>10}")
    print(f"  {'Brier':12} {brier:>10.4f} {brier_naive:>10.4f}")
    print(f"  {'log-loss':12} {ll:>10.4f} {ll_naive:>10.4f}")
    print(f"\n  (naive = always predict the base over-rate {base:.3f})")

    # ── save ──────────────────────────────────────────────────────────────────────
    summary = {
        "train_seasons": train_seasons,
        "test_seasons": test_seasons,
        "n_calls": n,
        "overall_hit_rate": overall_hit,
        "naive_always_over": always_over_hit,
        "naive_coin_flip": coin_flip,
        "confidence_buckets": conf_buckets,
        "calibration_deciles": calib,
        "brier": brier,
        "brier_naive": brier_naive,
        "log_loss": ll,
        "log_loss_naive": ll_naive,
        "base_over_rate": base,
    }
    out_path = os.path.join(args.out, "overunder_backtest.json")
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
