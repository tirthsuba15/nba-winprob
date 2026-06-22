"""
Win Probability Added (WPA) -- the fun, shareable layer.

For each play we measure how much the home team's win probability moved. The
biggest positive swings are the "clutchest" plays of the season. With real
play-by-play (which carries a player name per event) we roll WPA up per player
to answer: who added the most win probability all year?

This is the same idea as nflfastR's EPA/WPA leaderboards -- it turns a calibrated
model into stories ("the 10 most clutch shots of the season").

Usage:
  python wpa.py --data data --top 15
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd
from features import build_dataset, FEATURES


def compute_wpa(df, clf):
    """Add a per-moment win_prob and the change vs the previous moment (WPA)."""
    df = df.sort_values(["game_id", "secs_left"], ascending=[True, False]).copy()
    df["win_prob"] = clf.predict_proba(df[FEATURES].to_numpy(dtype=float))[:, 1]
    df["wpa"] = df.groupby("game_id")["win_prob"].diff().fillna(0.0)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--model", default="outputs/model.json")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--out", default="outputs/wpa_top_plays.csv")
    args = ap.parse_args()

    moments = pd.read_csv(os.path.join(args.data, "moments.csv"))
    games = pd.read_csv(os.path.join(args.data, "games.csv"))
    df = build_dataset(moments, games)

    from xgboost import XGBClassifier
    clf = XGBClassifier()
    clf.load_model(args.model)
    df = compute_wpa(df, clf)

    # Most clutch single moments (biggest swings toward the eventual winner).
    df["swing_for_winner"] = np.where(df["label_home_win"] == 1, df["wpa"], -df["wpa"])
    cols = ["game_id", "season", "period", "secs_left", "score_margin",
            "win_prob", "wpa", "swing_for_winner"]
    if "description" in df.columns:
        cols.insert(2, "description")
    top = df.sort_values("swing_for_winner", ascending=False).head(args.top)[cols]
    print(f"Top {args.top} most clutch moments (largest WP swing for the winner):\n")
    print(top.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    top.to_csv(args.out, index=False)
    print(f"\nsaved {args.out}")

    # Per-player leaderboard (only possible with real PBP that has player names).
    if "player" in df.columns:
        board = (df.groupby("player")["wpa"].sum()
                 .sort_values(ascending=False).head(args.top))
        print("\nTop players by total Win Probability Added:")
        print(board.to_string(float_format=lambda x: f"{x:+.2f}"))
        board.to_csv("outputs/wpa_top_players.csv")
    else:
        print("\n(player-level WPA leaderboard appears once real play-by-play "
              "with player names is fetched -- synthetic data has no players.)")


if __name__ == "__main__":
    main()
