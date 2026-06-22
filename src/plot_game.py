"""
The money shot: a live win-probability curve for a single game (ESPN-style).

Loads the trained XGBoost model, replays one game moment-by-moment, and plots
P(home win) over the course of the game with the final result annotated.

Usage:
  python plot_game.py --data data --game SYN0000123
  python plot_game.py --data data            # picks a close, swingy game for you
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd
from features import build_dataset, FEATURES
from team_names import team_name

REG_SECONDS = 48 * 60


def plot_curve(game_df, probs, meta, out_path=None):
    """game_df sorted by time; probs aligned home-win probabilities in [0,1].

    Returns the matplotlib Figure. Saves to out_path if provided.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = (REG_SECONDS - game_df["secs_left"].clip(lower=0).to_numpy()) / 60.0  # minutes elapsed
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhline(0.5, color="#bbb", lw=1, ls="--")
    ax.fill_between(x, 0.5, probs, where=probs >= 0.5, alpha=0.20, color="#1f77b4")
    ax.fill_between(x, 0.5, probs, where=probs < 0.5, alpha=0.20, color="#d62728")
    ax.plot(x, probs, color="#1f77b4", lw=2)
    for q in (12, 24, 36):
        ax.axvline(q, color="#eee", lw=1, zorder=0)

    ax.set_ylim(0, 1)
    ax.set_xlim(0, max(48, x.max()))
    ax.set_xlabel("Minutes elapsed")
    ax.set_ylabel("Home win probability")
    away = team_name(meta["away_team_id"])
    home = team_name(meta["home_team_id"])
    winner = home if meta["home_win"] else away
    ax.set_title(
        f"{away} @ {home}  |  final {int(meta['home_score'])}-{int(meta['away_score'])}  "
        f"({winner} win)"
    )
    note = "synthetic data (illustrative)" if str(meta["game_id"]).startswith("SYN") else str(meta["game_id"])
    ax.text(0.99, 0.02, note, transform=ax.transAxes, ha="right",
            color="#999", fontsize=8)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=130)
        print(f"saved {out_path}")
    return fig


def pick_swingy_game(df, games):
    """Pick a game whose win prob proxy (margin) crosses zero a lot -- fun to look at."""
    g = df.groupby("game_id")["score_margin"].agg(
        lambda s: int((np.sign(s).diff().fillna(0) != 0).sum()))
    gid = g.sort_values().index[-1]
    return gid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--game", default=None)
    ap.add_argument("--model", default="outputs/model.json")
    ap.add_argument("--out", default="outputs/winprob_curve.png")
    args = ap.parse_args()

    moments = pd.read_csv(os.path.join(args.data, "moments.csv"))
    games = pd.read_csv(os.path.join(args.data, "games.csv"))
    df = build_dataset(moments, games)

    gid = args.game or pick_swingy_game(df, games)
    gdf = df[df["game_id"] == gid].sort_values("secs_left", ascending=False)
    if gdf.empty:
        raise SystemExit(f"game {gid} not found")

    from xgboost import XGBClassifier
    clf = XGBClassifier()
    clf.load_model(args.model)
    probs = clf.predict_proba(gdf[FEATURES].to_numpy(dtype=float))[:, 1]

    meta = games[games["game_id"] == gid].iloc[0].to_dict()
    plot_curve(gdf, probs, meta, out_path=args.out)


if __name__ == "__main__":
    main()
