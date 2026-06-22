"""
Export the parquet data layer into the CSV schema the win-prob pipeline expects
(data/games.csv + data/moments.csv), so model.py / plot_game.py / wpa.py train on
all 5 seasons without changes.

Usage (after fetch_bulk.py + fetch_pbp.py):
    python src/data/export_winprob.py
    python src/data/export_winprob.py --seasons 2023-24 2024-25
"""
from __future__ import annotations
import argparse
import glob
import os
import pandas as pd

from common import DEFAULT_SEASONS, SEASON_TYPES, PBP_DIR, games_path

GAMES_COLS = ["game_id", "season", "game_date", "home_team_id", "away_team_id",
              "home_score", "away_score", "home_win"]
MOMENT_COLS = ["game_id", "season", "period", "secs_left", "score_margin",
               "description", "player", "label_home_win"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    ap.add_argument("--season-types", nargs="+", default=SEASON_TYPES)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    # games
    gframes = []
    for season in args.seasons:
        for st in args.season_types:
            p = games_path(season, st)
            if os.path.exists(p):
                gframes.append(pd.read_parquet(p))
    if not gframes:
        raise SystemExit("No games parquet found. Run fetch_bulk.py first.")
    games = pd.concat(gframes, ignore_index=True).drop_duplicates("game_id")

    # moments (pbp) — one parquet per game
    mframes = []
    for season in args.seasons:
        for p in glob.glob(os.path.join(PBP_DIR, f"season={season}", "game_id=*.parquet")):
            mframes.append(pd.read_parquet(p))
    if not mframes:
        raise SystemExit("No pbp parquet found. Run fetch_pbp.py first.")
    moments = pd.concat(mframes, ignore_index=True)

    # keep only moments whose game is in the games set
    moments = moments[moments["game_id"].isin(games["game_id"])]

    games[GAMES_COLS].to_csv(os.path.join(args.out, "games.csv"), index=False)
    moments[MOMENT_COLS].to_csv(os.path.join(args.out, "moments.csv"), index=False)
    print(f"Wrote {len(games):,} games and {len(moments):,} moments to {args.out}/")
    print(f"Seasons: {sorted(games['season'].unique())}")


if __name__ == "__main__":
    main()
