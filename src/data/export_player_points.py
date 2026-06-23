"""
Export the 5-season player-logs parquet into the schema the player-points
pipeline reads (data/player_gamelogs.csv), so model.py / predict.py / app.py /
backtest all train on all 5 seasons from one canonical file.

Usage:
    python src/data/export_player_points.py
    python src/data/export_player_points.py --seasons 2023-24 2024-25
"""
from __future__ import annotations
import argparse
import os
import pandas as pd

from common import DEFAULT_SEASONS, SEASON_TYPES, player_logs_path

# columns the player-points features.py expects
KEEP = ["player_id", "player_name", "team_id", "game_id", "game_date",
        "season", "season_type", "matchup", "min_dec", "fgm", "fga", "fta",
        "tov", "oreb", "pts"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    ap.add_argument("--season-types", nargs="+", default=SEASON_TYPES)
    ap.add_argument("--out", default="data/player_gamelogs.csv")
    args = ap.parse_args()

    frames = []
    for season in args.seasons:
        for st in args.season_types:
            p = player_logs_path(season, st)
            if os.path.exists(p):
                frames.append(pd.read_parquet(p))
    if not frames:
        raise SystemExit("No player_logs parquet found. Run fetch_bulk.py first.")

    df = pd.concat(frames, ignore_index=True)
    df["min_dec"] = pd.to_numeric(df["min"], errors="coerce").fillna(0.0)
    for c in ("fgm", "fga", "fta", "tov", "oreb", "pts"):
        if c not in df.columns:
            df[c] = 0.0
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date.astype(str)

    out = df[[c for c in KEEP if c in df.columns]].copy()
    out = out.sort_values(["player_id", "game_date"]).reset_index(drop=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out):,} player-game rows to {args.out}")
    print(f"Seasons: {sorted(out['season'].unique())} | "
          f"players: {out['player_id'].nunique():,} | "
          f"types: {sorted(out['season_type'].unique())}")


if __name__ == "__main__":
    main()
