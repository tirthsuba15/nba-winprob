"""
Fetch per-player game logs for one or more seasons via nba_api and write
data/player_gamelogs.csv.

Uses PlayerGameLogs (bulk endpoint) — one API call per season, so the full
2023-24 + 2024-25 pull is just 2 requests. Much faster than per-player calls.

Output schema (data/player_gamelogs.csv):
    player_id, player_name, team_id, game_id, game_date, season,
    matchup, wl, min_sec, min_dec,
    fgm, fga, fta, tov, oreb, pts

Usage:
    python src/player_points/fetch_gamelogs.py --seasons 2023-24 2024-25
"""
from __future__ import annotations
import argparse
import os
import sys
import time
import pandas as pd

_KEEP = [
    "PLAYER_ID", "PLAYER_NAME", "TEAM_ID",
    "GAME_ID", "GAME_DATE", "SEASON_YEAR",
    "MATCHUP", "WL",
    "MIN", "FGM", "FGA", "FTA", "TOV", "OREB", "PTS",
]
_RENAME = {
    "PLAYER_ID": "player_id",
    "PLAYER_NAME": "player_name",
    "TEAM_ID": "team_id",
    "GAME_ID": "game_id",
    "GAME_DATE": "game_date",
    "SEASON_YEAR": "season",
    "MATCHUP": "matchup",
    "WL": "wl",
    "MIN": "min_str",
    "FGM": "fgm",
    "FGA": "fga",
    "FTA": "fta",
    "TOV": "tov",
    "OREB": "oreb",
    "PTS": "pts",
}


def _retry(fn, tries=4, base=0.8):
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            time.sleep(base * (2 ** i))
    raise last


def _parse_min(s) -> float:
    """'35:22' -> 35.37, handles None/empty/already-float."""
    if s is None or s == "":
        return 0.0
    try:
        return float(s)
    except (ValueError, TypeError):
        pass
    try:
        parts = str(s).split(":")
        return float(parts[0]) + float(parts[1]) / 60
    except Exception:
        return 0.0


def fetch_season(season: str) -> pd.DataFrame:
    from nba_api.stats.endpoints import playergamelogs
    df = _retry(lambda: playergamelogs.PlayerGameLogs(
        season_nullable=season,
        season_type_nullable="Regular Season",
    ).get_data_frames()[0])

    # Keep only columns we need (tolerate missing optional cols)
    available = [c for c in _KEEP if c in df.columns]
    df = df[available].rename(columns={k: v for k, v in _RENAME.items() if k in available})

    df["min_dec"] = df["min_str"].apply(_parse_min)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date.astype(str)
    df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", required=True,
                    help="e.g. 2023-24 2024-25")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    try:
        import nba_api  # noqa: F401
    except ImportError:
        sys.exit("nba_api not installed.")

    os.makedirs(args.out, exist_ok=True)
    frames = []
    for season in args.seasons:
        print(f"Fetching {season} player game logs …")
        df = fetch_season(season)
        print(f"  {len(df):,} player-game rows")
        frames.append(df)
        time.sleep(1.0)

    out = pd.concat(frames, ignore_index=True)
    path = os.path.join(args.out, "player_gamelogs.csv")
    out.to_csv(path, index=False)
    print(f"\nWrote {len(out):,} rows to {path}")
    print(f"Players: {out['player_id'].nunique():,}  |  Seasons: {sorted(out['season'].unique())}")


if __name__ == "__main__":
    main()
