"""
Fetch team rosters (for player POSITION) via CommonTeamRoster — one call per
(team, season), ~150 calls for 5 seasons. Cached to parquet; skips if present.

Position is needed for the "opponent defense vs the player's position" feature.

Usage:
    python src/data/fetch_rosters.py
    python src/data/fetch_rosters.py --seasons 2024-25 --force
"""
from __future__ import annotations
import argparse
import os
import pandas as pd

from common import DEFAULT_SEASONS, RAW, retry, sleep, ensure_dirs, write_parquet

ROSTER_DIR = os.path.join(RAW, "rosters")


def roster_path(season: str) -> str:
    return os.path.join(ROSTER_DIR, f"season={season}.parquet")


def fetch_season_rosters(season: str) -> pd.DataFrame:
    from nba_api.stats.static import teams as static_teams
    from nba_api.stats.endpoints import commonteamroster
    rows = []
    for t in static_teams.get_teams():
        df = retry(lambda tid=t["id"]: commonteamroster.CommonTeamRoster(
            team_id=tid, season=season).get_data_frames()[0])
        if not df.empty:
            df = df.rename(columns={"PLAYER_ID": "player_id", "POSITION": "position"})
            df["team_id"] = t["id"]
            df["season"] = season
            rows.append(df[["player_id", "position", "team_id", "season"]])
        sleep(0.6)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    ensure_dirs()
    os.makedirs(ROSTER_DIR, exist_ok=True)

    for season in args.seasons:
        path = roster_path(season)
        if os.path.exists(path) and not args.force:
            print(f"  skip (cached): {path}")
            continue
        print(f"  fetch rosters {season} (30 teams) …", end=" ", flush=True)
        df = fetch_season_rosters(season)
        write_parquet(df, path)
        print(f"{len(df):,} players -> {path}")


if __name__ == "__main__":
    main()
