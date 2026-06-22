"""
Bulk (cheap) fetches for the 5-season data layer: games, player logs, team logs.

Each is a single nba_api call per (season, season_type) — ~15-20 calls total for
5 seasons x {Regular Season, Playoffs}. Output is parquet, partitioned by season.
Re-runs SKIP any partition that already exists (use --force to refetch).

Usage:
    python src/data/fetch_bulk.py                      # all default seasons, RS+PO
    python src/data/fetch_bulk.py --seasons 2024-25
    python src/data/fetch_bulk.py --force
"""
from __future__ import annotations
import argparse
import os
import pandas as pd

from common import (
    DEFAULT_SEASONS, SEASON_TYPES, retry, sleep, ensure_dirs, write_parquet,
    games_path, player_logs_path, team_logs_path,
)


# ── games (team rows -> one row per game) ─────────────────────────────────────────
def fetch_games(season: str, season_type: str) -> pd.DataFrame:
    from nba_api.stats.endpoints import leaguegamelog
    log = retry(lambda: leaguegamelog.LeagueGameLog(
        season=season, season_type_all_star=season_type
    ).get_data_frames()[0])

    rows = {}
    for _, r in log.iterrows():
        gid = r["GAME_ID"]
        rec = rows.setdefault(gid, {"game_id": gid, "game_date": r["GAME_DATE"]})
        side = "home" if "vs." in r["MATCHUP"] else "away"
        rec[f"{side}_team_id"] = int(r["TEAM_ID"])
        rec[f"{side}_score"] = int(r["PTS"])
    out = []
    for gid, rec in rows.items():
        if {"home_team_id", "away_team_id", "home_score", "away_score"} <= rec.keys():
            rec["home_win"] = int(rec["home_score"] > rec["away_score"])
            rec["season"] = season
            rec["season_type"] = season_type
            out.append(rec)
    return pd.DataFrame(out).sort_values("game_date").reset_index(drop=True)


# ── player game logs (minutes, availability inputs) ───────────────────────────────
def fetch_player_logs(season: str, season_type: str) -> pd.DataFrame:
    from nba_api.stats.endpoints import playergamelogs
    df = retry(lambda: playergamelogs.PlayerGameLogs(
        season_nullable=season, season_type_nullable=season_type
    ).get_data_frames()[0])
    df = df.copy()
    df["season"] = season
    df["season_type"] = season_type
    # Normalize the columns we care about; keep the rest too (parquet is cheap).
    df.columns = [c.lower() for c in df.columns]
    return df


# ── team game logs (pace / defensive-rating inputs) ───────────────────────────────
def fetch_team_logs(season: str, season_type: str) -> pd.DataFrame:
    from nba_api.stats.endpoints import teamgamelogs
    df = retry(lambda: teamgamelogs.TeamGameLogs(
        season_nullable=season, season_type_nullable=season_type
    ).get_data_frames()[0])
    df = df.copy()
    df["season"] = season
    df["season_type"] = season_type
    df.columns = [c.lower() for c in df.columns]
    return df


_FETCHERS = {
    "games": (fetch_games, games_path),
    "player_logs": (fetch_player_logs, player_logs_path),
    "team_logs": (fetch_team_logs, team_logs_path),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    ap.add_argument("--season-types", nargs="+", default=SEASON_TYPES)
    ap.add_argument("--only", nargs="+", default=list(_FETCHERS),
                    choices=list(_FETCHERS), help="subset of datasets to fetch")
    ap.add_argument("--force", action="store_true", help="refetch even if parquet exists")
    ap.add_argument("--sleep", type=float, default=0.6)
    args = ap.parse_args()

    ensure_dirs()
    n_fetched = n_skipped = 0
    for season in args.seasons:
        for st in args.season_types:
            for name in args.only:
                fetch_fn, path_fn = _FETCHERS[name]
                path = path_fn(season, st)
                if os.path.exists(path) and not args.force:
                    print(f"  skip (cached): {path}")
                    n_skipped += 1
                    continue
                print(f"  fetch: {name} {season} {st} …", end=" ", flush=True)
                try:
                    df = fetch_fn(season, st)
                    write_parquet(df, path)
                    print(f"{len(df):,} rows -> {path}")
                    n_fetched += 1
                except Exception as e:
                    print(f"FAILED: {e}")
                sleep(args.sleep)

    print(f"\nDone. Fetched {n_fetched}, skipped {n_skipped} cached partitions.")


if __name__ == "__main__":
    main()
