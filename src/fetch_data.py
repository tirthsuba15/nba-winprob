"""
Fetch real NBA data via nba_api and write the schema the pipeline expects.

Pulls, for each season:
  * the regular-season game list (to get final scores + home/away)
  * play-by-play for each game (to get the score margin over time)

Outputs (same schema as synth.py, so model.py / plot_game.py / wpa.py just work):
  data/games.csv    one row per game: final result, teams, date
  data/moments.csv  many rows per game: period, secs_left, score_margin, label,
                    plus `description` and `player` for the WPA leaderboard

This MUST run on your laptop (stats.nba.com blocks datacenter/CI IPs). It is
deliberately polite: small sleeps + retries so you don't get rate-limited.

Examples:
  python fetch_data.py --seasons 2024-25 --games 200
  python fetch_data.py --seasons 2020-21 2021-22 2022-23 2023-24 2024-25 --games 400
"""
from __future__ import annotations
import argparse
import os
import re
import time
import sys
import numpy as np
import pandas as pd

REG_SECONDS = 48 * 60


def _sleep(s=0.6):
    time.sleep(s)


def _retry(fn, tries=4, base=0.8):
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:           # network hiccup / rate limit
            last = e
            time.sleep(base * (2 ** i))
    raise last


def get_games(season):
    """Return DataFrame: game_id, game_date, home_team_id, away_team_id,
    home_score, away_score, home_win  (regular season)."""
    from nba_api.stats.endpoints import leaguegamelog
    log = _retry(lambda: leaguegamelog.LeagueGameLog(
        season=season, season_type_all_star="Regular Season"
    ).get_data_frames()[0])

    rows = {}
    for _, r in log.iterrows():
        gid = r["GAME_ID"]
        rec = rows.setdefault(gid, {"game_id": gid, "game_date": r["GAME_DATE"]})
        is_home = "vs." in r["MATCHUP"]
        side = "home" if is_home else "away"
        rec[f"{side}_team_id"] = int(r["TEAM_ID"])
        rec[f"{side}_score"] = int(r["PTS"])
    out = []
    for gid, rec in rows.items():
        if {"home_team_id", "away_team_id", "home_score", "away_score"} <= rec.keys():
            rec["home_win"] = int(rec["home_score"] > rec["away_score"])
            rec["season"] = season
            out.append(rec)
    df = pd.DataFrame(out).sort_values("game_date").reset_index(drop=True)
    return df


def _clock_to_secs_left(period, clock):
    """V3 clock is ISO-8601 duration like 'PT11M34.00S'. Return seconds
    remaining in regulation; overtime clamps to 0."""
    m = re.match(r"PT0*(\d+)M0*([\d.]+)S", str(clock))
    if m:
        in_period = int(m.group(1)) * 60 + float(m.group(2))
    else:                                   # fallback for 'MM:SS'
        try:
            mm, ss = str(clock).split(":")
            in_period = int(mm) * 60 + float(ss)
        except Exception:
            in_period = 0.0
    if period <= 4:
        return int(max(0, (4 - period) * 720 + in_period))
    return 0


def get_moments(game_id, season, home_win):
    """Play-by-play (V3) -> moment rows.

    PlayByPlayV2 is dead (the NBA API returns empty JSON for it -- see nba_api
    issue #591), so we use PlayByPlayV3. V3 gives scoreHome/scoreAway directly,
    so margin = home - away with no sign ambiguity. It also carries a clean
    `description` and `playerName` per event for the WPA leaderboard.
    """
    from nba_api.stats.endpoints import playbyplayv3
    pbp = _retry(lambda: playbyplayv3.PlayByPlayV3(
        game_id=game_id).get_data_frames()[0])
    if pbp.empty:
        return pbp

    recs = []
    last_h = last_a = 0
    for _, e in pbp.iterrows():
        try:
            last_h = int(e["scoreHome"])
        except (ValueError, TypeError):
            pass
        try:
            last_a = int(e["scoreAway"])
        except (ValueError, TypeError):
            pass
        period = int(e["period"])
        recs.append({
            "game_id": game_id,
            "season": season,
            "period": period,
            "secs_left": _clock_to_secs_left(period, e.get("clock")),
            "score_margin": last_h - last_a,   # already home - away
            "description": e.get("description", "") or "",
            "player": e.get("playerName", "") or "",
            "label_home_win": int(home_win),
        })
    return pd.DataFrame(recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", required=True,
                    help="e.g. 2024-25 or 2020-21 2021-22 ...")
    ap.add_argument("--games", type=int, default=200,
                    help="max games per season (sampled evenly across the season)")
    ap.add_argument("--out", default="data")
    ap.add_argument("--sleep", type=float, default=0.6)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    try:
        import nba_api  # noqa: F401
    except ImportError:
        sys.exit("nba_api not installed. Run: pip install nba_api")

    all_games, all_moments = [], []
    for season in args.seasons:
        print(f"\n=== {season} ===")
        games = get_games(season)
        print(f"  {len(games)} regular-season games found")
        if args.games and len(games) > args.games:
            idx = np.linspace(0, len(games) - 1, args.games).astype(int)
            games = games.iloc[idx].reset_index(drop=True)
            print(f"  sampling {len(games)} of them")
        all_games.append(games)

        for i, g in games.iterrows():
            try:
                mdf = get_moments(g["game_id"], season, g["home_win"])
                if not mdf.empty:
                    all_moments.append(mdf)
            except Exception as e:
                print(f"    skip {g['game_id']}: {e}")
            if (i + 1) % 25 == 0:
                print(f"    {i+1}/{len(games)} games done")
            _sleep(args.sleep)

    games_df = pd.concat(all_games, ignore_index=True)
    moments_df = pd.concat(all_moments, ignore_index=True)
    keep = ["game_id", "season", "game_date", "home_team_id", "away_team_id",
            "home_score", "away_score", "home_win"]
    games_df[keep].to_csv(os.path.join(args.out, "games.csv"), index=False)
    moments_df.to_csv(os.path.join(args.out, "moments.csv"), index=False)
    print(f"\nWrote {len(games_df):,} games and {len(moments_df):,} moments "
          f"to {args.out}/")


if __name__ == "__main__":
    main()
