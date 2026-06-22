"""
Overnight, RESUMABLE play-by-play fetch (the heavy job).

For every game in the cached games parquet, pull PlayByPlayV3 and write one
parquet per game. Designed to be killed and resumed:
  * A game is SKIPPED if its parquet already exists, or it's marked done/failed
    in the resume log (data/.fetch_log/pbp_progress.jsonl).
  * Progress is appended to the log after every game, so Ctrl-C is safe.

Run (after fetch_bulk.py has produced the games partitions):
    python src/data/fetch_pbp.py                       # all cached seasons
    python src/data/fetch_pbp.py --seasons 2024-25
    nohup python src/data/fetch_pbp.py > pbp.out 2>&1 &   # overnight

Re-run anytime — it continues where it left off.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import time
import pandas as pd

from common import (
    DEFAULT_SEASONS, SEASON_TYPES, GAMES_DIR, LOG_DIR, retry, sleep,
    ensure_dirs, write_parquet, pbp_path, games_path,
)

PROGRESS_LOG = os.path.join(LOG_DIR, "pbp_progress.jsonl")


def _clock_to_secs_left(period: int, clock) -> int:
    """ISO8601 'PT11M34.00S' -> seconds remaining in regulation. OT -> 0."""
    try:
        s = str(clock)
        mins = int(s.split("PT")[1].split("M")[0])
        secs = float(s.split("M")[1].split("S")[0])
        in_period = mins * 60 + secs
    except Exception:
        in_period = 0
    if period <= 4:
        return int(max(0, (4 - period) * 720 + in_period))
    return 0


def fetch_one(game_id: str, season: str, home_win: int) -> pd.DataFrame:
    from nba_api.stats.endpoints import playbyplayv3
    pbp = retry(lambda: playbyplayv3.PlayByPlayV3(game_id=game_id).get_data_frames()[0])
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
            "score_margin": last_h - last_a,
            "description": e.get("description", "") or "",
            "player": e.get("playerName", "") or "",
            "label_home_win": int(home_win),
        })
    return pd.DataFrame(recs)


def _load_done() -> set[str]:
    """Game ids already handled (from the resume log + existing parquet files)."""
    done = set()
    if os.path.exists(PROGRESS_LOG):
        for line in open(PROGRESS_LOG):
            try:
                rec = json.loads(line)
                if rec.get("status") in ("done", "empty"):
                    done.add(rec["game_id"])
            except Exception:
                continue
    # also treat any existing parquet as done (belt + suspenders)
    for p in glob.glob(os.path.join(GAMES_DIR, "..", "pbp", "season=*", "game_id=*.parquet")):
        gid = os.path.basename(p).replace("game_id=", "").replace(".parquet", "")
        done.add(gid)
    return done


def _log(rec: dict):
    with open(PROGRESS_LOG, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


def _all_games(seasons, season_types) -> pd.DataFrame:
    frames = []
    for season in seasons:
        for st in season_types:
            path = games_path(season, st)
            if os.path.exists(path):
                frames.append(pd.read_parquet(path))
    if not frames:
        raise SystemExit("No games parquet found. Run fetch_bulk.py first.")
    return pd.concat(frames, ignore_index=True).drop_duplicates("game_id")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    ap.add_argument("--season-types", nargs="+", default=SEASON_TYPES)
    ap.add_argument("--sleep", type=float, default=0.6)
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N new games (handy for a test run)")
    args = ap.parse_args()

    ensure_dirs()
    games = _all_games(args.seasons, args.season_types)
    done = _load_done()
    todo = games[~games["game_id"].isin(done)].reset_index(drop=True)

    total = len(games)
    print(f"{total:,} games total | {len(done):,} already done | {len(todo):,} to fetch")
    if args.limit:
        todo = todo.head(args.limit)
        print(f"  (limited to {len(todo)} this run)")

    started = time.time()
    n_done = 0
    for i, g in todo.iterrows():
        gid = g["game_id"]
        try:
            df = fetch_one(gid, g["season"], int(g["home_win"]))
            if df.empty:
                _log({"game_id": gid, "season": g["season"], "status": "empty"})
            else:
                write_parquet(df, pbp_path(g["season"], gid))
                _log({"game_id": gid, "season": g["season"], "status": "done",
                      "rows": int(len(df))})
            n_done += 1
        except Exception as e:
            _log({"game_id": gid, "season": g["season"], "status": "failed",
                  "error": str(e)[:200]})
            print(f"    FAILED {gid}: {e}")

        if n_done and n_done % 25 == 0:
            rate = n_done / (time.time() - started)
            remaining = (len(todo) - n_done) / rate if rate else 0
            print(f"    {n_done}/{len(todo)} this run "
                  f"| {rate*60:.0f} games/min | ~{remaining/3600:.1f} h left")
        sleep(args.sleep)

    print(f"\nRun complete: {n_done} games fetched this run. "
          f"Re-run to continue; progress in {PROGRESS_LOG}")


if __name__ == "__main__":
    main()
