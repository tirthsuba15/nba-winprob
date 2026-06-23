"""
HISTORICAL player-availability features — the leakage-safe injury proxy.

We never scrape injury reports. Instead we infer who PLAYED vs SAT from the box
scores themselves: a rotation player on a team with no game-log row that night
was inactive (injured / rested / DNP). All of this is known before tip-off, so
it is leakage-safe for training.

Per (team, season) we walk games in date order, maintaining each player's
season-to-date averages using ONLY prior games. For each game we then derive:

  * rotation_to_date   = players with >= MIN_GP prior games AND >= ROTATION_MIN
                         minutes/game to date (the expected rotation)
  * appeared           = players with a log row this game
  * inactive rotation  = rotation_to_date − appeared

Emitted per player-game:
  n_rotation_out      how many expected-rotation teammates sat
  top2_teammate_out   1 if either of the team's top-2 scorers (to date,
                      excluding this player) sat
  team_rotation_size  size of the expected rotation that night

Merge onto the points-model feature matrix by (player_id, game_id).
"""
from __future__ import annotations
import pandas as pd

ROTATION_MIN = 20.0   # minutes/game to count as a rotation player
MIN_GP = 3            # need this many prior games before we trust the average


def _to_min(v) -> float:
    if v is None or v == "":
        return 0.0
    try:
        return float(v)
    except (ValueError, TypeError):
        pass
    try:
        mm, ss = str(v).split(":")[:2]
        return float(mm) + float(ss) / 60.0
    except Exception:
        return 0.0


def build_availability(player_logs: pd.DataFrame) -> pd.DataFrame:
    """Return per player-game availability features (leakage-safe)."""
    df = player_logs.copy()
    df.columns = [c.lower() for c in df.columns]
    if "min_dec" in df.columns:
        df["min_dec"] = pd.to_numeric(df["min_dec"], errors="coerce").fillna(0.0)
    elif "min" in df.columns:
        df["min_dec"] = df["min"].apply(_to_min)
    else:
        df["min_dec"] = 0.0
    df["game_date"] = pd.to_datetime(df["game_date"])
    for col in ("player_id", "team_id"):
        df[col] = df[col].astype(int)
    if "pts" not in df.columns:
        df["pts"] = 0.0

    out_rows = []
    keys = ["team_id", "season"]
    for (team_id, season), tg in df.groupby(keys):
        # running per-player to-date accumulators (prior games only)
        gp: dict[int, int] = {}
        sum_min: dict[int, float] = {}
        sum_pts: dict[int, float] = {}

        for game_id, ggame in tg.sort_values("game_date").groupby("game_date", sort=True):
            # all rows in this team's slate on this date share one game_id
            gid = ggame["game_id"].iloc[0]
            appeared = set(ggame["player_id"].tolist())

            # to-date averages BEFORE folding in tonight's game
            avg_min = {p: sum_min[p] / gp[p] for p in gp if gp[p] > 0}
            avg_pts = {p: sum_pts[p] / gp[p] for p in gp if gp[p] > 0}

            rotation = {p for p in gp
                        if gp[p] >= MIN_GP and avg_min.get(p, 0.0) >= ROTATION_MIN}
            # top-2 scorers to date among players with enough games
            ranked = sorted(
                (p for p in gp if gp[p] >= MIN_GP),
                key=lambda p: avg_pts.get(p, 0.0), reverse=True,
            )
            top2 = set(ranked[:2])

            for p in appeared:
                others_rotation_out = rotation - appeared - {p}
                top2_out = int(any((t not in appeared) for t in (top2 - {p})))
                out_rows.append({
                    "player_id": p,
                    "game_id": gid,
                    "n_rotation_out": len(others_rotation_out),
                    "top2_teammate_out": top2_out,
                    "team_rotation_size": len(rotation),
                })

            # now fold tonight's game into the to-date accumulators
            for _, r in ggame.iterrows():
                p = int(r["player_id"])
                gp[p] = gp.get(p, 0) + 1
                sum_min[p] = sum_min.get(p, 0.0) + float(r["min_dec"])
                sum_pts[p] = sum_pts.get(p, 0.0) + float(r["pts"])

    return pd.DataFrame(out_rows)


def main():
    import argparse
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--player-logs", default=None,
                    help="parquet/csv of player logs; default reads data/player_gamelogs.csv")
    ap.add_argument("--out", default="data/availability.parquet")
    args = ap.parse_args()

    src = args.player_logs or os.path.join(args.data, "player_gamelogs.csv")
    logs = pd.read_parquet(src) if src.endswith(".parquet") else pd.read_csv(src)
    feats = build_availability(logs)
    feats.to_parquet(args.out, index=False)
    print(f"Wrote {len(feats):,} player-game availability rows to {args.out}")
    print(feats.describe(include="all").to_string())


if __name__ == "__main__":
    main()
