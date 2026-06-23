"""
Leakage-safe features for the player-points model (5-season version).

Every feature is computable BEFORE tip-off:
  - rolling stats use shift(1) so the current game is never in its own window
  - season-to-date average uses a cumulative mean up to (not including) this game
  - opponent pace / defensive rating come from the opponent's PRIOR games only
  - teammate-availability (who sat) is inferred from box scores in availability.py,
    which is itself strictly to-date

New signal vs the 1-season model:
  * teammate availability — n_rotation_out, top2_teammate_out
  * opponent pace + defensive rating from team logs (not a player-log approximation)
  * usage_roll5 + usage_trend (rising/falling), not just scoring averages
"""
from __future__ import annotations
import glob
import os
import numpy as np
import pandas as pd

from availability import build_availability

FEATURES = [
    "pts_roll5", "pts_roll10",
    "min_roll5", "min_roll10",
    "ppm_roll5",
    "pts_season_avg",
    "usage_roll5", "usage_trend",     # usage level + direction
    "opp_def_rating", "opp_pace",     # from team logs, to-date
    "n_rotation_out", "top2_teammate_out",   # availability proxy (the new signal)
    "home", "days_rest", "b2b",
]
TARGET = "pts"
TEAM_LOGS_GLOB = "data/raw/team_logs/season=*/team_logs_*.parquet"


# ── small helpers ────────────────────────────────────────────────────────────────
def _parse_home(matchup) -> int:
    return 1 if "vs." in str(matchup) else 0


def _shift_roll(series: pd.Series, window: int) -> pd.Series:
    return series.shift(1).rolling(window, min_periods=1).mean()


# ── opponent pace / defensive rating from TEAM logs ──────────────────────────────
def build_team_defense(team_logs: pd.DataFrame) -> pd.DataFrame:
    """Per (team_id, game_id): that team's to-date pace + defensive rating.

    DefRtg = 100 * opponent_points / possessions; pace = possessions per game.
    Both are rolling means over the team's PRIOR games (shift(1)) — leakage-safe.
    Merge onto a player row by the OPPONENT team id to get opp_def_rating/opp_pace.
    """
    df = team_logs.copy()
    df.columns = [c.lower() for c in df.columns]
    df["game_id"] = df["game_id"].astype(str).str.zfill(10)
    df["game_date"] = pd.to_datetime(df["game_date"])
    for c in ("fga", "fta", "tov", "oreb", "pts"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["poss"] = (df["fga"] + 0.44 * df["fta"] - df["oreb"] + df["tov"]).clip(lower=1)

    # pair the two teams in each game to get opponent points
    g = df[["game_id", "team_id", "pts", "poss"]]
    paired = g.merge(
        g.rename(columns={"team_id": "opp_id", "pts": "opp_pts", "poss": "opp_poss"}),
        on="game_id",
    )
    paired = paired[paired["team_id"] != paired["opp_id"]]
    paired = paired.merge(df[["team_id", "game_id", "game_date", "season"]],
                          on=["team_id", "game_id"])
    paired = paired.sort_values(["team_id", "game_date"])

    paired["def_rating_game"] = 100.0 * paired["opp_pts"] / paired["poss"]
    paired["opp_def_rating"] = (
        paired.groupby("team_id")["def_rating_game"]
        .transform(lambda s: s.shift(1).expanding().mean())
    )
    paired["opp_pace"] = (
        paired.groupby("team_id")["poss"]
        .transform(lambda s: s.shift(1).expanding().mean())
    )
    return paired[["team_id", "game_id", "opp_def_rating", "opp_pace"]].drop_duplicates(
        subset=["team_id", "game_id"]
    )


def _load_team_logs() -> pd.DataFrame | None:
    files = glob.glob(TEAM_LOGS_GLOB)
    if not files:
        return None
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


# ── player rolling / usage / rest ────────────────────────────────────────────────
def _add_player_rolling(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["player_id", "game_date"]).copy()

    def roll(col, w):
        return df.groupby("player_id")[col].transform(lambda s: _shift_roll(s, w))

    df["pts_roll5"] = roll("pts", 5)
    df["pts_roll10"] = roll("pts", 10)
    df["min_roll5"] = roll("min_dec", 5)
    df["min_roll10"] = roll("min_dec", 10)

    df["ppm_raw"] = df["pts"] / df["min_dec"].replace(0, np.nan)
    df["ppm_roll5"] = df.groupby("player_id")["ppm_raw"].transform(lambda s: _shift_roll(s, 5))
    df.drop(columns=["ppm_raw"], inplace=True)

    # usage proxy = FGA + 0.44*FTA + TOV ; level + direction
    df["usage_raw"] = df["fga"] + 0.44 * df["fta"] + df["tov"]
    df["usage_roll5"] = df.groupby("player_id")["usage_raw"].transform(lambda s: _shift_roll(s, 5))
    usage_roll10 = df.groupby("player_id")["usage_raw"].transform(lambda s: _shift_roll(s, 10))
    df["usage_trend"] = df["usage_roll5"] - usage_roll10     # >0 rising, <0 falling
    df.drop(columns=["usage_raw"], inplace=True)

    df["pts_season_avg"] = df.groupby(["player_id", "season"])["pts"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    return df


def _add_rest(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["player_id", "game_date"]).copy()
    df["game_date_dt"] = pd.to_datetime(df["game_date"])
    df["days_rest"] = (
        df.groupby("player_id")["game_date_dt"].transform(lambda s: s.diff().dt.days)
        .fillna(7).clip(upper=10)
    )
    df["b2b"] = (df["days_rest"] == 1).astype(int)
    df.drop(columns=["game_date_dt"], inplace=True)
    return df


# ── main entry point ───────────────────────────────────────────────────────────
def build_features(df: pd.DataFrame, team_logs: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build the full feature matrix from raw player gamelogs (5-season)."""
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    # NBA game ids are 10-char strings ("0022000001"); CSV may parse them as int
    # and drop leading zeros. Normalize so merges against parquet line up.
    df["game_id"] = df["game_id"].astype(str).str.zfill(10)
    if "min_dec" not in df.columns and "min" in df.columns:
        df["min_dec"] = pd.to_numeric(df["min"], errors="coerce").fillna(0.0)
    for c in ("fga", "fta", "tov", "oreb", "pts"):
        if c not in df.columns:
            df[c] = 0.0

    df["home"] = df["matchup"].apply(_parse_home)
    df = _add_rest(df)
    df = _add_player_rolling(df)

    # teammate availability (the new signal)
    avail = build_availability(df)
    df = df.merge(avail, on=["player_id", "game_id"], how="left")

    # opponent team id per player-game
    game_teams = df[["game_id", "team_id"]].drop_duplicates()
    game_opp = game_teams.merge(
        game_teams.rename(columns={"team_id": "opp_team_id"}), on="game_id"
    )
    game_opp = game_opp[game_opp["team_id"] != game_opp["opp_team_id"]]
    df = df.merge(game_opp, on=["game_id", "team_id"], how="left")

    # opponent pace + defensive rating from team logs
    if team_logs is None:
        team_logs = _load_team_logs()
    if team_logs is not None:
        team_def = build_team_defense(team_logs)
        df = df.merge(
            team_def.rename(columns={"team_id": "opp_team_id"}),
            on=["game_id", "opp_team_id"], how="left",
        )
    else:
        df["opp_def_rating"] = np.nan
        df["opp_pace"] = np.nan

    # fill cold-start NaNs with per-player-season means, then 0
    for col in FEATURES:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = df[col].fillna(
            df.groupby(["player_id", "season"])[col].transform("mean")
        )
    df[FEATURES] = df[FEATURES].fillna(0.0)
    df[TARGET] = df[TARGET].fillna(0.0)
    df = df.dropna(subset=[TARGET])

    meta = ["player_id", "player_name", "game_id", "game_date", "season",
            "team_id", "opp_team_id", "matchup"]
    keep = [c for c in meta if c in df.columns] + FEATURES + [TARGET]
    return df[keep].reset_index(drop=True)
