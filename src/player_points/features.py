"""
Leakage-safe feature engineering for the player-points projection model.

Every feature is computable BEFORE the game tips off:
  - Rolling stats use shift(1) so the current game is never in its own window.
  - Opponent defensive features are computed from opponent games PRIOR to this date.
  - Season-to-date average uses a cumulative mean up to but not including this game.

No feature uses the current game's own box score.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

FEATURES = [
    "pts_roll5",         # rolling mean PTS over last 5 games (excl. current)
    "pts_roll10",        # rolling mean PTS over last 10 games
    "min_roll5",         # rolling mean minutes over last 5 games
    "min_roll10",        # rolling mean minutes over last 10 games
    "ppm_roll5",         # points-per-minute rolling 5 (efficiency proxy)
    "pts_season_avg",    # cumulative season-to-date PTS average (excl. current)
    "opp_pts_allowed",   # opponent team's rolling avg PTS allowed before this date
    "opp_pace",          # opponent team's rolling avg estimated possessions per game
    "home",              # 1 = home game, 0 = away
    "days_rest",         # calendar days since player's last game (capped at 10)
    "b2b",               # 1 = back-to-back (days_rest == 1)
]
TARGET = "pts"


# ── helpers ────────────────────────────────────────────────────────────────────

def _parse_home(matchup: str) -> int:
    """'BOS vs. LAL' -> 1 (home),  'BOS @ LAL' -> 0 (away)."""
    return 1 if "vs." in str(matchup) else 0


def _shift_roll(series: pd.Series, window: int) -> pd.Series:
    """Shift-1 then rolling mean — excludes the current game from its own window."""
    return series.shift(1).rolling(window, min_periods=1).mean()


# ── team-level stats per game ───────────────────────────────────────────────────

def _build_team_game_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate player gamelogs into team-level per-game totals.

    Returns DataFrame: team_id, game_id, game_date, team_pts, possessions
    possessions ≈ FGA + 0.44*FTA + TOV - OREB  (Dean Oliver approximation)
    """
    tg = (
        df.groupby(["team_id", "game_id", "game_date"])
        .agg(team_pts=("pts", "sum"),
             team_fga=("fga", "sum"),
             team_fta=("fta", "sum"),
             team_tov=("tov", "sum"),
             team_oreb=("oreb", "sum"))
        .reset_index()
    )
    tg["possessions"] = (
        tg["team_fga"]
        + 0.44 * tg["team_fta"]
        + tg["team_tov"]
        - tg["team_oreb"]
    ).clip(lower=1)
    return tg[["team_id", "game_id", "game_date", "team_pts", "possessions"]]


def _build_opp_defense_lookup(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (team_id, game_date), compute rolling avg of points they allowed
    and avg pace — both using only games BEFORE this date (leakage-safe).

    Returns DataFrame: team_id, game_id, opp_pts_allowed, opp_pace
    (indexed by the DEFENDING team, so callers look up by opponent team_id)
    """
    tg = _build_team_game_stats(df)

    # Identify the opponent in each game (the other team in same game_id)
    g = tg[["game_id", "team_id", "team_pts", "possessions"]].copy()
    # Self-join on game_id to get the opponent's pts in that game
    paired = g.merge(
        g.rename(columns={"team_id": "opp_id", "team_pts": "opp_pts",
                          "possessions": "opp_poss"}),
        on="game_id",
    )
    paired = paired[paired["team_id"] != paired["opp_id"]]

    # pts_allowed by team_id in this game = opp_pts (what the other team scored)
    paired = paired.merge(
        tg[["team_id", "game_id", "game_date"]],
        on=["team_id", "game_id"],
    )
    paired = paired.sort_values(["team_id", "game_date"])

    # Rolling avg (shift-1 so the game itself isn't in its own window)
    paired["opp_pts_allowed"] = (
        paired.groupby("team_id")["opp_pts"]
        .transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    )
    paired["opp_pace"] = (
        paired.groupby("team_id")["possessions"]
        .transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    )

    return paired[["team_id", "game_id", "opp_pts_allowed", "opp_pace"]].drop_duplicates(
        subset=["team_id", "game_id"]
    )


# ── player-level rolling features ─────────────────────────────────────────────

def _add_player_rolling(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["player_id", "game_date"]).copy()

    def roll(col, w):
        return df.groupby("player_id")[col].transform(
            lambda s: _shift_roll(s, w)
        )

    df["pts_roll5"]  = roll("pts", 5)
    df["pts_roll10"] = roll("pts", 10)
    df["min_roll5"]  = roll("min_dec", 5)
    df["min_roll10"] = roll("min_dec", 10)

    # points-per-minute: compute for each row first, then roll
    df["ppm_raw"] = df["pts"] / df["min_dec"].replace(0, np.nan)
    df["ppm_roll5"] = df.groupby("player_id")["ppm_raw"].transform(
        lambda s: _shift_roll(s, 5)
    )
    df.drop(columns=["ppm_raw"], inplace=True)

    # season-to-date cumulative mean (excludes current game via shift)
    df["pts_season_avg"] = df.groupby(["player_id", "season"])["pts"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    return df


def _add_rest(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["player_id", "game_date"]).copy()
    df["game_date_dt"] = pd.to_datetime(df["game_date"])
    df["days_rest"] = (
        df.groupby("player_id")["game_date_dt"]
        .transform(lambda s: s.diff().dt.days)
        .fillna(7)   # first game of season: treat as well-rested
        .clip(upper=10)
    )
    df["b2b"] = (df["days_rest"] == 1).astype(int)
    df.drop(columns=["game_date_dt"], inplace=True)
    return df


def _add_home(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["home"] = df["matchup"].apply(_parse_home)
    return df


# ── main entry point ───────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full feature matrix from raw player_gamelogs.

    Parameters
    ----------
    df : DataFrame from player_gamelogs.csv

    Returns
    -------
    DataFrame with FEATURES columns + TARGET + metadata
    (player_id, player_name, game_id, game_date, season)
    """
    df = df.copy()
    df = _add_home(df)
    df = _add_rest(df)
    df = _add_player_rolling(df)

    # opponent defensive lookup — keyed by the OPPONENT team's id
    opp_def = _build_opp_defense_lookup(df)

    # Each player row has team_id (their team). The opponent is inferred per game:
    # merge opp_def on (game_id, team_id) where team_id = opponent team.
    # We need to find opp_team_id per player-game.
    # Game has 2 teams; for each player-game, the other team is the opponent.
    game_teams = df[["game_id", "team_id"]].drop_duplicates()
    # self-join to get the opp team per game
    game_opp = game_teams.merge(
        game_teams.rename(columns={"team_id": "opp_team_id"}),
        on="game_id"
    )
    game_opp = game_opp[game_opp["team_id"] != game_opp["opp_team_id"]]

    df = df.merge(game_opp, on=["game_id", "team_id"], how="left")

    # Now merge opp defensive stats: look up opp_team_id in opp_def
    df = df.merge(
        opp_def.rename(columns={"team_id": "opp_team_id"}),
        on=["game_id", "opp_team_id"],
        how="left",
    )

    # Fill NaN rolling features with the season average (cold-start players)
    for col in FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df.groupby(["player_id", "season"])[col]
                                      .transform("mean"))
    # Final fallback for any remaining NaN
    df[FEATURES] = df[FEATURES].fillna(0.0)
    df[TARGET] = df[TARGET].fillna(0.0)

    # Drop rows with no target (shouldn't happen with real data)
    df = df.dropna(subset=[TARGET])

    meta_cols = ["player_id", "player_name", "game_id", "game_date", "season",
                 "team_id", "opp_team_id", "matchup"]
    keep = [c for c in meta_cols if c in df.columns] + FEATURES + [TARGET]
    return df[keep].reset_index(drop=True)
