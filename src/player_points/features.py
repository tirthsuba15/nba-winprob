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

try:
    from .availability import build_availability
except ImportError:
    from availability import build_availability

# Points-model features. Minutes are NOT here as a raw average — the minutes
# sub-model's `pred_minutes` (appended in model.points_matrix) carries that
# signal instead. `min_season_avg` is computed and carried as metadata so the
# minutes sub-model can consume it, but the points model never sees it raw.
FEATURES = [
    "pts_roll5", "pts_roll10",
    "ppm_roll5",
    "pts_season_avg",
    "usage_roll5", "usage_trend",     # usage level + direction
    "opp_def_rating", "opp_pace",     # from team logs, to-date
    "opp_def_vs_pos",                 # opp pts allowed to this player's position, to-date
    "expected_mismatch",              # blowout proxy: |own net − opp net| (pregame)
    "n_rotation_out", "top2_teammate_out",   # availability proxy (the new signal)
    "home", "days_rest", "b2b",
]
TARGET = "pts"
TEAM_LOGS_GLOB = "data/raw/team_logs/season=*/team_logs_*.parquet"
ROSTER_GLOB = "data/raw/rosters/season=*.parquet"


# ── small helpers ────────────────────────────────────────────────────────────────
def _parse_home(matchup) -> int:
    return 1 if "vs." in str(matchup) else 0


def _shift_roll(series: pd.Series, window: int) -> pd.Series:
    return series.shift(1).rolling(window, min_periods=1).mean()


# ── opponent pace / defensive rating from TEAM logs ──────────────────────────────
def build_team_metrics(team_logs: pd.DataFrame) -> pd.DataFrame:
    """Per (team_id, game_id): that team's to-date off/def rating, net, pace.

    OffRtg = 100*own_pts/poss; DefRtg = 100*opp_pts/poss; Net = Off - Def;
    pace = possessions/game. All are rolling means over the team's PRIOR games
    (shift(1)) — leakage-safe. Used two ways in build_features:
      * merged by OPPONENT id  -> opp_def_rating, opp_pace
      * own & opp net rating   -> expected_mismatch (blowout proxy)
    """
    df = team_logs.copy()
    df.columns = [c.lower() for c in df.columns]
    df["game_id"] = df["game_id"].astype(str).str.zfill(10)
    df["game_date"] = pd.to_datetime(df["game_date"])
    for c in ("fga", "fta", "tov", "oreb", "pts"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["poss"] = (df["fga"] + 0.44 * df["fta"] - df["oreb"] + df["tov"]).clip(lower=1)

    g = df[["game_id", "team_id", "pts", "poss"]]
    paired = g.merge(
        g.rename(columns={"team_id": "opp_id", "pts": "opp_pts", "poss": "opp_poss"}),
        on="game_id",
    )
    paired = paired[paired["team_id"] != paired["opp_id"]]
    paired = paired.merge(df[["team_id", "game_id", "game_date", "season"]],
                          on=["team_id", "game_id"])
    paired = paired.sort_values(["team_id", "game_date"])

    paired["off_game"] = 100.0 * paired["pts"] / paired["poss"]
    paired["def_game"] = 100.0 * paired["opp_pts"] / paired["poss"]

    def td(col):
        return paired.groupby("team_id")[col].transform(
            lambda s: s.shift(1).expanding().mean())

    paired["off_rtg_td"] = td("off_game")
    paired["def_rtg_td"] = td("def_game")
    paired["pace_td"] = td("poss")
    paired["net_rtg_td"] = paired["off_rtg_td"] - paired["def_rtg_td"]
    return paired[["team_id", "game_id", "off_rtg_td", "def_rtg_td",
                   "net_rtg_td", "pace_td"]].drop_duplicates(subset=["team_id", "game_id"])


def _load_team_logs() -> pd.DataFrame | None:
    files = glob.glob(TEAM_LOGS_GLOB)
    if not files:
        return None
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def _pos_group(position) -> str | float:
    """'G-F'->'G', 'F-C'->'F', 'C'->'C'. First token wins."""
    if position is None or (isinstance(position, float) and np.isnan(position)):
        return np.nan
    return str(position).split("-")[0].strip().upper() or np.nan


def _load_positions() -> pd.DataFrame | None:
    """player_id, season -> pos_group from rosters (one row per player-season)."""
    files = glob.glob(ROSTER_GLOB)
    if not files:
        return None
    r = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    r["pos_group"] = r["position"].apply(_pos_group)
    r = r.dropna(subset=["pos_group"])
    # a player may sit on >1 roster in a season; keep their most common position
    r = (r.groupby(["player_id", "season"])["pos_group"]
         .agg(lambda s: s.value_counts().index[0]).reset_index())
    return r


def build_opp_def_vs_pos(df: pd.DataFrame, game_opp: pd.DataFrame) -> pd.DataFrame:
    """Per (team_id=defender, game_id, pos_group): to-date pts allowed to that
    position. Leakage-safe (shift+expanding over the defender's prior games).
    `df` must already carry pos_group and game_date.
    """
    d = df.dropna(subset=["pos_group"]).copy()
    # points SCORED by each team's players of each position, per game
    scored = (d.groupby(["game_id", "team_id", "pos_group"], as_index=False)["pts"].sum())
    # points ALLOWED by defender D = scored by D's opponent of that position
    allowed = game_opp.merge(
        scored.rename(columns={"team_id": "opp_team_id", "pts": "allowed_pts"}),
        on=["game_id", "opp_team_id"], how="inner",
    )  # columns: game_id, team_id(=defender D), opp_team_id, pos_group, allowed_pts
    dates = d[["game_id", "team_id", "game_date"]].drop_duplicates()
    allowed = allowed.merge(dates, on=["game_id", "team_id"], how="left")
    allowed = allowed.sort_values(["team_id", "pos_group", "game_date"])
    allowed["opp_def_vs_pos"] = (
        allowed.groupby(["team_id", "pos_group"])["allowed_pts"]
        .transform(lambda s: s.shift(1).expanding().mean())
    )
    return allowed[["team_id", "game_id", "pos_group", "opp_def_vs_pos"]]


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
    # season-to-date average minutes (leakage-safe) — input to the minutes sub-model
    df["min_season_avg"] = df.groupby(["player_id", "season"])["min_dec"].transform(
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

    # opponent pace + defensive rating from team logs, and a blowout proxy
    # (expected_mismatch = |own net rating − opp net rating|, to-date)
    if team_logs is None:
        team_logs = _load_team_logs()
    if team_logs is not None:
        tm = build_team_metrics(team_logs)
        # opponent's to-date defense + pace (player faces the opponent)
        df = df.merge(
            tm[["team_id", "game_id", "def_rtg_td", "pace_td", "net_rtg_td"]]
            .rename(columns={"team_id": "opp_team_id", "def_rtg_td": "opp_def_rating",
                             "pace_td": "opp_pace", "net_rtg_td": "opp_net"}),
            on=["game_id", "opp_team_id"], how="left",
        )
        # own team's to-date net rating
        df = df.merge(
            tm[["team_id", "game_id", "net_rtg_td"]].rename(columns={"net_rtg_td": "own_net"}),
            on=["game_id", "team_id"], how="left",
        )
        df["expected_mismatch"] = (df["own_net"] - df["opp_net"]).abs()
    else:
        df["opp_def_rating"] = np.nan
        df["opp_pace"] = np.nan
        df["expected_mismatch"] = np.nan

    # opponent defense vs the player's position (needs rosters for position)
    positions = _load_positions()
    if positions is not None:
        df = df.merge(positions, on=["player_id", "season"], how="left")
        df["game_date"] = pd.to_datetime(df["game_date"])
        dvp = build_opp_def_vs_pos(df, game_opp)
        # player faces opponent: match defender = opp_team_id, position = player's
        df = df.merge(
            dvp.rename(columns={"team_id": "opp_team_id"}),
            on=["game_id", "opp_team_id", "pos_group"], how="left",
        )
    else:
        df["opp_def_vs_pos"] = np.nan

    # LEAKAGE GUARD: never fill NaNs with a forward-looking statistic (e.g. a
    # season-wide mean peeks at games after this row). XGBoost handles NaN
    # natively (it learns a default split direction), so we leave cold-start
    # gaps as NaN rather than imputing from the future. Rest/home/b2b are always
    # defined; rolling/availability/opponent features may be NaN early-season.
    for col in FEATURES:
        if col not in df.columns:
            df[col] = np.nan
    df[TARGET] = df[TARGET].fillna(0.0)
    df = df.dropna(subset=[TARGET])

    meta = ["player_id", "player_name", "game_id", "game_date", "season",
            "team_id", "opp_team_id", "matchup", "min_dec", "min_season_avg"]
    keep = [c for c in meta if c in df.columns] + FEATURES + [TARGET]
    return df[keep].reset_index(drop=True)
