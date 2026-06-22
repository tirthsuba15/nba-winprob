"""
Feature engineering + leakage-safe train/test splitting.

Every feature here is computable AT THE MOMENT it describes -- score margin,
seconds left, period, and a *pregame* Elo edge. Nothing uses the final score.
The split is by GAME and by SEASON so no game appears in both train and test,
and we test on the most recent season(s) only -- the honest way to estimate how
the model would have performed on games it never saw.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from elo import add_pregame_elo

FEATURES = [
    "score_margin",     # home points - away points, right now
    "secs_left",        # seconds remaining in regulation
    "frac_left",        # secs_left / 2880
    "period",
    "elo_diff",         # pregame home Elo - away Elo (leakage-safe)
    "margin_x_fracleft" # interaction: a 5-pt lead late >> a 5-pt lead early
]
TARGET = "label_home_win"
REG_SECONDS = 48 * 60


def build_dataset(moments: pd.DataFrame, games: pd.DataFrame):
    """Join pregame Elo onto moments and engineer model features."""
    games_elo = add_pregame_elo(games)
    df = moments.merge(
        games_elo[["game_id", "season", "elo_diff", "elo_prob_home"]],
        on=["game_id", "season"], how="left",
    )
    df["frac_left"] = df["secs_left"].clip(lower=0) / REG_SECONDS
    df["margin_x_fracleft"] = df["score_margin"] * df["frac_left"]
    df["elo_diff"] = df["elo_diff"].fillna(0.0)
    return df


def split_by_season(df: pd.DataFrame, test_seasons):
    """Hold out entire seasons for testing -- no game crosses the boundary."""
    test_mask = df["season"].isin(test_seasons)
    train, test = df[~test_mask].copy(), df[test_mask].copy()
    return train, test


def Xy(df: pd.DataFrame):
    return df[FEATURES].to_numpy(dtype=float), df[TARGET].to_numpy(dtype=int)
