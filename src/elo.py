"""
Leakage-safe Elo ratings for NBA teams.

Elo is a simple, strong, interpretable team-strength signal (the same family of
ratings FiveThirtyEight used for NBA). Crucially we compute it CHRONOLOGICALLY:
each game's *pregame* rating uses only games that finished before it. That makes
the resulting `elo_diff` a legitimate feature -- it never peeks at the outcome
it is being used to predict.

Public API:
    add_pregame_elo(games_df) -> games_df with pregame_elo_home/away + elo_prob_home
"""
from __future__ import annotations
import pandas as pd

K = 20.0            # update speed
HOME_ADV = 100.0    # Elo points of home-court advantage
MEAN_REVERT = 0.75  # carry 75% of last season's rating into the next
BASE = 1500.0


def _expected(elo_home, elo_away):
    return 1.0 / (1.0 + 10 ** (-((elo_home + HOME_ADV) - elo_away) / 400.0))


def add_pregame_elo(games: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `games` with pregame Elo columns added.

    Required columns: game_id, season, game_date, home_team_id, away_team_id,
    home_win. Rows are processed in (season, date) order.
    """
    g = games.copy()
    g["game_date"] = pd.to_datetime(g["game_date"])
    g = g.sort_values(["game_date", "game_id"]).reset_index(drop=True)

    ratings: dict[int, float] = {}
    last_season: dict[int, str] = {}
    home_elo, away_elo, elo_prob = [], [], []

    for _, row in g.iterrows():
        h, a, season = row["home_team_id"], row["away_team_id"], row["season"]
        for tid in (h, a):
            if tid not in ratings:
                ratings[tid] = BASE
                last_season[tid] = season
            elif last_season[tid] != season:
                # New season: regress toward the mean (roster turnover).
                ratings[tid] = MEAN_REVERT * ratings[tid] + (1 - MEAN_REVERT) * BASE
                last_season[tid] = season

        eh, ea = ratings[h], ratings[a]
        p = _expected(eh, ea)
        home_elo.append(eh)
        away_elo.append(ea)
        elo_prob.append(p)

        # Post-game update (only AFTER recording the pregame values).
        outcome = float(row["home_win"])
        ratings[h] = eh + K * (outcome - p)
        ratings[a] = ea + K * ((1 - outcome) - (1 - p))

    g["pregame_elo_home"] = home_elo
    g["pregame_elo_away"] = away_elo
    g["elo_diff"] = g["pregame_elo_home"] - g["pregame_elo_away"]
    g["elo_prob_home"] = elo_prob
    return g
