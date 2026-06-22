"""
Synthetic NBA data generator.

Produces the SAME schema as fetch_data.py so the whole pipeline can be
developed and tested offline (no network, no nba_api). The synthetic world is
deliberately "learnable": final outcome depends on pregame team strength (Elo)
plus the in-game score margin and time remaining -- exactly the structure a
real win-probability model should pick up. This lets us verify the model
actually beats the baselines and is well calibrated before touching real data.

Outputs:
  data/games.csv    one row per game (final result + teams + date)
  data/moments.csv  many rows per game (the game state at sampled timestamps)
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd

REG_SECONDS = 48 * 60  # 2880 seconds in regulation


def _simulate_game(rng, game_id, season, date, home_id, away_id,
                   home_elo, away_elo, n_samples=40):
    # Pregame win prob from Elo (+ home-court bump ~ 100 Elo points).
    elo_diff = (home_elo + 100) - away_elo
    p_home = 1.0 / (1.0 + 10 ** (-elo_diff / 400.0))

    # Expected final margin scales with elo edge; add noise.
    exp_margin = (elo_diff / 25.0)
    final_margin = rng.normal(exp_margin, 12.0)
    # Nudge so the sign roughly matches a Bernoulli(p_home) outcome.
    if rng.random() < p_home and final_margin < 0:
        final_margin = abs(final_margin)
    elif rng.random() > p_home and final_margin > 0:
        final_margin = -abs(final_margin)
    final_margin = int(round(final_margin)) or (1 if rng.random() < p_home else -1)
    home_win = int(final_margin > 0)

    home_score = 110 + max(final_margin, 0) + rng.integers(-6, 7)
    away_score = home_score - final_margin

    # Build a margin path as a Brownian bridge from 0 -> final_margin.
    # Scale the wobble so mid-game lead swings have ~9-point std (realistic).
    t = np.linspace(0, 1, REG_SECONDS)
    walk = rng.normal(0, 1, REG_SECONDS).cumsum()
    bridge = walk - t * walk[-1]                  # pin both ends to 0
    wobble_mid_std = 9.0
    bridge *= wobble_mid_std / (np.sqrt(REG_SECONDS) / 2.0)
    margin_path = final_margin * t + bridge       # drift to final + wobble
    margin_path = np.round(margin_path).astype(int)

    # Sample moments across the game.
    idx = np.sort(rng.choice(REG_SECONDS, size=min(n_samples, REG_SECONDS),
                             replace=False))
    rows = []
    for i in idx:
        secs_elapsed = int(i)
        secs_left = REG_SECONDS - secs_elapsed
        period = min(4, secs_elapsed // (12 * 60) + 1)
        rows.append({
            "game_id": game_id,
            "season": season,
            "game_date": date,
            "home_team_id": home_id,
            "away_team_id": away_id,
            "period": period,
            "secs_left": secs_left,
            "score_margin": int(margin_path[i]),
            "label_home_win": home_win,
        })

    game_row = {
        "game_id": game_id,
        "season": season,
        "game_date": date,
        "home_team_id": home_id,
        "away_team_id": away_id,
        "home_score": int(home_score),
        "away_score": int(away_score),
        "home_win": home_win,
    }
    return rows, game_row


def generate(seasons, games_per_season, n_teams=30, seed=7):
    rng = np.random.default_rng(seed)
    # Persistent "true" team strengths; drift a little each season.
    true_elo = {tid: rng.normal(1500, 110) for tid in range(1, n_teams + 1)}

    all_moments, all_games = [], []
    gid = 0
    for season in seasons:
        for tid in true_elo:
            true_elo[tid] += rng.normal(0, 25)  # offseason roster churn
        base_date = pd.Timestamp(f"{season.split('-')[0]}-10-20")
        for g in range(games_per_season):
            gid += 1
            home, away = rng.choice(range(1, n_teams + 1), size=2, replace=False)
            date = (base_date + pd.Timedelta(days=int(g / 5))).strftime("%Y-%m-%d")
            moments, game = _simulate_game(
                rng, f"SYN{gid:07d}", season, date, int(home), int(away),
                true_elo[int(home)], true_elo[int(away)],
            )
            all_moments.extend(moments)
            all_games.append(game)
    return pd.DataFrame(all_moments), pd.DataFrame(all_games)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+",
                    default=["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"])
    ap.add_argument("--games", type=int, default=300,
                    help="games per season")
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    moments, games = generate(args.seasons, args.games, seed=args.seed)
    moments.to_csv(os.path.join(args.out, "moments.csv"), index=False)
    games.to_csv(os.path.join(args.out, "games.csv"), index=False)
    print(f"Wrote {len(games):,} games and {len(moments):,} moments "
          f"across {len(args.seasons)} seasons to {args.out}/")
    print("NOTE: synthetic data -- for pipeline testing only, not real results.")


if __name__ == "__main__":
    main()
