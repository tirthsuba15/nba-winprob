# Operating contract for this repo

Read this before changing anything. These rules are what make the project
credible to anyone who looks closely (recruiters, ML engineers).

## What this is
A calibrated NBA win-probability model. Given the state of a game (score, time
left, team strength) it estimates P(home team wins). The achievement is rigor,
not beating Vegas.

## Stack (hard constraint: laptop-only, no GPU needed)
NBA win probability is **tabular** data. The right tool is gradient-boosted
trees (XGBoost), which train in seconds on a CPU. This is the industry-standard
approach (nflfastR's NFL win-prob model is XGBoost). A GPU buys ~nothing here.
Allowed deps: nba_api, pandas, numpy, scikit-learn, xgboost, matplotlib.

## Non-negotiable rules
1. **No data leakage.** A feature may only use information available at that
   moment. Pregame strength features (Elo) use only prior games. Never let the
   final score leak into a feature.
2. **Split by game AND season.** Hold out whole seasons for testing. No game
   appears in both train and test.
3. **Calibration is first-class.** Always report log-loss + Brier + a
   reliability diagram. A 70% prediction must win ~70% of the time. Accuracy
   alone is not enough.
4. **No fabricated numbers.** Every number in the README must come from a real
   run (`outputs/summary.json`). Synthetic-data figures must be labelled as
   illustrative.
5. **Honest baselines.** Always compare against naive (base rate), Elo-only,
   and a simple logistic model so the model's value-add is explicit.

## Layout
- `src/synth.py`     synthetic data (offline dev/testing) — same schema as real
- `src/fetch_data.py`real data via nba_api (run on laptop; stats.nba.com blocks CI)
- `src/elo.py`       leakage-safe chronological Elo ratings
- `src/features.py`  feature engineering + season-based split
- `src/model.py`     train/eval XGBoost + baselines + calibration
- `src/plot_game.py` the live win-probability curve for one game
- `src/wpa.py`       Win Probability Added — clutch plays / player leaderboard

## Data schema (synth and real must match)
- `data/games.csv`:  game_id, season, game_date, home_team_id, away_team_id,
  home_score, away_score, home_win
- `data/moments.csv`: game_id, season, period, secs_left, score_margin,
  label_home_win [, description, player]

## Roadmap (don't claim these are done until they are)
- Player/lineup features: on-court lineup strength from substitution parsing.
- Deep-learning bake-off: a sequence model (LSTM/Transformer) over play-by-play,
  compared head-to-head with XGBoost on the SAME calibrated backtest. This is
  the honest place a GPU helps — and the comparison is itself the story.
- Streamlit demo: pick a game, watch the curve.
