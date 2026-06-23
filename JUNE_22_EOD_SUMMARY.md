# June 22 — End of Day Summary (Ship Log)
_Repo: nba-winprob · branch: main · last commit: 8e0af33 "chore: fix Streamlit deprecation warnings"_
_Remote: https://github.com/tirthsuba15/nba-winprob (all work pushed)_

## ▶ START HERE  (the one next action)
Run the overnight play-by-play pull — it's the one thing blocking 5-season
win-probability training. Everything else is cached and ready.
First command: `nohup python src/data/fetch_pbp.py > pbp.out 2>&1 &` then `tail -f pbp.out`
Done when: `data/.fetch_log/pbp_progress.jsonl` has ~6,400 "done" lines (currently 4),
then `python src/data/catalog.py --summary` shows pbp ≈ 3M rows.
Note: ~3–4 hrs, resumable (re-run to continue); must run on the laptop — stats.nba.com blocks datacenter IPs.

## Next steps (ordered)
1. [ ] Run full PBP fetch — `src/data/fetch_pbp.py` — done when progress log ≈ 6,400 done
2. [ ] Export 5-season win-prob CSVs — `python src/data/export_winprob.py` — done when data/games.csv has ~6,400 games
3. [ ] Retrain win-prob on 5 seasons — `python src/model.py --data data --test-seasons 2024-25` — done when outputs/summary.json shows 5 seasons; update README results table
4. [ ] (Optional) Upgrade injury feed — balldontlie paid tier OR MCP — `src/player_points/news.py` — done when `python src/player_points/news.py` returns injuries (currently 401)
5. [ ] (Optional) Re-run win-prob curve + reliability on real 5-season data — `python src/plot_game.py` — done when outputs/*.png regenerate

## Current state
- ✅ Working & verified:
  - Win-prob model on 2 seasons (2023-24+2024-25): calibrated XGBoost, logistic baseline beats it (honest). outputs/ committed.
  - Streamlit dashboard (`app.py`): Win Probability + Player Props tabs, HTTP 200, no deprecation warnings, team names readable.
  - Player-points model (`src/player_points/`): 5-season, distribution (mean + 80% interval @ alphas 0.085/0.915, coverage 0.801, width 14.46), MAE 4.55 vs naive 4.75.
  - Over/under: isotonic-calibrated P(over) persisted in bundle + applied in serving (predict.py + app.py). Selective coverage top-5% = 0.7596 (plateau) / 0.764 (after coverage retune). Overall 0.576 < 0.63 tripwire. Brier 0.240 / log-loss 0.673 beat naive.
  - 5-season data layer: games/player_logs/team_logs/rosters as parquet + DuckDB catalog (cached, rate-limited, resumable). bulk pulls DONE.
  - news.py live injury feed scaffolded (balldontlie key in gitignored .env; 401 graceful fallback + MCP seam).
  - HANDOFF.md written with plateau result + feature importance + next-lift path.
- 🚧 In progress / not done:
  - Full PBP fetch — only 4 sample games pulled. Win-prob still on the old 2-season data.
- 🔴 Known issues:
  - balldontlie `player_injuries` returns 401 (free-tier key); live injuries inert until paid tier or MCP.
  - Player-points "Projected points" is the mean estimate, not strictly the median (no q50 model).

## Open questions / decisions pending
- Win-prob: train on all 5 seasons after PBP pull, or keep 2-season? (5 is the honest upgrade.)
- Injury feed: pay for balldontlie GOAT tier, wire the MCP, or switch to an odds API? Needed for any real player-points lift.

## Blocked — do when online / on laptop
- Overnight PBP pull (network-heavy, laptop only).
- balldontlie paid tier / odds API (account + payment).

## Reload context (for a cold start)
- Key files: `CLAUDE.md` (hard rules: no leakage, calibration-first, no fabricated numbers), `HANDOFF.md` (player-points plateau detail), `src/player_points/model.py` (bundle: minutes sub-model + mean/q10/q90 + isotonic po_cal; Q_LO/Q_HI=0.085/0.915), `src/data/` (data layer), `app.py` (dashboard).
- Mental model: two SEPARATE models. (1) Win probability — XGBoost on live game state, calibrated. (2) Player points — separate package, projects a points DISTRIBUTION; over/under "line" = player's own season average (NOT Vegas), so the edge is purely the deviation signal and ~0.76 top-5% is the honest ceiling.
- Run it: `streamlit run app.py` (dashboard) · `python src/player_points/backtest_overunder.py` (coverage curve + leakage tripwire).

---
## History
### June 22 — built player-points model, 5-season data layer, calibrated over/under; win-prob calibration
- Changed: 14 commits 8ec2164→8e0af33. New: src/player_points/ (features, model, predict, odds, news, availability, backtest_overunder), src/data/ (fetch_bulk, fetch_pbp, fetch_rosters, catalog, export_*), app.py Player Props tab, HANDOFF.md. Win-prob: isotonic calibration added.
- Outcome: calibrated win-prob; full player-points pipeline with distribution projections + selective-coverage over/under (top-5% ~0.76, plateau confirmed, leakage-clean); 5-season parquet/DuckDB data layer (bulk done, PBP pending); dashboard live with both tabs.
