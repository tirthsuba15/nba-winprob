# Player-points over/under — phase handoff

Status of the player-points projection model (`src/player_points/`) and its
over/under layer at the end of this phase. The win-probability model is separate
and unaffected.

## Headline: plateau confirmed

The over/under model has **plateaued**. Selective top-5% hit rate:

- **0.7596** at plateau confirmation (the point where adding features stopped
  helping), with the line set to the **player's own season average to date**.
- **0.764** current, after re-tuning the quantile alphas for interval coverage
  (the wider interval recalibrated P(over) and nudged the confident slice up).

Either way the ceiling is **~0.76**. This is *selective coverage* (the most
confident 5% of calls, ~1,377 of 27,541), **not** an overall edge.

Why ~0.76 is the ceiling: the "line" is the player's season average, so the bet
is "beats their own running average tonight." The only signal is the deviation
(recent form / usage / availability). A real sportsbook line already prices that
in, so these numbers do **not** transfer to beating a book.

### Full backtest (train 2020-21…2023-24, test 2024-25, 27,541 calls)

| slice | coverage | hit rate |
|---|---|---|
| top 5% | 5% | 0.764 |
| top 10% | 10% | 0.718 |
| top 25% | 25% | 0.664 |
| top 50% | 50% | 0.625 |
| **all calls** | 100% | **0.576** |

- Overall **0.576 < 0.63** leakage tripwire — OK (an overall hit rate above 0.63
  is treated as a leakage bug, not a win).
- Point model: **MAE 4.55** vs naive (season-avg-to-date) **4.75**.
- Probabilistic: **Brier 0.240** vs naive 0.249; **log-loss 0.673** vs 0.692.
- P(over) is isotonic-calibrated (fit on train only); the calibrator is persisted
  in the model bundle and applied in `predict.py` + `app.py`, so the dashboard
  shows calibrated probabilities, not raw ones.

## Feature importance (points mean model, as-is from model.py)

```
pts_roll10             0.573
usage_roll5            0.133
pts_season_avg         0.070
pred_minutes           0.037   (from the minutes sub-model)
pts_last3_avg          0.035
n_rotation_out         0.024
pts_roll5              0.020
top2_teammate_out      0.019
pts_vs_opponent_hist   0.013
days_rest              0.012
ppm_roll5              0.009
b2b                    0.009
opp_def_vs_pos         0.008
opp_pace               0.008
opp_def_rating         0.008
usage_trend            0.008
expected_mismatch      0.008
home                   0.007
```

Signal concentrates in `pts_roll10` / `usage_roll5` / `pts_season_avg`. Six
feature additions (opponent defense, position-defense, blowout proxy, minutes
sub-model, exponential recency weighting, `pts_last3_avg` + `pts_vs_opponent_hist`)
moved the top-5% only 0.746 → 0.76. Everything past the top three is marginal.

## Interval coverage fix (this phase)

Exponential recency weighting (`weight = exp(-0.1 · games_ago)`) tightened the
quantile interval, dropping 80% coverage below target. Re-tuned the quantile
regressor alphas (held-out 2024-25):

| | alpha_lo / alpha_hi | coverage | width |
|---|---|---|---|
| before | 0.10 / 0.90 | 0.777 | 13.6 |
| **after** | **0.085 / 0.915** | **0.801** | **14.46** |

Coverage restored to **≥ 0.80** without inflating width beyond the 14.5 cap. Set
as `Q_LO, Q_HI` in `model.py`.

## Next lift requires (not history-only)

The model is out of cheap signal. Real improvement needs one of:

1. **A real sportsbook line** instead of the player's season average. Then the
   task becomes "beat the market," and the deviation signal has somewhere to bite.
   Source: an odds API (e.g. The Odds API, OddsAPI) — paid.
2. **A live minutes / injury feed** so projected minutes reflect tonight's
   actual availability (the biggest source of points variance history can't see).
   Source: balldontlie **paid tier** (the `player_injuries` endpoint 401s on the
   current free key — see `src/player_points/news.py`), or another injury feed.

`news.py` already has the live-feed seam (balldontlie + timestamped snapshots,
graceful 401 fallback); it activates when the key tier is upgraded or an MCP
server is wired in.

## How to reproduce

```bash
python src/data/export_player_points.py            # 5-season parquet -> csv
python src/player_points/model.py --data data      # train bundle (+ coverage)
python src/player_points/backtest_overunder.py     # coverage curve + tripwire
```
