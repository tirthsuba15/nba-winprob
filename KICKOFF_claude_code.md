# Claude Code kickoff prompts

IMPORTANT — two modes, don't mix them:
- AUTONOMOUS (a `/goal` or Stop hook is active, or you ran with
  `--dangerously-skip-permissions`): the agent must NOT pause to ask you
  questions — a Stop hook will deadlock it in a "goal not met, continuing" loop.
  Use Prompt 1: every decision is pre-baked as a default.
- INTERACTIVE (plain `claude`, no goal hook): the agent CAN pause and ask you.
  Use Prompt 2 for judgment-heavy work like the player-points model.

If you ever get stuck in a stop-hook loop again: the turn eventually frees up —
just type your answer, or clear the goal / disable the Stop hook before rerunning.

================================================================================
PROMPT 1 — AUTONOMOUS (use for the real win-prob baseline + dashboard)
================================================================================

Read `CLAUDE.md` and `README.md` first and treat CLAUDE.md's rules as hard
constraints (no data leakage, split by game/season, calibration is first-class,
no fabricated numbers, honest baselines). Do NOT ask me to confirm anything —
make the sensible choice and keep going until done. Decisions are pre-decided
below.

Do these in order, committing after each:

1. Fetch real data: `python src/fetch_data.py --seasons 2023-24 2024-25 --games 300`
2. Train + evaluate: `python src/model.py --data data --test-seasons 2024-25`
3. Read `outputs/summary.json` and write the REAL metrics into the README results
   table (replace the placeholder). Never invent numbers.
4. Regenerate the real curve (`python src/plot_game.py`) and reliability diagram;
   make sure the README points at them and remove the "synthetic/illustrative"
   note once they're real.
5. Build a Streamlit dashboard `app.py` at repo root: dropdown to pick a game →
   live win-prob curve + metrics table + reliability diagram + WPA "most clutch
   plays". Cache data + model so it's laptop-fast. Add `streamlit` to
   requirements.txt.
6. Commit everything with clear messages.

If a step fails, fix it and continue; only stop when all 6 are done.

================================================================================
PROMPT 2 — INTERACTIVE (use later for the player-points model; run plain `claude`)
================================================================================

Read `CLAUDE.md` first. We're adding a NEW, SEPARATE model under
`src/player_points/` — do not mix it into win-prob. Goal: project a player's
points for an upcoming game as a DISTRIBUTION (expected value + range), not a
single number. Propose a plan and check with me before coding.

Design notes to discuss:
- Target/metric: MAE + interval coverage (does the 80% range contain the actual
  80% of the time?) vs a "season average" baseline. Leakage-safe backtest.
- Features: projected minutes (the dominant driver), recent + season usage and
  scoring rate, opponent defensive rating + pace, home/away, days rest / B2B.
- News: stub a pluggable `news.py` for injuries / inactives / starting lineups.
  nba_api does NOT provide tomorrow's injury status — flag where a real source
  must plug in, and let me pick the source before you build a scraper.
- No new dependencies or data sources without asking me first.
