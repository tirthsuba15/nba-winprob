"""
Project a player's points for an upcoming game as a distribution.

Outputs: expected points + 80% prediction interval (10th–90th percentile).
Loads the trained models from outputs/player_points_models.pkl.

Usage:
    # Project by player name (partial match OK):
    python src/player_points/predict.py --player "LeBron James" --opp-team LAL

    # With a specific game date:
    python src/player_points/predict.py --player "Stephen Curry" --opp-team BOS --date 2025-03-01

    # Show top projected scorers for a given opponent:
    python src/player_points/predict.py --top 20 --opp-team MIA
"""
from __future__ import annotations
import argparse
import os
import pickle
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

try:
    from .features import build_features, FEATURES
    from .news import apply_adjustments
    from .odds import prob_over, round_to_half
    from .model import calibrate_p, validate_bundle
except ImportError:
    from features import build_features, FEATURES
    from news import apply_adjustments
    from odds import prob_over, round_to_half
    from model import calibrate_p, validate_bundle


def _load_models(path="outputs/player_points_models.pkl"):
    if not os.path.exists(path):
        sys.exit(f"Models not found at {path}. Run: python src/player_points/model.py --data data")
    with open(path, "rb") as fh:
        return validate_bundle(pickle.load(fh))


def _load_gamelogs(data_dir="data"):
    path = os.path.join(data_dir, "player_gamelogs.csv")
    if not os.path.exists(path):
        sys.exit(f"Game logs not found at {path}. Run: python src/player_points/fetch_gamelogs.py --seasons 2023-24 2024-25")
    return pd.read_csv(path)


def project_player(player_name: str, opp_abbrev: str | None, game_date: str,
                   models, df_features: pd.DataFrame) -> dict | None:
    """
    Build a synthetic feature row for an upcoming game and predict.

    Uses the player's most recent game's features as the base row, then
    updates: home flag, days_rest, b2b, and opp defensive stats for the
    given opponent.

    Returns dict with keys: player_name, expected, lo, hi, or None if not found.
    """
    mask = df_features["player_name"].str.contains(player_name, case=False, na=False)
    if not mask.any():
        print(f"  Player '{player_name}' not found in game logs.")
        return None

    player_rows = df_features[mask].sort_values("game_date")
    player_id = int(player_rows.iloc[0]["player_id"])

    # Use latest game's rolling features as the starting point
    latest = player_rows.iloc[-1].copy()
    last_date = pd.Timestamp(latest["game_date"])
    next_date = pd.Timestamp(game_date)
    rest_days = max(1, (next_date - last_date).days)

    row = latest[FEATURES].copy()
    row["days_rest"] = min(rest_days, 10)
    row["b2b"] = int(rest_days == 1)

    # Override opponent defensive stats if opp_abbrev provided
    if opp_abbrev:
        opp_mask = (
            df_features["matchup"].str.contains(opp_abbrev, case=False, na=False)
        )
        if opp_mask.any():
            opp_latest = df_features[opp_mask].sort_values("game_date").iloc[-1]
            row["opp_def_rating"] = opp_latest.get("opp_def_rating", row["opp_def_rating"])
            row["opp_pace"]        = opp_latest.get("opp_pace", row["opp_pace"])

    # news.py hook — plug injury adjustments in here
    row_dict = apply_adjustments(row.to_dict(), player_id, game_date)
    if row_dict.get("injury_flag") and row_dict.get("status") == "Out":
        print(f"  {player_name} marked Out per news.py — skipping.")
        return None

    # build a 1-row frame, append pred_minutes via the minutes sub-model
    try:
        from .model import points_matrix
    except ImportError:
        from model import points_matrix
    fr = pd.DataFrame([{f: row_dict.get(f, row[f]) for f in FEATURES}])
    fr["min_season_avg"] = float(latest["min_season_avg"]) if "min_season_avg" in latest else np.nan
    X = points_matrix(fr, models["minutes"])
    y_mean = float(models["mean"].predict(X)[0])
    y_lo   = float(models["lo"].predict(X)[0])
    y_hi   = float(models["hi"].predict(X)[0])

    # Enforce ordering
    y_lo  = min(y_lo, y_mean)
    y_hi  = max(y_hi, y_mean)
    y_lo  = max(0.0, y_lo)

    season_avg = float(latest["pts_season_avg"]) if "pts_season_avg" in latest else y_mean
    return {
        "player_name": latest["player_name"],
        "expected": round(y_mean, 1),
        "lo_80":    round(y_lo, 1),
        "hi_80":    round(y_hi, 1),
        "interval": f"{y_lo:.1f}–{y_hi:.1f}",
        "mean_raw": y_mean,
        "lo_raw":   y_lo,
        "hi_raw":   y_hi,
        "season_avg": season_avg,
    }


def _default_line(result):
    """Season-avg-to-date line; fall back to the projection if no prior average."""
    sa = result.get("season_avg")
    base = sa if (sa is not None and sa == sa) else result["mean_raw"]  # NaN check
    return round_to_half(max(0.0, base))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--player", default=None, help="Player name (partial match)")
    ap.add_argument("--opp-team", default=None, help="Opponent team abbreviation, e.g. LAL")
    ap.add_argument("--date", default=None,
                    help="Upcoming game date YYYY-MM-DD (default: latest in data + 1 day)")
    ap.add_argument("--top", type=int, default=None,
                    help="Show top N projected scorers vs --opp-team")
    ap.add_argument("--line", type=float, default=None,
                    help="Over/under line. Default: player's season avg to date "
                         "rounded to nearest 0.5 (informational only, not betting advice)")
    ap.add_argument("--data", default="data")
    ap.add_argument("--models", default="outputs/player_points_models.pkl")
    args = ap.parse_args()

    models = _load_models(args.models)
    raw = _load_gamelogs(args.data)

    print("Building features …")
    df = build_features(raw)

    if args.date is None:
        latest_date = pd.Timestamp(df["game_date"].max())
        game_date = (latest_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        game_date = args.date

    if args.top is not None:
        # Project top N players in the dataset
        players = df["player_name"].unique()
        rows = []
        for p in players:
            result = project_player(p, args.opp_team, game_date, models, df)
            if result:
                line = args.line if args.line is not None else _default_line(result)
                p_over = calibrate_p(models, prob_over(result["mean_raw"], result["lo_raw"], result["hi_raw"], line))
                result["line"] = line
                result["over_pct"] = round(p_over * 100)
                result["lean"] = "Over" if p_over >= 0.5 else "Under"
                rows.append(result)
        out = pd.DataFrame(rows).sort_values("expected", ascending=False).head(args.top)
        out.index = range(1, len(out) + 1)
        print(f"\nTop {args.top} projected scorers vs {args.opp_team or 'any'} on {game_date}:\n")
        print(out[["player_name", "expected", "interval", "line", "over_pct", "lean"]]
              .rename(columns={"player_name": "Player", "expected": "Projected",
                               "interval": "80% range", "line": "Line",
                               "over_pct": "Over %", "lean": "Lean"})
              .to_string())
        print("\n(informational only — not betting advice)")
    elif args.player:
        result = project_player(args.player, args.opp_team, game_date, models, df)
        if result:
            line = args.line if args.line is not None else _default_line(result)
            p_over = calibrate_p(models, prob_over(result["mean_raw"], result["lo_raw"], result["hi_raw"], line))
            print(f"\n{result['player_name']} vs {args.opp_team or '?'} on {game_date}:")
            print(f"  Expected: {result['expected']} pts")
            print(f"  80% interval: {result['interval']} pts")
            print(f"  Projected {result['expected']} | Line {line:.1f} | "
                  f"Over {p_over:.0%} / Under {1 - p_over:.0%}")
            print("  (informational only — not betting advice)")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
