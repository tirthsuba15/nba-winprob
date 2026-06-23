"""
NBA Win Probability + Player Props Dashboard — Streamlit app.

Two tabs:
  1. Win Probability — live curve, clutch plays (WPA), metrics, reliability,
     season-wide player WPA leaderboard.
  2. Player Props — project a player's points vs an opponent as a distribution
     (expected value + 80% interval) with the top contributing features.

Run:
  streamlit run app.py
"""
from __future__ import annotations
import os
import sys
import pickle
import json

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from features import build_dataset, FEATURES                       # win-prob
from wpa import compute_wpa
from plot_game import plot_curve
from team_names import team_name, team_name_from_abbrev
from player_points.features import build_features as build_pp_features, FEATURES as PP_FEATURES
from player_points.odds import prob_over, round_to_half
from player_points.model import calibrate_p


# ── display helpers ──────────────────────────────────────────────────────────────
def _fmt_date(d) -> str:
    """'2025-04-13' -> 'Apr 13, 2025' (falls back to the raw value)."""
    try:
        return pd.Timestamp(d).strftime("%b %d, %Y")
    except Exception:
        return str(d)


def _quarter_label(period) -> str:
    p = int(period)
    if p <= 4:
        return f"Q{p}"
    n = p - 4
    return "OT" if n == 1 else f"OT{n}"


def _time_left(secs_left, period) -> str:
    """Seconds remaining within the current quarter, as mm:ss (clamped at 0)."""
    t = int(secs_left) - (4 - int(period)) * 720
    t = max(0, t)
    return f"{t // 60}:{t % 60:02d}"

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NBA Win Probability + Player Props",
    page_icon="🏀",
    layout="wide",
)


# ── cached loaders: win probability ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    path = "outputs/model_calibrated.pkl"
    if not os.path.exists(path):
        st.error("Calibrated model not found. Run: python src/model.py --data data --test-seasons 2024-25")
        st.stop()
    with open(path, "rb") as fh:
        return pickle.load(fh)


@st.cache_data
def load_raw_data():
    games = pd.read_csv("data/games.csv")
    moments = pd.read_csv("data/moments.csv")
    return games, moments


@st.cache_data
def load_dataset():
    games, moments = load_raw_data()
    return build_dataset(moments, games), games


@st.cache_data
def compute_wpa_all():
    df, _ = load_dataset()
    clf = load_model()
    return compute_wpa(df, clf)


@st.cache_data
def load_metrics():
    path = "outputs/summary.json"
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


# ── cached loaders: player points ───────────────────────────────────────────────
@st.cache_resource
def load_pp_models():
    path = "outputs/player_points_models.pkl"
    if not os.path.exists(path):
        return None
    from player_points.model import validate_bundle
    with open(path, "rb") as fh:
        return validate_bundle(pickle.load(fh))


@st.cache_data
def load_pp_features():
    path = "data/player_gamelogs.csv"
    if not os.path.exists(path):
        return None
    raw = pd.read_csv(path)
    return build_pp_features(raw)


@st.cache_data
def load_pp_summary():
    path = "outputs/player_points_summary.json"
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


# ── helpers: player points prediction ───────────────────────────────────────────
def _opp_abbrev_from_matchup(matchup: str) -> str | None:
    """'GSW vs. BOS' or 'GSW @ BOS' -> 'BOS'."""
    s = str(matchup)
    for sep in (" vs. ", " @ "):
        if sep in s:
            return s.split(sep)[1].strip()
    return None


@st.cache_data
def opponent_options(df_features: pd.DataFrame) -> list[str]:
    opps = df_features["matchup"].apply(_opp_abbrev_from_matchup).dropna().unique()
    return sorted(o for o in opps if o)


def build_projection_row(df_features, player_name, opp_abbrev, game_date):
    """Build a single pre-game feature row (1-row DataFrame) for a player vs an
    opponent. Start from the player's most recent game's features, refresh
    rest/home and (if given) opponent defensive stats. Returns (row_df, latest)
    or (None, None). pred_minutes is added later by the bundle's minutes model.
    """
    mask = df_features["player_name"] == player_name
    if not mask.any():
        return None, None
    player_rows = df_features[mask].sort_values("game_date")
    latest = player_rows.iloc[-1].copy()

    # carry min_season_avg too — the minutes sub-model needs it
    cols = PP_FEATURES + (["min_season_avg"] if "min_season_avg" in latest else [])
    row = latest[cols].astype(float).copy()

    last_date = pd.Timestamp(latest["game_date"])
    next_date = pd.Timestamp(game_date)
    rest_days = max(1, (next_date - last_date).days)
    row["days_rest"] = min(rest_days, 10)
    row["b2b"] = float(rest_days == 1)

    if opp_abbrev:
        opp_def = df_features[
            df_features["matchup"].apply(_opp_abbrev_from_matchup) == opp_abbrev
        ].sort_values("game_date")
        if not opp_def.empty:
            ol = opp_def.iloc[-1]
            row["opp_def_rating"] = float(ol["opp_def_rating"])
            row["opp_pace"] = float(ol["opp_pace"])

    return row.to_frame().T[cols], latest


def _points_X(row_df, bundle):
    """Append pred_minutes (from the minutes sub-model) to a feature frame."""
    from player_points.model import points_matrix
    return points_matrix(row_df, bundle["minutes"])


def predict_points(bundle, row_df):
    """Return (mean, lo, hi) with quantile-crossing fix."""
    X = _points_X(row_df, bundle)
    y_mean = float(bundle["mean"].predict(X)[0])
    y_lo = float(bundle["lo"].predict(X)[0])
    y_hi = float(bundle["hi"].predict(X)[0])
    y_lo = max(0.0, min(y_lo, y_mean))
    y_hi = max(y_hi, y_mean)
    return y_mean, y_lo, y_hi


def feature_contributions(bundle, row_df):
    """Per-prediction SHAP-style contributions via XGBoost pred_contribs.

    Uses the points mean model over the full feature list (incl. pred_minutes).
    """
    import xgboost as xgb
    feats = bundle.get("features", PP_FEATURES)
    X = _points_X(row_df, bundle)
    dm = xgb.DMatrix(X, feature_names=feats)
    contribs = bundle["mean"].get_booster().predict(dm, pred_contribs=True)[0]
    pairs = list(zip(feats, contribs[:-1]))
    return sorted(pairs, key=lambda p: -abs(p[1]))


@st.cache_data
def props_leaderboard(opp_abbrev: str | None, game_date: str) -> pd.DataFrame:
    """Project every player vs the chosen opponent; return an over/under table.

    Each player's line defaults to their own season-average-to-date rounded to
    0.5. Batched prediction keeps it laptop-fast. Informational only.
    """
    df = load_pp_features()
    bundle = load_pp_models()
    if df is None or bundle is None:
        return pd.DataFrame()

    rows_df, meta = [], []
    for p in df["player_name"].dropna().unique():
        row_df, latest = build_projection_row(df, p, opp_abbrev, game_date)
        if row_df is None:
            continue
        rows_df.append(row_df)
        season_avg = float(latest["pts_season_avg"]) if "pts_season_avg" in latest else 0.0
        meta.append((p, season_avg))

    if not rows_df:
        return pd.DataFrame()

    from player_points.model import points_matrix
    batch = pd.concat(rows_df, ignore_index=True)
    M = points_matrix(batch, bundle["minutes"])
    y_mean = bundle["mean"].predict(M).clip(min=0)
    y_lo = bundle["lo"].predict(M).clip(min=0)
    y_hi = bundle["hi"].predict(M).clip(min=0)
    y_lo = np.minimum(y_lo, y_mean)
    y_hi = np.maximum(y_hi, y_mean)

    rows = []
    for i, (p, savg) in enumerate(meta):
        base = savg if savg == savg else float(y_mean[i])   # NaN -> projection
        line = max(0.0, round_to_half(base))
        po = calibrate_p(bundle, prob_over(float(y_mean[i]), float(y_lo[i]), float(y_hi[i]), line))
        rows.append({
            "Player": p,
            "Projected": round(float(y_mean[i]), 1),
            "Line": line,
            "Over %": round(po * 100),
            "Lean": "Over" if po >= 0.5 else "Under",
        })
    return pd.DataFrame(rows).sort_values("Projected", ascending=False).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — WIN PROBABILITY
# ═══════════════════════════════════════════════════════════════════════════════
def render_winprob_tab(wpa_df, games):
    st.subheader("Pick a game")
    c1, c2 = st.columns([3, 1])

    def game_label(gid):
        row = games[games["game_id"] == gid].iloc[0]
        return (f"{_fmt_date(row['game_date'])} — "
                f"{team_name(row['away_team_id'])} @ {team_name(row['home_team_id'])} "
                f"({int(row['away_score'])}–{int(row['home_score'])})")

    with c2:
        season_filter = st.selectbox(
            "Season",
            ["All"] + sorted(games["season"].unique().tolist(), reverse=True),
            key="wp_season_filter",
        )
    pool = games if season_filter == "All" else games[games["season"] == season_filter]
    game_ids = pool.sort_values("game_date", ascending=False)["game_id"].tolist()

    with c1:
        selected_id = st.selectbox("Game", game_ids, format_func=game_label, key="wp_game")

    meta = games[games["game_id"] == selected_id].iloc[0].to_dict()
    game_df = wpa_df[wpa_df["game_id"] == selected_id].sort_values("secs_left", ascending=False)

    away_nm, home_nm = team_name(meta["away_team_id"]), team_name(meta["home_team_id"])
    winner_nm = home_nm if meta["home_win"] else away_nm
    st.markdown(
        f"**{_fmt_date(meta['game_date'])}** — {away_nm} @ {home_nm}  "
        f"| Final: {int(meta['home_score'])}–{int(meta['away_score'])}  "
        f"| **{winner_nm} win** | Season: {meta['season']}"
    )

    if game_df.empty:
        st.warning("No play-by-play data for this game.")
    else:
        probs = game_df["win_prob"].to_numpy()
        fig = plot_curve(game_df, probs, meta)
        st.pyplot(fig, use_container_width=True)

    # clutch plays
    st.subheader("Most clutch plays — this game")
    if not game_df.empty:
        gdc = game_df.copy()
        gdc["swing_for_winner"] = np.where(
            gdc["label_home_win"] == 1, gdc["wpa"], -gdc["wpa"]
        )
        top = gdc.sort_values("swing_for_winner", ascending=False).head(10)

        disp = pd.DataFrame({
            "Player":    top["player"] if "player" in top.columns else "",
            "Play":      top["description"] if "description" in top.columns else "",
            "Quarter":   top["period"].apply(_quarter_label),
            "Time left": top.apply(lambda r: _time_left(r["secs_left"], r["period"]), axis=1),
            "Margin":    top["score_margin"].astype(int),
            "Win prob":  top["win_prob"],
            "WP added":  top["wpa"],
        }).reset_index(drop=True)
        disp.index += 1
        st.dataframe(
            disp.style.format(
                {"Margin": "{:+d}", "Win prob": "{:.0%}", "WP added": "{:+.2f}"}
            ),
            use_container_width=True,
        )
    else:
        st.info("No play-by-play data available.")

    # metrics + reliability
    st.divider()
    cl, cr = st.columns([1, 1])
    with cl:
        st.subheader("Model metrics (2024-25 test season)")
        summary = load_metrics()
        if summary:
            met_df = (pd.DataFrame(summary["metrics"]).set_index("model")
                      .rename(columns={"log_loss": "Log-loss", "brier": "Brier",
                                       "accuracy": "Accuracy"}))
            met_df.index.name = "Model"
            st.dataframe(
                met_df.style.format("{:.4f}")
                .highlight_min(subset=["Log-loss", "Brier"], color="#d4edda")
                .highlight_max(subset=["Accuracy"], color="#d4edda"),
                use_container_width=True,
            )
            st.caption(
                f"Train: {summary['n_train_moments']:,} moments "
                f"({', '.join(summary['test_seasons'][0:1])} held out)  "
                f"| Test: {summary['n_test_moments']:,} moments"
            )
        else:
            st.info("Run `python src/model.py --data data --test-seasons 2024-25`.")
    with cr:
        st.subheader("Reliability diagram")
        diag_path = "outputs/reliability_diagram.png"
        if os.path.exists(diag_path):
            st.image(diag_path, use_container_width=True)
        else:
            st.info("Reliability diagram not found.")

    # player WPA leaderboard
    st.divider()
    st.subheader("Player WPA leaderboard (season-wide)")
    season_choice = st.selectbox(
        "Season", sorted(wpa_df["season"].unique().tolist(), reverse=True),
        key="wpa_season",
    )
    season_wpa = wpa_df[wpa_df["season"] == season_choice]
    if "player" in season_wpa.columns:
        board = (season_wpa[season_wpa["player"].str.strip() != ""]
                 .groupby("player")["wpa"].sum().sort_values(ascending=False))
        top_n = st.slider("Show top N players", 5, 50, 20)
        board_df = board.head(top_n).reset_index()
        board_df.columns = ["player", "total_wpa"]
        board_df.index += 1
        st.dataframe(board_df.style.format({"total_wpa": "{:+.3f}"}),
                     use_container_width=True)
    else:
        st.info("Player WPA requires real play-by-play with player names.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PLAYER PROPS
# ═══════════════════════════════════════════════════════════════════════════════
def render_player_props_tab():
    st.subheader("Project a player's points (distribution, not a single number)")

    models = load_pp_models()
    df_pp = load_pp_features()
    if models is None or df_pp is None:
        st.info(
            "Player-points model not built yet. Run:\n\n"
            "```\n"
            "python src/player_points/fetch_gamelogs.py --seasons 2023-24 2024-25\n"
            "python src/player_points/model.py --data data\n"
            "```"
        )
        return

    # pick player + opponent
    players = sorted(df_pp["player_name"].dropna().unique().tolist())
    default_idx = players.index("Stephen Curry") if "Stephen Curry" in players else 0
    c1, c2 = st.columns(2)
    with c1:
        player_name = st.selectbox("Player", players, index=default_idx, key="pp_player")
    with c2:
        opps = opponent_options(df_pp)
        opp_abbrev = (st.selectbox("Opponent", opps, format_func=team_name_from_abbrev,
                                   key="pp_opp") if opps else None)

    opp_full = team_name_from_abbrev(opp_abbrev)
    next_date = (pd.Timestamp(df_pp["game_date"].max()) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    row_df, latest = build_projection_row(df_pp, player_name, opp_abbrev, next_date)
    if row_df is None:
        st.warning(f"No game-log history for {player_name}.")
        return

    y_mean, y_lo, y_hi = predict_points(models, row_df)

    st.markdown(f"**{player_name}** vs **{opp_full}** — projected for next game ({_fmt_date(next_date)})")
    m1, m2 = st.columns([1, 2])
    m1.metric("Projected points", f"{y_mean:.1f}")
    m2.markdown(f"### 80% range: {y_lo:.1f} – {y_hi:.1f} points")
    st.caption(f"There's an estimated 80% chance {player_name} scores between "
               f"**{y_lo:.1f}** and **{y_hi:.1f}** points — the spread is the honest "
               f"uncertainty a single number hides.")

    # ── over / under (informational only) ───────────────────────────────────────
    st.markdown("#### Over / Under")
    st.warning("Informational only — **not betting advice.** The over/under uses a "
               "simple Normal approximation of the projected distribution.")

    season_avg = float(latest["pts_season_avg"]) if "pts_season_avg" in latest else y_mean
    default_line = max(0.0, round_to_half(season_avg))
    line = st.number_input(
        "Line", min_value=0.0, value=float(default_line), step=0.5, key="pp_line",
        help="Defaults to the player's season average to date, rounded to the nearest 0.5.",
    )
    p_over = calibrate_p(models, prob_over(y_mean, y_lo, y_hi, line))
    p_under = 1.0 - p_over
    call = "OVER" if p_over >= 0.5 else "UNDER"
    confidence = max(p_over, p_under)
    st.markdown(
        f"**Projected {y_mean:.1f}** | **Line {line:.1f}** | "
        f"Over **{p_over:.0%}** / Under **{p_under:.0%}**"
    )
    st.markdown(f"**Call: {call}** · confidence **{confidence:.0%}** (calibrated)")
    st.caption("⚠️ **Line = the player's season average to date, NOT a Vegas line.** "
               "This only measures whether they beat their own running average — it is "
               "not a market edge.")

    # ── leaderboard: top scorers vs this opponent ────────────────────────────────
    st.subheader(f"Top projected scorers vs {opp_full}")
    n = st.slider("Show top N", 5, 40, 15, key="pp_leaderboard_n")
    lb = props_leaderboard(opp_abbrev, next_date)
    if lb.empty:
        st.info("No projections available.")
    else:
        def _color_lean(v):
            return "color: #1a7f37; font-weight: 600" if v == "Over" else "color: #b42318; font-weight: 600"
        top_lb = lb.head(n).reset_index(drop=True)
        top_lb.index += 1
        st.dataframe(
            top_lb.style
            .format({"Projected": "{:.1f}", "Line": "{:.1f}", "Over %": "{:.0f}%"})
            .map(_color_lean, subset=["Lean"]),
            use_container_width=True,
        )
        st.caption("Each line defaults to that player's season average to date "
                   "(rounded to 0.5). Informational only — not betting advice.")

    # top contributing features (per-prediction, via XGBoost pred_contribs)
    st.subheader("Top contributing features")
    contribs = feature_contributions(models, row_df)
    contrib_df = pd.DataFrame(contribs[:8], columns=["feature", "contribution"])
    contrib_df["abs"] = contrib_df["contribution"].abs()
    chart_df = contrib_df.set_index("feature")["contribution"]
    st.bar_chart(chart_df, horizontal=True)
    st.caption("Points added/removed from the baseline projection by each feature, "
               "for this specific player-game (XGBoost contribution values). "
               "Positive = pushes the projection up.")

    # show the underlying feature row for transparency
    with st.expander("Feature values used for this projection"):
        fv = pd.DataFrame({"feature": PP_FEATURES,
                           "value": row_df.iloc[0][PP_FEATURES].to_numpy(dtype=float)})
        st.dataframe(fv.style.format({"value": "{:.2f}"}), use_container_width=True)

    # backtest honesty note
    st.divider()
    summ = load_pp_summary()
    if summ:
        st.subheader("Backtest (train 2023-24 → test 2024-25)")
        res_df = pd.DataFrame(summ["results"]).set_index("model")
        st.dataframe(res_df.style.format(
            {"mae": "{:.4f}", "coverage_80": "{:.3f}", "interval_width": "{:.2f}"},
            na_rep="—",
        ), use_container_width=True)
        st.caption(
            f"{summ['n_players_test']:,} players, {summ['n_test']:,} test player-games. "
            "The model barely beats a season-average baseline — the variance that history "
            "can't capture lives in **minutes** (injuries, lineup changes). That signal "
            "needs an injury/lineup feed; `src/player_points/news.py` is the stubbed seam "
            "where it plugs in. The 80% interval is how we express that uncertainty honestly."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
st.sidebar.title("🏀 NBA Models")
st.sidebar.markdown(
    "A calibrated win-probability model and a separate player-points "
    "projection model — both built with leakage-safe backtests."
)

with st.spinner("Loading data and model…"):
    wpa_df = compute_wpa_all()
    _, games = load_dataset()

tab_wp, tab_pp = st.tabs(["📈 Win Probability", "🎯 Player Props"])
with tab_wp:
    st.title("NBA Win Probability")
    render_winprob_tab(wpa_df, games)
with tab_pp:
    st.title("Player Points Projection")
    render_player_props_tab()
