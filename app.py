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
from player_points.features import build_features as build_pp_features, FEATURES as PP_FEATURES

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
    with open(path, "rb") as fh:
        return pickle.load(fh)


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
    """Build a single pre-game feature row for a player vs an opponent.

    Mirrors predict.project_player: start from the player's most recent game's
    rolling features, then refresh rest/home and (if given) opponent defensive
    stats. Returns (feature_array, latest_row) or (None, None).
    """
    mask = df_features["player_name"] == player_name
    if not mask.any():
        return None, None
    player_rows = df_features[mask].sort_values("game_date")
    latest = player_rows.iloc[-1].copy()

    row = latest[PP_FEATURES].astype(float).copy()

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
            row["opp_pts_allowed"] = float(ol["opp_pts_allowed"])
            row["opp_pace"] = float(ol["opp_pace"])

    X = row.to_numpy(dtype=float).reshape(1, -1)
    return X, latest


def predict_points(models, X):
    """Return (mean, lo, hi) with quantile-crossing fix."""
    y_mean = float(models["mean"].predict(X)[0])
    y_lo = float(models["lo"].predict(X)[0])
    y_hi = float(models["hi"].predict(X)[0])
    y_lo = max(0.0, min(y_lo, y_mean))
    y_hi = max(y_hi, y_mean)
    return y_mean, y_lo, y_hi


def feature_contributions(model, X):
    """Per-prediction SHAP-style contributions via XGBoost pred_contribs.

    No extra dependency — uses the booster directly. Returns list of
    (feature, contribution) sorted by absolute impact; bias term dropped.
    """
    import xgboost as xgb
    booster = model.get_booster()
    dm = xgb.DMatrix(X, feature_names=PP_FEATURES)
    contribs = booster.predict(dm, pred_contribs=True)[0]  # (n_features + 1,)
    pairs = list(zip(PP_FEATURES, contribs[:-1]))
    return sorted(pairs, key=lambda p: -abs(p[1]))


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — WIN PROBABILITY
# ═══════════════════════════════════════════════════════════════════════════════
def render_winprob_tab(wpa_df, games):
    st.subheader("Pick a game")
    c1, c2 = st.columns([3, 1])

    def game_label(gid):
        row = games[games["game_id"] == gid].iloc[0]
        return (f"{row['game_date']}  {row['away_team_id']} @ {row['home_team_id']}  "
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

    winner_str = "Home" if meta["home_win"] else "Away"
    st.markdown(
        f"**{meta['game_date']}** — {meta['away_team_id']} @ {meta['home_team_id']}  "
        f"| Final: {int(meta['home_score'])}–{int(meta['away_score'])}  "
        f"| **{winner_str} win** | Season: {meta['season']}"
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
        cols = ["period", "secs_left", "score_margin", "win_prob", "wpa", "swing_for_winner"]
        if "description" in gdc.columns:
            cols = ["description"] + cols
        if "player" in gdc.columns:
            cols = ["player"] + cols
        top_plays = (gdc.sort_values("swing_for_winner", ascending=False)
                     .head(10)[cols].reset_index(drop=True))
        top_plays.index += 1
        st.dataframe(
            top_plays.style.format(
                {"win_prob": "{:.1%}", "wpa": "{:+.3f}", "swing_for_winner": "{:+.3f}",
                 "score_margin": "{:+d}", "secs_left": "{:.0f}"}
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
            met_df = pd.DataFrame(summary["metrics"]).set_index("model")
            st.dataframe(
                met_df.style.format("{:.4f}")
                .highlight_min(subset=["log_loss", "brier"], color="#d4edda")
                .highlight_max(subset=["accuracy"], color="#d4edda"),
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
        opp_abbrev = st.selectbox("Opponent", opps, key="pp_opp") if opps else None

    next_date = (pd.Timestamp(df_pp["game_date"].max()) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    X, latest = build_projection_row(df_pp, player_name, opp_abbrev, next_date)
    if X is None:
        st.warning(f"No game-log history for {player_name}.")
        return

    y_mean, y_lo, y_hi = predict_points(models, X)

    st.markdown(f"**{player_name}** vs **{opp_abbrev or '?'}** — projected for next game ({next_date})")
    m1, m2, m3 = st.columns(3)
    m1.metric("Expected points", f"{y_mean:.1f}")
    m2.metric("80% interval low (q10)", f"{y_lo:.1f}")
    m3.metric("80% interval high (q90)", f"{y_hi:.1f}")
    st.caption(f"There's an estimated 80% chance {player_name} scores between "
               f"**{y_lo:.1f}** and **{y_hi:.1f}** points — the spread is the honest "
               f"uncertainty a single number hides.")

    # top contributing features (per-prediction, via XGBoost pred_contribs)
    st.subheader("Top contributing features")
    contribs = feature_contributions(models["mean"], X)
    contrib_df = pd.DataFrame(contribs[:8], columns=["feature", "contribution"])
    contrib_df["abs"] = contrib_df["contribution"].abs()
    chart_df = contrib_df.set_index("feature")["contribution"]
    st.bar_chart(chart_df, horizontal=True)
    st.caption("Points added/removed from the baseline projection by each feature, "
               "for this specific player-game (XGBoost contribution values). "
               "Positive = pushes the projection up.")

    # show the underlying feature row for transparency
    with st.expander("Feature values used for this projection"):
        fv = pd.DataFrame({"feature": PP_FEATURES, "value": X[0]})
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
