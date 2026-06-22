"""
NBA Win Probability Dashboard — Streamlit app.

Pick a game from the dropdown to see:
  - Live win-probability curve
  - Most clutch plays from that game (WPA)
  - Season metrics table
  - Reliability diagram

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

from features import build_dataset, FEATURES
from wpa import compute_wpa
from plot_game import plot_curve

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NBA Win Probability",
    page_icon="🏀",
    layout="wide",
)


# ── cached loaders ─────────────────────────────────────────────────────────────
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


# ── data ───────────────────────────────────────────────────────────────────────
with st.spinner("Loading data and model…"):
    wpa_df = compute_wpa_all()
    _, games = load_dataset()

# ── sidebar — game picker ───────────────────────────────────────────────────────
st.sidebar.title("🏀 NBA Win Probability")
st.sidebar.markdown("Pick a game to explore its live win-probability curve and clutch plays.")

game_ids = games.sort_values("game_date", ascending=False)["game_id"].tolist()

def game_label(gid):
    row = games[games["game_id"] == gid].iloc[0]
    return f"{row['game_date']}  {row['away_team_id']} @ {row['home_team_id']}  ({int(row['away_score'])}–{int(row['home_score'])})"

selected_id = st.sidebar.selectbox(
    "Game",
    game_ids,
    format_func=game_label,
)

season_filter = st.sidebar.selectbox(
    "Filter by season",
    ["All"] + sorted(games["season"].unique().tolist(), reverse=True),
)
if season_filter != "All":
    filtered_ids = games[games["season"] == season_filter]["game_id"].tolist()
    selected_id = st.sidebar.selectbox(
        "Game (filtered)",
        filtered_ids,
        format_func=game_label,
        key="game_filtered",
    )

# ── main — win-prob curve ───────────────────────────────────────────────────────
meta = games[games["game_id"] == selected_id].iloc[0].to_dict()
game_df = wpa_df[wpa_df["game_id"] == selected_id].sort_values(
    "secs_left", ascending=False
)

st.title("NBA Win Probability")
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

# ── clutch plays for this game ──────────────────────────────────────────────────
st.subheader("Most clutch plays — this game")
if not game_df.empty:
    game_df_copy = game_df.copy()
    game_df_copy["swing_for_winner"] = np.where(
        game_df_copy["label_home_win"] == 1,
        game_df_copy["wpa"],
        -game_df_copy["wpa"],
    )
    cols = ["period", "secs_left", "score_margin", "win_prob", "wpa", "swing_for_winner"]
    if "description" in game_df_copy.columns:
        cols = ["description"] + cols
    if "player" in game_df_copy.columns:
        cols = ["player"] + cols
    top_plays = (
        game_df_copy.sort_values("swing_for_winner", ascending=False)
        .head(10)[cols]
        .reset_index(drop=True)
    )
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

# ── metrics table + reliability diagram ────────────────────────────────────────
st.divider()
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Model metrics (2024-25 test season)")
    summary = load_metrics()
    if summary:
        met_df = pd.DataFrame(summary["metrics"]).set_index("model")
        st.dataframe(
            met_df.style.format("{:.4f}").highlight_min(
                subset=["log_loss", "brier"], color="#d4edda"
            ).highlight_max(
                subset=["accuracy"], color="#d4edda"
            ),
            use_container_width=True,
        )
        st.caption(
            f"Train: {summary['n_train_moments']:,} moments ({', '.join(summary['test_seasons'][0:1])} held out)  "
            f"| Test: {summary['n_test_moments']:,} moments"
        )
    else:
        st.info("Run `python src/model.py --data data --test-seasons 2024-25` to generate metrics.")

with col_right:
    st.subheader("Reliability diagram")
    diag_path = "outputs/reliability_diagram.png"
    if os.path.exists(diag_path):
        st.image(diag_path, use_container_width=True)
    else:
        st.info("Reliability diagram not found.")

# ── player WPA leaderboard (season-wide) ───────────────────────────────────────
st.divider()
st.subheader("Player WPA leaderboard (season-wide)")
season_choice = st.selectbox(
    "Season",
    sorted(wpa_df["season"].unique().tolist(), reverse=True),
    key="wpa_season",
)
season_wpa = wpa_df[wpa_df["season"] == season_choice]

if "player" in season_wpa.columns:
    board = (
        season_wpa[season_wpa["player"].str.strip() != ""]
        .groupby("player")["wpa"]
        .sum()
        .sort_values(ascending=False)
    )
    top_n = st.slider("Show top N players", 5, 50, 20)
    board_df = board.head(top_n).reset_index()
    board_df.columns = ["player", "total_wpa"]
    board_df.index += 1
    st.dataframe(
        board_df.style.format({"total_wpa": "{:+.3f}"}),
        use_container_width=True,
    )
else:
    st.info("Player WPA requires real play-by-play data with player names.")
