"""
Shared helpers for the data layer: rate-limited retries, parquet I/O, paths.

All nba_api fetches are deliberately polite (small sleeps + exponential backoff)
because stats.nba.com rate-limits and blocks datacenter IPs.
"""
from __future__ import annotations
import os
import time
import pandas as pd

# Regular-season + playoff seasons we target.
DEFAULT_SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
SEASON_TYPES = ["Regular Season", "Playoffs"]

# Layout roots
DATA_ROOT = "data"
RAW = os.path.join(DATA_ROOT, "raw")
GAMES_DIR = os.path.join(RAW, "games")
PBP_DIR = os.path.join(RAW, "pbp")
PLAYER_LOGS_DIR = os.path.join(RAW, "player_logs")
TEAM_LOGS_DIR = os.path.join(RAW, "team_logs")
SNAPSHOT_DIR = os.path.join(DATA_ROOT, "snapshots", "injuries")
LOG_DIR = os.path.join(DATA_ROOT, ".fetch_log")
CATALOG = os.path.join(DATA_ROOT, "catalog.duckdb")


def _slug(season_type: str) -> str:
    return season_type.lower().replace(" ", "_")


def sleep(s: float = 0.6):
    time.sleep(s)


def retry(fn, tries: int = 5, base: float = 0.8):
    """Call fn() with exponential backoff on any exception."""
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:               # network hiccup / rate limit
            last = e
            time.sleep(base * (2 ** i))
    raise last


def ensure_dirs():
    for d in (GAMES_DIR, PBP_DIR, PLAYER_LOGS_DIR, TEAM_LOGS_DIR,
              SNAPSHOT_DIR, LOG_DIR):
        os.makedirs(d, exist_ok=True)


def write_parquet(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)


def read_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


# ── partitioned paths ────────────────────────────────────────────────────────────
def games_path(season: str, season_type: str) -> str:
    return os.path.join(GAMES_DIR, f"season={season}", f"games_{_slug(season_type)}.parquet")


def player_logs_path(season: str, season_type: str) -> str:
    return os.path.join(PLAYER_LOGS_DIR, f"season={season}", f"player_logs_{_slug(season_type)}.parquet")


def team_logs_path(season: str, season_type: str) -> str:
    return os.path.join(TEAM_LOGS_DIR, f"season={season}", f"team_logs_{_slug(season_type)}.parquet")


def pbp_path(season: str, game_id: str) -> str:
    return os.path.join(PBP_DIR, f"season={season}", f"game_id={game_id}.parquet")


def load_env_key(name: str = "BALLDONTLIE_API_KEY") -> str | None:
    """Read a key from the process env or a local .env (never committed)."""
    if os.environ.get(name):
        return os.environ[name]
    env_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_path):
        for line in open(env_path):
            line = line.strip()
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip()
    return None
