"""
LIVE injury / availability feed (for FORWARD predictions only).

Source: balldontlie API (https://api.balldontlie.io/v1/player_injuries).
Historical training availability does NOT come from here — it is inferred
leakage-safely from box scores in `availability.py`. This module is purely for
*today's* injuries when projecting an upcoming game.

Auth: reads BALLDONTLIE_API_KEY from the environment or a gitignored .env.
Snapshots: every successful pull is saved, timestamped, under
data/snapshots/injuries/ so we build a forward history we can later backtest.

⚠ Tier note: the player_injuries endpoint is on balldontlie's PAID tier. With a
free/lower-tier key it returns HTTP 401; this module then degrades to an empty
result (no crash) and logs a warning. It activates automatically once the key's
tier includes injuries, or swap in the balldontlie MCP server at the marked seam.

⚠ ID mapping: balldontlie player ids differ from nba_api player ids. Live lookups
match on player NAME (case-insensitive). Pass the name to get_injury_adjustments
for reliable matching; matching by nba_api id alone is not supported.
"""
from __future__ import annotations
import json
import logging
import os
import urllib.request
import urllib.error
from datetime import datetime, timezone

log = logging.getLogger(__name__)

API_URL = "https://api.balldontlie.io/v1/player_injuries"
SNAPSHOT_DIR = os.path.join("data", "snapshots", "injuries")


def _load_key(name: str = "BALLDONTLIE_API_KEY") -> str | None:
    if os.environ.get(name):
        return os.environ[name]
    env_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_path):
        for line in open(env_path):
            line = line.strip()
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip()
    return None


# ── live fetch ────────────────────────────────────────────────────────────────────
def fetch_live_injuries(per_page: int = 100) -> tuple[int, list[dict]]:
    """Return (http_status, injuries). Empty list on any non-200 (incl. 401).

    --- MCP SEAM -------------------------------------------------------------
    To route through the balldontlie MCP server instead of the REST API,
    replace the urllib block below with the MCP tool call and map its result
    into the same list-of-dicts shape.
    -------------------------------------------------------------------------
    """
    key = _load_key()
    if not key:
        log.warning("BALLDONTLIE_API_KEY not set — live injuries unavailable.")
        return 0, []

    results, cursor = [], None
    status = 200
    while True:
        url = f"{API_URL}?per_page={per_page}" + (f"&cursor={cursor}" if cursor else "")
        req = urllib.request.Request(url, headers={"Authorization": key})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                payload = json.loads(r.read())
        except urllib.error.HTTPError as e:
            status = e.code
            if e.code == 401:
                log.warning("balldontlie injuries -> 401 (key's tier lacks the "
                            "player_injuries endpoint). Returning empty.")
            else:
                log.warning("balldontlie injuries -> HTTP %s.", e.code)
            return status, []
        except Exception as e:                       # network error
            log.warning("balldontlie injuries fetch failed: %s", e)
            return -1, []

        results.extend(payload.get("data", []))
        cursor = payload.get("meta", {}).get("next_cursor")
        if not cursor:
            break
    return status, results


def snapshot_injuries() -> str | None:
    """Fetch and save a timestamped snapshot. Returns the path, or None if empty."""
    status, injuries = fetch_live_injuries()
    if not injuries:
        log.info("No injuries returned (status %s) — nothing snapshotted.", status)
        return None
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(SNAPSHOT_DIR, f"{ts}.json")
    with open(path, "w") as fh:
        json.dump({"captured_utc": ts, "count": len(injuries), "data": injuries}, fh)
    log.info("Snapshotted %d injuries -> %s", len(injuries), path)
    return path


def _latest_snapshot() -> list[dict]:
    if not os.path.isdir(SNAPSHOT_DIR):
        return []
    snaps = sorted(f for f in os.listdir(SNAPSHOT_DIR) if f.endswith(".json"))
    if not snaps:
        return []
    with open(os.path.join(SNAPSHOT_DIR, snaps[-1])) as fh:
        return json.load(fh).get("data", [])


# ── interface used by predict.py ───────────────────────────────────────────────────
def get_injury_adjustments(player_id: int, game_date: str, player_name: str | None = None) -> dict:
    """Return pre-game injury info for a player from the latest snapshot.

    Matches on player_name (case-insensitive) because balldontlie ids differ
    from nba_api ids. Returns {} when no live data exists (the common case until
    a paid-tier key or the MCP is wired in) — callers then use history only.
    """
    if not player_name:
        return {}
    injuries = _latest_snapshot()
    if not injuries:
        return {}
    target = player_name.strip().lower()
    for inj in injuries:
        pl = inj.get("player", {})
        full = f"{pl.get('first_name','')} {pl.get('last_name','')}".strip().lower()
        if full == target:
            status = inj.get("status", "")
            return {
                "injury_flag": True,
                "status": status,
                "projected_min": None,
                "notes": inj.get("description", "") or inj.get("return_date", ""),
            }
    return {}


def apply_adjustments(feature_row: dict, player_id: int, game_date: str,
                      player_name: str | None = None) -> dict:
    """Merge injury adjustments into a feature row before prediction."""
    adj = get_injury_adjustments(player_id, game_date, player_name)
    if not adj:
        return feature_row
    row = dict(feature_row)
    if adj.get("projected_min") is not None:
        log.warning("news.py override: %s projected_min=%.1f on %s",
                    player_name or player_id, adj["projected_min"], game_date)
        row["min_roll5"] = adj["projected_min"]
        row["min_roll10"] = adj["projected_min"]
    row["_injury"] = adj
    return row


def main():
    """CLI: take and print a live snapshot (for cron / manual capture)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    status, injuries = fetch_live_injuries()
    print(f"balldontlie status: {status} | injuries returned: {len(injuries)}")
    if injuries:
        path = snapshot_injuries()
        print(f"snapshot saved: {path}")
        for inj in injuries[:5]:
            pl = inj.get("player", {})
            print(f"  {pl.get('first_name','')} {pl.get('last_name','')}: {inj.get('status','')}")
    else:
        print("No injuries available (see tier note in this file's docstring).")


if __name__ == "__main__":
    main()
