"""
Pluggable interface for pre-game injury and lineup data.

THIS IS A STUB. It returns empty dicts until a real data source is wired in.

─── WHERE TO PLUG IN A REAL SOURCE ────────────────────────────────────────────

Option A — NBA official injury report (PDF, released ~90 min before tip-off):
    URL: https://ak-static.cms.nba.com/referee/injury/Injury-Report_<date>_<time>.pdf
    Parse with pdfplumber or camelot. Map player names → player_id via nba_api.

Option B — ESPN unofficial injury endpoint:
    GET https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries
    Returns JSON; no auth required but unofficial and may break without notice.

Option C — RotoBaller / RotoWire RSS:
    XML feeds with structured injury tags. Requires a paid subscription for
    reliable real-time data; free tiers have ~2-hour delay.

The return dict shape is intentional: callers (features.py) check for
'projected_min' and log a WARNING if the stub is replaced and starts returning
real data — making the seam visible during development.

────────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import logging

log = logging.getLogger(__name__)


def get_injury_adjustments(player_id: int, game_date: str) -> dict:
    """
    Return pre-game injury / lineup info for one player.

    Parameters
    ----------
    player_id : int
        NBA player ID (as in nba_api).
    game_date : str
        ISO date string, e.g. '2025-01-15'. Callers pass the game date so that
        a real implementation can fetch the correct injury report.

    Returns
    -------
    dict with optional keys:
        injury_flag     (bool)   — player is listed as injured/questionable
        status          (str)    — 'Out' | 'Questionable' | 'Probable' | 'GTD'
        projected_min   (float)  — override projected minutes if available
        notes           (str)    — free-text description

    Empty dict means "no adjustment — use historical features as-is."
    """
    # ── STUB: replace this block with a real source ──────────────────────────
    return {}
    # ─────────────────────────────────────────────────────────────────────────


def apply_adjustments(feature_row: dict, player_id: int, game_date: str) -> dict:
    """
    Merge injury adjustments into a feature row before prediction.

    If projected_min is provided, overrides min_roll5 / min_roll10.
    If injury_flag is True and status == 'Out', caller should skip prediction
    entirely and return None (handled in predict.py).
    """
    adj = get_injury_adjustments(player_id, game_date)
    if not adj:
        return feature_row

    row = dict(feature_row)
    if adj.get("projected_min") is not None:
        log.warning(
            "news.py override: player %s projected_min=%.1f on %s",
            player_id, adj["projected_min"], game_date,
        )
        row["min_roll5"] = adj["projected_min"]
        row["min_roll10"] = adj["projected_min"]

    return row
