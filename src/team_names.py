"""
Human-readable NBA team names from nba_api's OFFLINE static data.

`nba_api.stats.static.teams` is bundled with the package — no network call, no
rate limit. We cache the lookups so repeated calls are free.
"""
from __future__ import annotations
from functools import lru_cache


@lru_cache(maxsize=1)
def _by_id() -> dict[int, str]:
    try:
        from nba_api.stats.static import teams
        return {t["id"]: t["full_name"] for t in teams.get_teams()}
    except Exception:
        return {}


@lru_cache(maxsize=1)
def _by_abbrev() -> dict[str, str]:
    try:
        from nba_api.stats.static import teams
        return {t["abbreviation"]: t["full_name"] for t in teams.get_teams()}
    except Exception:
        return {}


def team_name(team_id) -> str:
    """Map a team ID to its full name; falls back to the raw id as a string."""
    try:
        tid = int(team_id)
    except (ValueError, TypeError):
        return str(team_id)
    return _by_id().get(tid, str(team_id))


def team_name_from_abbrev(abbrev) -> str:
    """Map a team abbreviation (e.g. 'BOS') to its full name; falls back to input."""
    if abbrev is None:
        return "?"
    return _by_abbrev().get(str(abbrev), str(abbrev))
