"""
Build a small DuckDB catalog of VIEWS over the parquet data layer.

DuckDB reads parquet directly, so the catalog stores no copies — just named
views with globs. Query it from Python or the duckdb CLI:

    import duckdb; con = duckdb.connect("data/catalog.duckdb")
    con.sql("SELECT season, count(*) FROM games GROUP BY 1 ORDER BY 1")

Usage:
    python src/data/catalog.py            # (re)build the views
    python src/data/catalog.py --summary  # build, then print row counts
"""
from __future__ import annotations
import argparse
import os

from common import CATALOG, RAW

VIEWS = {
    "games":       f"{RAW}/games/season=*/games_*.parquet",
    "player_logs": f"{RAW}/player_logs/season=*/player_logs_*.parquet",
    "team_logs":   f"{RAW}/team_logs/season=*/team_logs_*.parquet",
    "pbp":         f"{RAW}/pbp/season=*/game_id=*.parquet",
}


def build(summary: bool = False):
    import duckdb
    os.makedirs(os.path.dirname(CATALOG), exist_ok=True)
    con = duckdb.connect(CATALOG)
    for name, pattern in VIEWS.items():
        # union_by_name tolerates schema drift across partitions.
        matches = __import__("glob").glob(pattern)
        if not matches:
            print(f"  (no files yet for view '{name}': {pattern})")
            continue
        con.execute(
            f"CREATE OR REPLACE VIEW {name} AS "
            f"SELECT * FROM read_parquet('{pattern}', union_by_name=true)"
        )
        print(f"  view {name:12} -> {len(matches):,} parquet file(s)")

    if summary:
        print("\nRow counts:")
        for name in VIEWS:
            try:
                n = con.sql(f"SELECT count(*) FROM {name}").fetchone()[0]
                print(f"  {name:12} {n:>12,}")
            except Exception as e:
                print(f"  {name:12} (unavailable: {str(e)[:60]})")
    con.close()
    print(f"\nCatalog written to {CATALOG}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    build(summary=args.summary)


if __name__ == "__main__":
    main()
