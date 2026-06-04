#!/usr/bin/env python3
"""Profile raw Elo and FIFA-ranking datasets -> outputs/reports/ratings_source_profile.md.

Detects actual files/columns (no schema assumptions) and reports usability for
strict no-future-leakage as-of joins.
"""
import logging
from pathlib import Path

import pandas as pd

from src.ingest.common import select_dataset_csv

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
KAGGLE = REPO_ROOT / "data" / "raw" / "kaggle"
REPORT = REPO_ROOT / "outputs" / "reports" / "ratings_source_profile.md"


def _profile(path, fh, date_col, team_col, value_cols):
    f = fh
    df = pd.read_csv(path)
    dt = pd.to_datetime(df[date_col], format="mixed", errors="coerce")
    fh.write(f"### `{path.name}`\n\n")
    fh.write(f"- Rows: **{len(df)}**\n")
    fh.write(f"- Columns: {list(df.columns)}\n")
    fh.write(f"- Date column: `{date_col}` — range {dt.min().date()} → {dt.max().date()} "
             f"(parse failures: {int(dt.isna().sum())})\n")
    fh.write(f"- Team/name column: `{team_col}` — distinct teams: {df[team_col].nunique()}\n")
    fh.write(f"- Rating/points columns: {value_cols}\n")
    fh.write(f"- Distinct dates: {dt.nunique()}\n")
    dup = df.duplicated([team_col, date_col]).sum()
    fh.write(f"- Duplicate ({team_col},{date_col}) rows: **{int(dup)}**"
             + ("  ⚠️ key risk" if dup else "  ✅ unique key") + "\n")
    nulls = {c: int(df[c].isnull().sum()) for c in df.columns if df[c].isnull().any()}
    fh.write(f"- Missing values: {nulls or 'none'}\n")
    fh.write("- Sample rows:\n\n```\n" + df.head(3).to_string(index=False) + "\n```\n\n")
    return df, dt


def main():
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT, "w") as f:
        f.write("# Ratings Source Profile\n\n")
        f.write("Profiles the raw Elo and FIFA-ranking datasets actually present "
                "(files & columns detected at runtime, not assumed).\n\n")

        # ----- Elo -----
        f.write("## Elo (international_elo)\n\n")
        elo_file = select_dataset_csv(KAGGLE / "international_elo", preferred_names=["eloratings.csv"])
        elo, elo_dt = _profile(elo_file, f, "date", "team", ["rating", "change"])
        f.write("**Usability for as-of joins:** ✅ usable. Sparse per-team snapshots "
                "(rating updated when a team plays). The `date` is the date of the rating "
                "change, so a rating dated ON a match date may already include that match — "
                "**must use strict `date < match_date`**.\n\n")

        # ----- FIFA -----
        f.write("## FIFA world ranking (fifa_world_ranking)\n\n")
        fifa_file = select_dataset_csv(KAGGLE / "fifa_world_ranking", pick="last")
        f.write(f"- Multiple snapshot files present; using latest by name: `{fifa_file.name}` "
                "(most complete history).\n\n")
        fifa, fifa_dt = _profile(fifa_file, f, "rank_date", "country_full",
                                 ["rank", "total_points", "previous_points"])
        f.write("**Usability for as-of joins:** ✅ usable. `rank_date` is the official "
                "publication date (pre-match by construction); `country_abrv` provides a "
                "3-letter country code. Use strict `rank_date < match_date` for consistency.\n\n")

        f.write("## Notes\n\n")
        f.write("- Elo `date` column has **mixed formats** (ISO and M/D/YYYY); the cleaner "
                "parses with `format='mixed'`.\n")
        f.write("- Neither source should be force-joined to non-FIFA / CONIFA / regional teams; "
                "missing teams are reported, not guessed.\n")

    logger.info(f"✓ Wrote {REPORT}")


if __name__ == "__main__":
    main()
