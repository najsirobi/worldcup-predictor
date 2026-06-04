#!/usr/bin/env python3
"""Build data/processed/model_matrix_baseline.parquet from the rated backbone.

Adds tournament/time features + strict no-leakage rolling form. Excludes
unplayed/future fixtures (none present in matches_clean, but guarded anyway).
"""
import logging
from pathlib import Path

import pandas as pd

from src.features.model_matrix import build_model_matrix, FORM_STATS
from src.features.rating_momentum import MOMENTUM_FEATURES

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
RATED = ROOT / "data" / "interim" / "matches_with_ratings.parquet"
CLEAN = ROOT / "data" / "interim" / "matches_clean.parquet"
OUT = ROOT / "data" / "processed" / "model_matrix_baseline.parquet"
REPORT = ROOT / "outputs" / "reports" / "model_matrix_baseline_report.md"


def main():
    src = RATED if RATED.exists() else CLEAN
    df = pd.read_parquet(src)
    df["date"] = pd.to_datetime(df["date"])

    # Guard: only played matches (valid integer scores), no future fixtures.
    df = df[df["home_score"].notna() & df["away_score"].notna()].copy()

    mm = build_model_matrix(df)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    mm.to_parquet(OUT, index=False)
    logger.info(f"✓ Wrote {OUT} ({len(mm)} rows, {len(mm.columns)} cols)")

    # ---- report ----
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    rating_cols = ["home_elo", "away_elo", "elo_diff", "home_fifa_points",
                   "away_fifa_points", "fifa_points_diff"]
    form_cols = [f"home_{s}" for s in FORM_STATS]

    def cov(sub):
        return {
            "rows": len(sub),
            "elo_diff": f"{sub['elo_diff'].notna().mean()*100:.1f}%" if "elo_diff" in sub else "n/a",
            "fifa_points_diff": f"{sub['fifa_points_diff'].notna().mean()*100:.1f}%" if "fifa_points_diff" in sub else "n/a",
            "form_ppm_10": f"{sub['home_ppm_10'].notna().mean()*100:.1f}%",
        }

    with open(REPORT, "w") as f:
        f.write("# Model Matrix (baseline) Report\n\n")
        f.write(f"- Source: `{src.relative_to(ROOT)}`\n")
        f.write(f"- Rows (played matches): **{len(mm)}**\n")
        f.write(f"- Columns: {len(mm.columns)}\n")
        f.write(f"- Date range: {mm['date'].min().date()} → {mm['date'].max().date()}\n\n")

        f.write("## Feature coverage by era\n\n")
        f.write("| Era | Rows | elo_diff | fifa_points_diff | form_ppm_10 |\n|---|--:|--:|--:|--:|\n")
        for label, sub in [("All", mm), ("1992+", mm[mm.match_year >= 1992]), ("2000+", mm[mm.match_year >= 2000])]:
            c = cov(sub)
            f.write(f"| {label} | {c['rows']} | {c['elo_diff']} | {c['fifa_points_diff']} | {c['form_ppm_10']} |\n")
        f.write("\n")

        f.write("## Features\n\n")
        f.write("- Ratings: " + ", ".join(f"`{c}`" for c in rating_cols) + "\n")
        f.write("- Tournament: `tournament_category`, `is_world_cup`, `is_world_cup_qualifier`, "
                "`is_continental_championship`, `is_friendly`\n")
        f.write("- Time: `match_year`, `days_since_2000`\n")
        f.write("- Rolling form (per side + `_diff`): " + ", ".join(f"`{s}`" for s in FORM_STATS) + "\n\n")
        present_momentum = [feature for feature in MOMENTUM_FEATURES if feature in mm.columns]
        if present_momentum:
            f.write("- Rating momentum: " + ", ".join(f"`{s}`" for s in present_momentum) + "\n\n")

        f.write("## No-leakage guarantees\n\n")
        f.write("- Rolling form excludes the current match (`shift(1)`) and only uses prior matches "
                "(`rolling(window)` backward); verified in `tests/test_model_matrix_no_leakage.py`.\n")
        f.write("- Ratings are joined strictly before the match date (Phase 3 as-of join).\n")
        f.write("- Unplayed/future fixtures excluded (valid scores required).\n\n")

        f.write("## Modelling guidance\n\n")
        f.write("- Train rating-based models primarily on **2000+** rows with complete ratings.\n")
        f.write("- 1992+ usable as a secondary experiment (FIFA rankings start 1992-12-31).\n")
        f.write("- ⚠️ No model trained in this step.\n")

    logger.info(f"✓ Wrote {REPORT}")


if __name__ == "__main__":
    main()
