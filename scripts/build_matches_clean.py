#!/usr/bin/env python3
"""Build the model-ready historical match backbone (data/interim/matches_clean.parquet).

Loads the raw international results, standardizes the schema, drops unplayed/future
fixtures, adds target/label columns, validates invariants, reports unmapped team
names, and writes the parquet + a markdown report.

No model is trained here. No rolling-form features, ratings joins, or country
context are added. RULES_AND_SCORING.md is the scoring/objective source.
"""
import logging
from pathlib import Path

import pandas as pd

from src.ingest.matches import (
    load_international_results,
    coerce_match_types,
    filter_played_matches,
    add_match_targets,
    validate_clean_matches,
    find_unmapped_teams,
    BACKBONE_COLUMNS,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
INTERIM_PATH = REPO_ROOT / "data" / "interim" / "matches_clean.parquet"
REPORT_PATH = REPO_ROOT / "outputs" / "reports" / "matches_clean_report.md"

TARGET_COLUMNS = [
    "result_label", "home_points", "away_points",
    "goal_diff", "total_goals", "home_goals", "away_goals",
]


def main():
    logger.info("Building historical match backbone...\n")

    raw = load_international_results()
    source_rows = len(raw)

    typed = coerce_match_types(raw)
    played, dropped = filter_played_matches(typed)
    clean = add_match_targets(played)
    validate_clean_matches(clean)

    ordered_cols = BACKBONE_COLUMNS + TARGET_COLUMNS
    clean = clean[ordered_cols].sort_values("date").reset_index(drop=True)

    unmapped = find_unmapped_teams(clean)

    INTERIM_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(INTERIM_PATH, index=False)
    logger.info(f"  ✓ Wrote {INTERIM_PATH} ({len(clean)} rows)")

    # ---- report ----
    dmin, dmax = clean["date"].min(), clean["date"].max()
    dropped_info = ""
    if len(dropped):
        dd = pd.to_datetime(dropped["date"])
        tourns = dropped["tournament"].value_counts().to_dict()
        dropped_info = (
            f"- Dropped unplayed/future fixtures (null score): **{len(dropped)}**\n"
            f"  - date range of dropped: {dd.min().date()} → {dd.max().date()}\n"
            f"  - tournaments: {tourns}\n"
        )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write("# Historical Match Backbone Report\n\n")
        f.write("## Source\n\n")
        f.write("- Source dataset: `data/raw/kaggle/international_results/results.csv` "
                "(martj42 international results), via `src.ingest.matches.load_international_results`\n")
        f.write("- Scoring/objective source of truth: `Rules of the game/RULES_AND_SCORING.md`\n\n")

        f.write("## Row counts\n\n")
        f.write(f"- Raw source rows: **{source_rows}**\n")
        f.write(dropped_info)
        f.write(f"- Clean played-match rows: **{len(clean)}**\n\n")

        f.write("## Date range (played matches)\n\n")
        f.write(f"- {dmin.date()} → {dmax.date()}\n\n")

        f.write("## Schema\n\n")
        f.write("Backbone columns: " + ", ".join(f"`{c}`" for c in BACKBONE_COLUMNS) + "\n\n")
        f.write("Target columns: " + ", ".join(f"`{c}`" for c in TARGET_COLUMNS) + "\n\n")
        f.write("Target definitions:\n")
        f.write("- `result_label`: home_win / draw / away_win\n")
        f.write("- `home_points` / `away_points`: 3 / 1 / 0 (football points)\n")
        f.write("- `goal_diff`: home_score - away_score\n")
        f.write("- `total_goals`: home_score + away_score\n")
        f.write("- `home_goals` / `away_goals`: copies of the scores for convenience\n\n")

        f.write("## Validation results\n\n")
        f.write("All invariants passed (build aborts otherwise):\n")
        f.write("- ✅ every `date` parses to a timestamp\n")
        f.write("- ✅ scores are non-negative integers, no nulls\n")
        f.write("- ✅ `home_team` / `away_team` non-null\n")
        f.write("- ✅ `home_team` != `away_team`\n")
        f.write("- ✅ `tournament` non-null\n\n")
        f.write(f"Result-label distribution: {clean['result_label'].value_counts().to_dict()}\n\n")

        f.write("## Team-name mapping coverage\n\n")
        total_teams = len(set(clean['home_team']) | set(clean['away_team']))
        f.write(f"- Distinct teams in backbone: **{total_teams}**\n")
        f.write(f"- Teams **without** an entry in `data/reference/team_name_map.csv` "
                f"(source `international_results`): **{len(unmapped)}**\n")
        f.write("- Team names are NOT silently merged/normalized. The reference map is "
                "currently a small identity map; the unmapped teams below must be added "
                "to `team_name_map.csv` before name normalization/ratings joins.\n")
        if unmapped:
            preview = ", ".join(unmapped[:40])
            f.write(f"\n<details><summary>Unmapped teams ({len(unmapped)})</summary>\n\n")
            f.write(preview + (" …" if len(unmapped) > 40 else "") + "\n\n</details>\n")
        f.write("\n")

        f.write("## Unresolved assumptions\n\n")
        f.write("- Future WC 2026 fixtures present in the raw file (null scores) were "
                "excluded from the historical backbone; they are not results.\n")
        f.write("- Team-name canonicalization deferred until the reference map is expanded "
                "(reported above, not guessed).\n\n")

        f.write("## Status\n\n")
        f.write("- ⚠️ **No model has been trained.** No rolling-form features, no Elo/FIFA "
                "rating joins, no country-context features, no tournament simulation.\n")
        f.write("- `RULES_AND_SCORING.md` is now the project's scoring/objective source.\n")

    logger.info(f"  ✓ Wrote {REPORT_PATH}")
    logger.info(f"\nRows: {len(clean)} | dropped(unplayed): {len(dropped)} | "
                f"date {dmin.date()}→{dmax.date()} | unmapped teams: {len(unmapped)}")


if __name__ == "__main__":
    main()
