#!/usr/bin/env python3
"""Generate data/reference/fif8a_group_stage_template.csv from RULES_AND_SCORING.md.

Parses the 72 group fixtures + template odds from the canonical spec, validates
the structure (12 groups, 6 matches each, positive odds), writes the CSV, and
writes a markdown report. Does not invent missing fixtures or odds.
"""
import logging
from pathlib import Path

import pandas as pd

from src.ingest.fif8a_template import (
    parse_fif8a_md,
    validate_fif8a_group_template,
    GROUPS,
    EXPECTED_TOTAL_MATCHES,
    EXPECTED_MATCHES_PER_GROUP,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
CSV_PATH = REPO_ROOT / "data" / "reference" / "fif8a_group_stage_template.csv"
REPORT_PATH = REPO_ROOT / "outputs" / "reports" / "fif8a_template_report.md"


def main():
    logger.info("Building FIF8A group-stage template from RULES_AND_SCORING.md...\n")

    df = parse_fif8a_md()

    # Validate; capture result for the report rather than crashing silently.
    validation_ok = True
    validation_msg = "all checks passed"
    try:
        validate_fif8a_group_template(df, require_full=True)
    except ValueError as e:
        validation_ok = False
        validation_msg = str(e)

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    logger.info(f"  ✓ Wrote {CSV_PATH} ({len(df)} rows)")

    counts = df.groupby("group").size().to_dict()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write("# FIF8A Group-Stage Template Report\n\n")
        f.write("## Source\n\n")
        f.write("- Canonical source: `Rules of the game/RULES_AND_SCORING.md` (§6 fixtures & odds)\n")
        f.write("- Underlying: FIF8A player template XLSX (FIFA rankings dated 2026-05-21)\n")
        f.write("- Output: `data/reference/fif8a_group_stage_template.csv`\n\n")

        f.write("## Row counts\n\n")
        f.write(f"- Matches parsed: **{len(df)}** (expected {EXPECTED_TOTAL_MATCHES})\n")
        f.write(f"- Groups: **{len(set(df['group']))}** (expected {len(GROUPS)})\n")
        f.write(f"- Matches per group: {counts} (expected {EXPECTED_MATCHES_PER_GROUP} each)\n\n")

        if len(df):
            dd = pd.to_datetime(df["date"])
            f.write("## Date range\n\n")
            f.write(f"- {dd.min().date()} → {dd.max().date()}\n\n")

        f.write("## Validation results\n\n")
        if validation_ok:
            f.write("- ✅ exactly 12 groups A–L\n")
            f.write("- ✅ exactly 6 matches per group (72 total)\n")
            f.write("- ✅ each match has team_a, team_b, rate_a, rate_draw, rate_b\n")
            f.write("- ✅ all odds are positive numbers\n")
            f.write("- ✅ match_number values are unique\n\n")
        else:
            f.write(f"- ❌ validation FAILED:\n\n```\n{validation_msg}\n```\n\n")

        f.write("## Odds semantics\n\n")
        f.write("- `rate_a` / `rate_draw` / `rate_b` = template odds for Team A win / draw / "
                "Team B win.\n")
        f.write("- Odds are **template-derived** (FIFA-ranking-point Poisson, odd = 1/probability), "
                "**not bookmaker odds**.\n\n")

        f.write("## Unresolved assumptions\n\n")
        f.write("- Odds are rounded to 2 decimals as published in RULES_AND_SCORING.md §6; "
                "full-precision values exist in the source XLSX if needed later.\n")
        f.write("- Group assignments reflect the template's seeded/assumed draw "
                "(per RULES_AND_SCORING.md §7); re-verify against the official final draw.\n")
        f.write("- Team A / Team B is only a fixture designation (no home advantage implied).\n\n")

        f.write("## Status\n\n")
        f.write("- ⚠️ **No model has been trained.** No tournament simulation performed.\n")
        f.write("- `RULES_AND_SCORING.md` is the scoring/objective source of truth.\n")

    logger.info(f"  ✓ Wrote {REPORT_PATH}")
    logger.info(f"\nMatches: {len(df)} | groups: {sorted(set(df['group']))} | "
                f"validation: {'OK' if validation_ok else 'FAILED'}")
    if not validation_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
