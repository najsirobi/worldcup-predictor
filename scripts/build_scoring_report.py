#!/usr/bin/env python3
"""Validate data/reference/scoring_rules.yml and write a scoring rules report.

The YAML is the structured scoring/objective spec derived from RULES_AND_SCORING.md.
This script loads & validates it and documents it; it does not score anything.
"""
import logging
from pathlib import Path

from src.ingest.rules_and_scoring import (
    load_scoring_rules,
    validate_scoring_rules,
    REQUIRED_NUMERIC_FIELDS,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
REPORT_PATH = REPO_ROOT / "outputs" / "reports" / "scoring_rules_report.md"


def main():
    rules = load_scoring_rules()

    ok, msg = True, "all required fields present and valid"
    try:
        validate_scoring_rules(rules)
    except ValueError as e:
        ok, msg = False, str(e)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write("# Scoring Rules Report\n\n")
        f.write("## Source\n\n")
        f.write("- Spec file: `data/reference/scoring_rules.yml`\n")
        f.write("- Derived from: `Rules of the game/RULES_AND_SCORING.md` "
                "(FIF8A player guide PDF + template XLSX)\n\n")

        f.write("## Validation results\n\n")
        f.write(f"- {'✅ ' + msg if ok else '❌ ' + msg}\n\n")

        f.write("## Point values\n\n")
        f.write("| Field | Value |\n|---|---|\n")
        for field in REQUIRED_NUMERIC_FIELDS:
            f.write(f"| `{field}` | {rules.get(field)} |\n")
        f.write("\n")
        f.write(f"- `odds_are_template_derived`: {rules.get('odds_are_template_derived')}\n")
        f.write(f"- `odds_source`: {rules.get('odds_source')}\n")
        maxima = rules.get("maxima", {})
        if maxima:
            f.write(f"- Derived maxima: {maxima}\n")
        f.write("\n")

        f.write("## Objective\n\n")
        f.write("The model's objective is to **maximise expected points** under these rules, "
                "not merely to predict football outcomes. Scoring functions live in "
                "`src/evaluation/scoring.py` (skeletons for this phase).\n\n")
        for note in rules.get("modelling_notes", []):
            f.write(f"- {note}\n")
        f.write("\n## Status\n\n")
        f.write("- ⚠️ **No model has been trained.** Prediction-selection / optimisation not implemented.\n")
        f.write("- `RULES_AND_SCORING.md` is now the scoring/objective source of truth.\n")

    logger.info(f"  ✓ Wrote {REPORT_PATH} (validation: {'OK' if ok else 'FAILED'})")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
