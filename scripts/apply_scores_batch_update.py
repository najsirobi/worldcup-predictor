#!/usr/bin/env python3
"""Apply a batch of match scores from a CSV file (Travel Mode, Task C).

Reads ``data/live/scores_batch_update.csv`` (columns:
``match_number,team_a_goals,team_b_goals,status,notes``), validates every row,
and applies the whole batch atomically to ``data/live/scores_override.csv``.

Rules:
    * all rows are validated before anything is written;
    * if one row is invalid the whole batch is rejected (no partial apply);
    * matches not listed in the batch keep their existing values;
    * changed rows are stamped ``source=batch_update`` and a fresh ``updated_at``.

Writes a report to ``outputs/reports/scores_batch_update_report.md``.
Does NOT retrain, fetch APIs, or change predictions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.live.batch_update import BATCH_FIELDS, BatchValidationError, apply_batch
from src.live.scores_override import OVERRIDE_PATH, load_override, utc_now_iso, write_override

ROOT = Path(__file__).resolve().parents[1]
BATCH_PATH = ROOT / "data" / "live" / "scores_batch_update.csv"
REPORT_PATH = ROOT / "outputs" / "reports" / "scores_batch_update_report.md"
SOURCE = "batch_update"


def read_batch_rows(path: Path) -> list[dict]:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in BATCH_FIELDS if c not in frame.columns]
    if missing:
        raise BatchValidationError([f"batch CSV missing column(s): {missing}"])
    return frame.to_dict(orient="records")


def write_report(applied: list[dict], errors: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Batch Score Update Report", "", f"_Generated: {utc_now_iso()}_", ""]
    if errors:
        lines += [
            f"## ❌ Batch rejected ({len(errors)} error(s)) — nothing was applied",
            "",
            *[f"- {e}" for e in errors],
            "",
        ]
    else:
        lines += [
            f"## ✅ Batch applied ({len(applied)} match(es), source `{SOURCE}`)",
            "",
            "| # | Match | Status | Score | Notes |",
            "|---|-------|--------|-------|-------|",
        ]
        for a in applied:
            lines.append(
                f"| {a['match_number']} | {a['team_a']} v {a['team_b']} | "
                f"{a['status']} | {a['score']} | {a['notes']} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, default=BATCH_PATH)
    parser.add_argument("--file", type=Path, default=OVERRIDE_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()

    frame = load_override(args.file)
    try:
        rows = read_batch_rows(args.batch)
        updated, applied = apply_batch(frame, rows, source=SOURCE)
    except BatchValidationError as exc:
        write_report([], exc.errors, args.report)
        raise SystemExit(
            "ERROR: batch rejected, nothing applied:\n  - " + "\n  - ".join(exc.errors)
        )

    write_override(updated, args.file)
    write_report(applied, [], args.report)
    print(f"Applied {len(applied)} match(es) from {args.batch} (source {SOURCE}).")


if __name__ == "__main__":
    main()
