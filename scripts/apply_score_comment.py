#!/usr/bin/env python3
"""Apply scores parsed from a GitHub Issue comment (Travel Mode, Task D).

Reads a comment body (from ``--comment-file`` or stdin), extracts the
``/WK-SCORES ... /END-WK-SCORES`` block, validates every row, and applies the
whole batch atomically to ``data/live/scores_override.csv`` with
``source=github_issue_comment``. On any error nothing is written.

Writes a report to ``outputs/reports/score_comment_ingestion_report.md`` and
exits non-zero on failure so the calling workflow can report back.
Does NOT retrain, fetch APIs, or change predictions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.live.batch_update import BatchValidationError, apply_batch
from src.live.score_comment_parser import CommentParseError, parse_comment
from src.live.scores_override import OVERRIDE_PATH, load_override, utc_now_iso, write_override

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "outputs" / "reports" / "score_comment_ingestion_report.md"
SOURCE = "github_issue_comment"


def write_report(applied: list[dict], errors: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# GitHub Issue Comment Score Ingestion Report", "", f"_Generated: {utc_now_iso()}_", ""]
    if errors:
        lines += [
            f"## ❌ Update rejected ({len(errors)} error(s)) — nothing was applied",
            "",
            *[f"- {e}" for e in errors],
            "",
            "Fix the comment and post it again between the "
            "`/WK-SCORES` and `/END-WK-SCORES` markers.",
            "",
        ]
    else:
        lines += [
            f"## ✅ Applied {len(applied)} match(es) (source `{SOURCE}`)",
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


def _fail(errors: list[str], report: Path) -> None:
    write_report([], errors, report)
    sys.stderr.write("ERROR: comment update rejected, nothing applied:\n")
    for e in errors:
        sys.stderr.write(f"  - {e}\n")
    raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comment-file", type=Path, default=None, help="File with the comment body (else stdin)."
    )
    parser.add_argument("--file", type=Path, default=OVERRIDE_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()

    body = args.comment_file.read_text(encoding="utf-8") if args.comment_file else sys.stdin.read()

    try:
        rows = parse_comment(body)
    except CommentParseError as exc:
        _fail([str(exc)], args.report)

    frame = load_override(args.file)
    try:
        updated, applied = apply_batch(frame, rows, source=SOURCE)
    except BatchValidationError as exc:
        _fail(exc.errors, args.report)

    write_override(updated, args.file)
    write_report(applied, [], args.report)
    print(f"Applied {len(applied)} match(es) from issue comment (source {SOURCE}).")


if __name__ == "__main__":
    main()
