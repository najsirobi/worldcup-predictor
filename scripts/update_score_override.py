#!/usr/bin/env python3
"""Update a single match in the live score override file (Travel Mode, Task A).

Example:
    python scripts/update_score_override.py \\
        --match-number 1 --team-a-goals 2 --team-b-goals 1 --status played \\
        --notes "Group A opener"

Validation (delegated to src.live.scores_override):
    * the match number must exist
    * goals must be non-negative integers
    * status 'played' requires both goals
    * no duplicate match rows
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.live.scores_override import (
    OVERRIDE_PATH,
    VALID_STATUSES,
    load_override,
    update_match,
    write_override,
)
from src.live.submission_guard import guard_frozen_submission


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--match-number",
        type=int,
        required=True,
        help="Fixture number 1-72 (group) or 73-104 (knockout).",
    )
    parser.add_argument("--team-a-goals", type=int, default=None)
    parser.add_argument("--team-b-goals", type=int, default=None)
    parser.add_argument("--status", choices=VALID_STATUSES, default="played")
    parser.add_argument("--notes", type=str, default=None)
    parser.add_argument("--source", type=str, default="manual")
    parser.add_argument(
        "--advanced-team",
        type=str,
        default=None,
        help="Knockout only: team that advances when the match is level (shoot-out).",
    )
    parser.add_argument("--file", type=Path, default=OVERRIDE_PATH)
    args = parser.parse_args()

    with guard_frozen_submission("update_score_override.py"):
        frame = load_override(args.file)
        try:
            updated = update_match(
                frame,
                match_number=args.match_number,
                team_a_goals=args.team_a_goals,
                team_b_goals=args.team_b_goals,
                status=args.status,
                notes=args.notes,
                source=args.source,
                advanced_team=args.advanced_team,
            )
        except ValueError as exc:
            raise SystemExit(f"ERROR: {exc}")

        write_override(updated, args.file)
        row = updated.loc[updated["match_number"] == args.match_number].iloc[0]
        score = (
            f"{row['team_a_goals']}-{row['team_b_goals']}"
            if row["status"] == "played"
            else "(no result)"
        )
        print(
            f"Updated match {args.match_number}: {row['team_a']} {score} {row['team_b']} "
            f"[{row['status']}] at {row['updated_at']}"
        )


if __name__ == "__main__":
    main()
