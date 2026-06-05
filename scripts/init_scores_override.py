#!/usr/bin/env python3
"""Initialise the manual live score override file (Travel Mode, Task A).

Creates ``data/live/scores_override.csv`` from the frozen fixture template with
all 72 group-stage matches set to ``scheduled`` and empty goals. Safe to run
repeatedly, but refuses to clobber an existing file with entered results unless
``--force`` is passed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.live.scores_override import (
    EXPECTED_MATCH_COUNT,
    KNOCKOUT_MATCH_NUMBERS,
    OVERRIDE_PATH,
    TEMPLATE_PATH,
    TOTAL_MATCH_COUNT,
    build_initial_override,
    load_override,
    write_override,
)
from src.live.submission_guard import guard_frozen_submission


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=TEMPLATE_PATH)
    parser.add_argument("--out", type=Path, default=OVERRIDE_PATH)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite even if the file already has entered results.",
    )
    args = parser.parse_args()

    with guard_frozen_submission("init_scores_override.py"):
        if args.out.exists() and not args.force:
            existing = load_override(args.out)
            played = (existing["status"] == "played").sum()
            if played:
                raise SystemExit(
                    f"{args.out} already exists with {played} played match(es). "
                    "Refusing to overwrite. Pass --force to reset to a blank slate."
                )

        frame = build_initial_override(args.template)
        group_rows = int((frame["match_number"] <= EXPECTED_MATCH_COUNT).sum())
        knockout_rows = int(frame["match_number"].isin(KNOCKOUT_MATCH_NUMBERS).sum())
        if group_rows != EXPECTED_MATCH_COUNT or len(frame) != TOTAL_MATCH_COUNT:
            raise SystemExit(
                f"Expected {EXPECTED_MATCH_COUNT} group + {len(KNOCKOUT_MATCH_NUMBERS)} "
                f"knockout = {TOTAL_MATCH_COUNT} matches, got {len(frame)} "
                f"({group_rows} group, {knockout_rows} knockout)."
            )
        write_override(frame, args.out)
        print(
            f"Initialised {args.out} with {group_rows} group + {knockout_rows} "
            f"knockout scheduled matches ({len(frame)} total)."
        )


if __name__ == "__main__":
    main()
