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
    OVERRIDE_PATH,
    TEMPLATE_PATH,
    build_initial_override,
    load_override,
    write_override,
)


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

    if args.out.exists() and not args.force:
        existing = load_override(args.out)
        played = (existing["status"] == "played").sum()
        if played:
            raise SystemExit(
                f"{args.out} already exists with {played} played match(es). "
                "Refusing to overwrite. Pass --force to reset to a blank slate."
            )

    frame = build_initial_override(args.template)
    if len(frame) != EXPECTED_MATCH_COUNT:
        raise SystemExit(
            f"Expected {EXPECTED_MATCH_COUNT} matches from template, got {len(frame)}."
        )
    write_override(frame, args.out)
    print(f"Initialised {args.out} with {len(frame)} scheduled matches.")


if __name__ == "__main__":
    main()
