#!/usr/bin/env python3
"""Check the immutable frozen-submission manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.live.submission_guard import (
    FROZEN_MANIFEST_PATH,
    FrozenSubmissionIntegrityError,
    verify_manifest,
    write_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help=(
            "Rewrite FROZEN_MANIFEST.json from current files. Use only when "
            "explicitly freezing a submitted candidate."
        ),
    )
    args = parser.parse_args()

    try:
        if args.write_manifest:
            manifest = write_manifest(FROZEN_MANIFEST_PATH, root=ROOT)
            print(
                "Wrote frozen submission manifest: "
                f"{FROZEN_MANIFEST_PATH.relative_to(ROOT)} "
                f"({len(manifest['files'])} files)."
            )
            return

        result = verify_manifest(FROZEN_MANIFEST_PATH, root=ROOT)
    except FrozenSubmissionIntegrityError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    print(
        "Frozen submission integrity OK: "
        f"{len(result.checked_files)} file hashes match "
        f"{result.manifest_path.relative_to(ROOT)}."
    )


if __name__ == "__main__":
    main()
