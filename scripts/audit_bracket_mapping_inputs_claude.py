"""Audit bracket mapping inputs for WC 2026 Phase 7.

Scans local repo files for bracket-related terms and reports which candidate
sources exist. Does NOT import or modify any production simulation module.

Usage:
    python scripts/audit_bracket_mapping_inputs_claude.py [--repo-root PATH]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


BRACKET_KEYWORDS = [
    "round of 32",
    "R32",
    "round of 16",
    "R16",
    "knockout bracket",
    "bracket mapping",
    "best third",
    "best-third",
    "third.place",
    "BT1",
    "slot",
    "side_of_bracket",
    "opponent_slot",
    "winner_to",
    "loser_to",
]

TEXT_EXTENSIONS = {
    ".md", ".txt", ".csv", ".yml", ".yaml", ".py", ".json", ".html", ".rst",
}

SKIP_DIRS = {".venv", "__pycache__", ".git", "node_modules", ".egg-info"}


def scan_file(path: Path, keywords: list[str]) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace").lower()
    except OSError:
        return []
    hits = []
    for kw in keywords:
        if kw.lower() in text:
            hits.append(kw)
    return hits


def scan_repo(root: Path) -> dict[str, list[str]]:
    results: dict[str, list[str]] = {}
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        if any(skip in path.parts for skip in SKIP_DIRS):
            continue
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        hits = scan_file(path, BRACKET_KEYWORDS)
        if hits:
            results[str(path.relative_to(root))] = hits
    return results


def check_bracket_mapping_file(root: Path) -> dict[str, bool]:
    candidates = [
        "data/reference/knockout_bracket_mapping_manual.csv",
        "data/reference/knockout_bracket_mapping_manual_claude.csv",
    ]
    return {c: (root / c).exists() for c in candidates}


def report(root: Path) -> None:
    print("=" * 70)
    print("WC 2026 Bracket Mapping Input Audit")
    print(f"Repo root: {root}")
    print("=" * 70)

    mapping_files = check_bracket_mapping_file(root)
    print("\n--- Bracket mapping file status ---")
    for path, exists in mapping_files.items():
        status = "FOUND" if exists else "MISSING"
        print(f"  [{status}]  {path}")

    print("\n--- Files containing bracket-related keywords ---")
    hits = scan_repo(root)
    if not hits:
        print("  (none found)")
    else:
        for rel_path, keywords in sorted(hits.items()):
            print(f"\n  {rel_path}")
            for kw in sorted(set(keywords)):
                print(f"    · {kw}")

    print("\n--- Summary ---")
    mapping_present = any(mapping_files.values())
    print(
        f"  Bracket mapping file present:  {'YES' if mapping_present else 'NO — Phase 7 path-aware simulation will fail'}"
    )
    print(
        f"  Files with bracket keywords:   {len(hits)}"
    )
    if not mapping_present:
        print(
            "\n  ACTION REQUIRED: populate data/reference/knockout_bracket_mapping_manual.csv"
            "\n  from the official FIFA WC 2026 bracket draw before running Phase 7."
        )
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Path to repo root (default: parent of this script's parent dir)",
    )
    args = parser.parse_args(argv)

    if args.repo_root:
        root = Path(args.repo_root).resolve()
    else:
        root = Path(__file__).resolve().parent.parent

    if not root.exists():
        print(f"ERROR: repo root not found: {root}", file=sys.stderr)
        return 1

    report(root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
