"""QA script for Annexe C best-third assignment table and R32 bracket mapping.

Reads production files if present and writes a QA report to
outputs/reports/annex_c_qa_report_claude.md.

Does NOT modify any production file.

Usage:
    python scripts/qa_annex_c_mapping_claude.py [--repo-root PATH]
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent

PRODUCTION_FILES = {
    "annex_c": "data/reference/third_place_assignment_annex_c.csv",
    "r32_mapping": "data/reference/round_of_32_mapping.csv",
    "bracket_mapping": "data/reference/knockout_bracket_mapping.csv",
    "progression": "data/reference/knockout_round_progression.csv",
}

SLOT_COLUMNS = ["slot_1A", "slot_1B", "slot_1D", "slot_1E", "slot_1G", "slot_1I", "slot_1K", "slot_1L"]
VALID_GROUPS = set("ABCDEFGHIJKL")
# R32 is defined in round_of_32_mapping.csv; the progression file covers R16 onward.
REQUIRED_ROUNDS = {"R16", "QF", "SF", "Final"}


def _parse_groups(value: Any) -> list[str]:
    """qualified_third_groups is a comma-separated list, e.g. 'E,F,G,H,I,J,K,L'."""
    return [tok.strip().upper() for tok in str(value).split(",") if tok.strip()]


def _parse_slot(value: Any) -> str:
    """Slot cells are position-prefixed, e.g. '3E' -> group letter 'E'."""
    return str(value).strip().upper().lstrip("3")


def _import_pandas() -> Any:
    try:
        import pandas as pd
        return pd
    except ImportError:
        print("ERROR: pandas not available.", file=sys.stderr)
        sys.exit(1)


def check_file_exists(root: Path, key: str, rel_path: str) -> tuple[bool, str]:
    path = root / rel_path
    if path.exists():
        return True, f"FOUND   {rel_path}"
    return False, f"MISSING {rel_path}"


def qa_annex_c(df: Any, pd: Any) -> list[str]:
    """Run all Annexe C validation checks. Returns list of result lines."""
    lines = []

    def ok(msg: str) -> None:
        lines.append(f"  [PASS] {msg}")

    def fail(msg: str) -> None:
        lines.append(f"  [FAIL] {msg}")

    def warn(msg: str) -> None:
        lines.append(f"  [WARN] {msg}")

    # R1 — row count
    expected_rows = 495
    if len(df) == expected_rows:
        ok(f"R1 row count: {len(df)} == {expected_rows}")
    else:
        fail(f"R1 row count: {len(df)} != {expected_rows} (expected {expected_rows})")

    # R2 — required columns
    required_cols = ["option_number", "qualified_third_groups"] + SLOT_COLUMNS
    missing_cols = [c for c in required_cols if c not in df.columns]
    if not missing_cols:
        ok("R2 all required columns present")
    else:
        fail(f"R2 missing columns: {missing_cols}")
        lines.append("  Cannot proceed with row-level checks.")
        return lines

    # R3 — unique keys
    dup_option = df["option_number"].duplicated().sum()
    dup_groups = df["qualified_third_groups"].duplicated().sum()
    if dup_option == 0:
        ok("R3a option_number is unique")
    else:
        fail(f"R3a {dup_option} duplicate option_number values")
    if dup_groups == 0:
        ok("R3b qualified_third_groups is unique")
    else:
        fail(f"R3b {dup_groups} duplicate qualified_third_groups values")
        dupes = df[df["qualified_third_groups"].duplicated(keep=False)]["qualified_third_groups"].unique()
        lines.append(f"    Examples: {list(dupes[:5])}")

    # R4 — valid group letters
    bad = []
    for _, row in df.iterrows():
        vals = _parse_groups(row["qualified_third_groups"])
        vals += [_parse_slot(row[col]) for col in SLOT_COLUMNS]
        if any(ch not in VALID_GROUPS for v in vals for ch in v):
            bad.append(row["option_number"])
    if not bad:
        ok("R4 all group letters within A-L")
    else:
        fail(f"R4 invalid group letters found in {len(bad)} rows")
        lines.append(f"    Examples: {bad[:5]}")

    # R5 — qualified_third_groups must list exactly 8 distinct valid groups
    bad_r5 = []
    for val in df["qualified_third_groups"]:
        groups = _parse_groups(val)
        if len(groups) != 8 or len(set(groups)) != 8 or not all(ch in VALID_GROUPS for ch in groups):
            bad_r5.append(val)
    if not bad_r5:
        ok("R5 all qualified_third_groups list 8 distinct valid groups")
    else:
        fail(f"R5 {len(bad_r5)} malformed qualified_third_groups values")
        lines.append(f"    Examples: {bad_r5[:5]}")

    # R6 — slot assignments cover exactly the 8 qualified groups per row
    r6_failures = []
    for _, row in df.iterrows():
        qualified = set(_parse_groups(row["qualified_third_groups"]))
        assigned = {_parse_slot(row[col]) for col in SLOT_COLUMNS}
        if assigned != qualified:
            r6_failures.append(
                f"row {row['option_number']}: qualified={sorted(qualified)} assigned={sorted(assigned)}"
            )
    if not r6_failures:
        ok("R6 all rows: slot assignments exactly cover qualified groups")
    else:
        fail(f"R6 {len(r6_failures)} rows have assignment/qualification mismatch")
        lines.append(f"    Examples: {r6_failures[:3]}")

    # R7 — no duplicate slot assignments within a row
    r7_failures = []
    for _, row in df.iterrows():
        slot_vals = [_parse_slot(row[col]) for col in SLOT_COLUMNS]
        if len(slot_vals) != len(set(slot_vals)):
            dupes = [v for v in slot_vals if slot_vals.count(v) > 1]
            r7_failures.append(f"row {row['option_number']}: duplicates={dupes}")
    if not r7_failures:
        ok("R7 no duplicate slot assignments within any row")
    else:
        fail(f"R7 {len(r7_failures)} rows have duplicate slot assignments")
        lines.append(f"    Examples: {r7_failures[:3]}")

    # R8 — all 495 combinations present
    expected_combos = {",".join(c) for c in combinations("ABCDEFGHIJKL", 8)}
    found_combos = {",".join(_parse_groups(v)) for v in df["qualified_third_groups"]}
    missing_combos = expected_combos - found_combos
    extra_combos = found_combos - expected_combos
    if not missing_combos and not extra_combos:
        ok(f"R8 all {len(expected_combos)} expected combinations present")
    else:
        if missing_combos:
            fail(f"R8 {len(missing_combos)} combinations missing from table")
            lines.append(f"    Examples missing: {sorted(missing_combos)[:5]}")
        if extra_combos:
            fail(f"R8 {len(extra_combos)} unexpected combinations in table")
            lines.append(f"    Examples extra: {sorted(extra_combos)[:5]}")

    # Examples of valid rows
    valid_rows = []
    for _, row in df.iterrows():
        qualified = set(_parse_groups(row["qualified_third_groups"]))
        assigned = {_parse_slot(row[col]) for col in SLOT_COLUMNS}
        slot_vals = [_parse_slot(row[col]) for col in SLOT_COLUMNS]
        if (
            assigned == qualified
            and len(slot_vals) == len(set(slot_vals))
            and len(qualified) == 8
            and all(ch in VALID_GROUPS for ch in qualified)
        ):
            valid_rows.append(row)
        if len(valid_rows) == 5:
            break
    if valid_rows:
        lines.append(f"\n  --- 5 valid row examples ---")
        for row in valid_rows:
            slot_str = ", ".join(f"{col}={row[col]}" for col in SLOT_COLUMNS)
            lines.append(f"  option={row['option_number']} groups={row['qualified_third_groups']} | {slot_str}")

    return lines


def qa_r32_mapping(df: Any, pd: Any) -> list[str]:
    lines = []

    def ok(msg: str) -> None:
        lines.append(f"  [PASS] {msg}")

    def fail(msg: str) -> None:
        lines.append(f"  [FAIL] {msg}")

    required = {"match_number", "team_a_source", "team_b_source", "team_a_source_type", "team_b_source_type"}
    missing = required - set(df.columns)
    if missing:
        fail(f"Missing columns: {sorted(missing)}")
        return lines
    ok("Required columns present")

    # R9 — exactly 16 R32 matches
    if len(df) == 16:
        ok("R9 R32 has exactly 16 matches")
    else:
        fail(f"R9 R32 match count: {len(df)} != 16")

    n_matches = df["match_number"].nunique()
    if n_matches == 16:
        ok("R9 16 distinct R32 match_numbers")
    else:
        fail(f"R9 {n_matches} distinct R32 match_numbers (expected 16)")

    # R11 — exactly 8 best-third slots feed the R32
    best_third = (
        df["team_a_source_type"].eq("best_third").sum()
        + df["team_b_source_type"].eq("best_third").sum()
    )
    if best_third == 8:
        ok("R11 exactly 8 best-third slots in R32")
    else:
        fail(f"R11 {best_third} best-third slots (expected 8)")

    # No empty team sources
    empty_src = (
        df["team_a_source"].isna().sum() + df["team_b_source"].isna().sum()
    )
    if empty_src == 0:
        ok("R11 all R32 slots have a team source")
    else:
        fail(f"R11 {empty_src} empty team sources")

    return lines


def qa_progression(df: Any, pd: Any) -> list[str]:
    lines = []

    def ok(msg: str) -> None:
        lines.append(f"  [PASS] {msg}")

    def fail(msg: str) -> None:
        lines.append(f"  [FAIL] {msg}")

    if "round" not in df.columns:
        fail("Missing 'round' column")
        return lines
    found_rounds = set(df["round"].astype(str).str.strip())
    missing_rounds = REQUIRED_ROUNDS - found_rounds
    if not missing_rounds:
        ok(f"R10 all required rounds present: {sorted(REQUIRED_ROUNDS)}")
    else:
        fail(f"R10 missing rounds: {sorted(missing_rounds)}")

    # R10b — progression must reach a winner via the Final
    reaches_winner = False
    if "winner_to_match" in df.columns:
        final_rows = df[df["round"].astype(str).str.strip().eq("Final")]
        reaches_winner = final_rows["winner_to_match"].astype(str).str.strip().str.lower().eq("winner").any()
    if reaches_winner:
        ok("R10b Final resolves to a tournament Winner")
    else:
        fail("R10b progression does not resolve to a Winner")

    return lines


def run_qa(root: Path, output_path: Path) -> None:
    pd = _import_pandas()

    report_lines = [
        "# Annexe C QA Report (Claude)",
        "",
        f"Generated: 2026-06-04",
        f"Repo root: {root}",
        "",
        "---",
        "",
        "## 1. File Existence",
        "",
    ]

    file_status: dict[str, bool] = {}
    for key, rel_path in PRODUCTION_FILES.items():
        exists, msg = check_file_exists(root, key, rel_path)
        file_status[key] = exists
        report_lines.append(f"- {msg}")
    report_lines.append("")

    any_found = any(file_status.values())
    if not any_found:
        report_lines += [
            "**No production mapping files are available yet.**",
            "",
            "The Codex Phase 7B agent has not yet written the production mapping files.",
            "Re-run this script after Codex finishes to validate its output.",
            "",
            "Expected files:",
        ]
        for rel_path in PRODUCTION_FILES.values():
            report_lines.append(f"- `{rel_path}`")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(report_lines))
        print(f"QA report written (no production files yet): {output_path}")
        return

    # Annexe C checks
    report_lines += ["---", "", "## 2. Annexe C Table QA", ""]
    if file_status["annex_c"]:
        try:
            df_annex = pd.read_csv(root / PRODUCTION_FILES["annex_c"])
            report_lines.append(f"- Rows loaded: {len(df_annex)}")
            report_lines.append(f"- Columns: {list(df_annex.columns)}")
            report_lines.append("")
            results = qa_annex_c(df_annex, pd)
            report_lines.extend(results)
        except Exception as exc:
            report_lines.append(f"  [ERROR] Failed to load: {exc}")
    else:
        report_lines.append("- **Not available yet.**")
    report_lines.append("")

    # R32 mapping checks
    report_lines += ["---", "", "## 3. R32 Mapping QA", ""]
    r32_path = PRODUCTION_FILES["r32_mapping"]
    if not file_status["r32_mapping"] and file_status["bracket_mapping"]:
        r32_path = PRODUCTION_FILES["bracket_mapping"]
        report_lines.append(f"- Using fallback: `{r32_path}`")
    if file_status["r32_mapping"] or file_status["bracket_mapping"]:
        try:
            df_r32 = pd.read_csv(root / r32_path)
            report_lines.append(f"- Rows loaded: {len(df_r32)}")
            results = qa_r32_mapping(df_r32, pd)
            report_lines.extend(results)
        except Exception as exc:
            report_lines.append(f"  [ERROR] Failed to load: {exc}")
    else:
        report_lines.append("- **Not available yet.**")
    report_lines.append("")

    # Progression checks
    report_lines += ["---", "", "## 4. Round Progression QA", ""]
    if file_status["progression"]:
        try:
            df_prog = pd.read_csv(root / PRODUCTION_FILES["progression"])
            results = qa_progression(df_prog, pd)
            report_lines.extend(results)
        except Exception as exc:
            report_lines.append(f"  [ERROR] Failed to load: {exc}")
    else:
        report_lines.append("- **Not available yet.**")
    report_lines.append("")

    # Summary
    report_lines += [
        "---",
        "",
        "## 5. Summary",
        "",
        "| File | Status |",
        "|---|---|",
    ]
    for key, rel_path in PRODUCTION_FILES.items():
        status = "Found" if file_status[key] else "Missing"
        report_lines.append(f"| `{rel_path}` | {status} |")
    report_lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(report_lines))
    print(f"QA report written: {output_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=None)
    args = parser.parse_args(argv)
    root = Path(args.repo_root).resolve() if args.repo_root else REPO_ROOT
    output_path = root / "outputs/reports/annex_c_qa_report_claude.md"
    run_qa(root, output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
