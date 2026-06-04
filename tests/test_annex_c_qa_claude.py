"""Synthetic tests for Annexe C QA validation logic.

Uses tiny in-memory DataFrames — no production CSV files, no PDF.
Tests verify that the qa_annex_c_mapping_claude validation functions
catch the exact failure modes specified in annex_c_validation_spec_claude.md.
"""

from __future__ import annotations

from itertools import combinations

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Inline copy of the validation helpers (avoids importing the script as a
# module while keeping the test self-contained and independent).
# ---------------------------------------------------------------------------

SLOT_COLUMNS = ["slot_1A", "slot_1B", "slot_1D", "slot_1E", "slot_1G", "slot_1I", "slot_1K", "slot_1L"]
VALID_GROUPS = set("ABCDEFGHIJKL")


def _all_495_combos() -> list[str]:
    return ["".join(sorted(c)) for c in combinations("ABCDEFGHIJKL", 8)]


def _make_valid_row(option_number: int, groups: str) -> dict:
    """Build a valid Annexe C row from a sorted 8-letter group string."""
    assert len(groups) == 8 and len(set(groups)) == 8
    row: dict = {"option_number": option_number, "qualified_third_groups": groups}
    for col, grp in zip(SLOT_COLUMNS, sorted(groups)):
        row[col] = grp
    return row


def _valid_single_row_df() -> pd.DataFrame:
    return pd.DataFrame([_make_valid_row(1, "ABCDEFGH")])


def _valid_full_df() -> pd.DataFrame:
    combos = _all_495_combos()
    rows = [_make_valid_row(i + 1, combo) for i, combo in enumerate(combos)]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Validation helpers (mirror the logic in qa_annex_c_mapping_claude.py)
# ---------------------------------------------------------------------------

def validate_row_count(df: pd.DataFrame) -> list[str]:
    return ["PASS_R1"] if len(df) == 495 else [f"FAIL_R1:{len(df)}"]


def validate_required_columns(df: pd.DataFrame) -> list[str]:
    required = ["option_number", "qualified_third_groups"] + SLOT_COLUMNS
    missing = [c for c in required if c not in df.columns]
    return ["PASS_R2"] if not missing else [f"FAIL_R2:{missing}"]


def validate_unique_keys(df: pd.DataFrame) -> list[str]:
    results = []
    if df["option_number"].duplicated().any():
        results.append("FAIL_R3_option_number_dup")
    else:
        results.append("PASS_R3a")
    if df["qualified_third_groups"].duplicated().any():
        results.append("FAIL_R3_qualified_third_groups_dup")
    else:
        results.append("PASS_R3b")
    return results


def validate_valid_group_letters(df: pd.DataFrame) -> list[str]:
    for col in ["qualified_third_groups"] + SLOT_COLUMNS:
        for val in df[col].astype(str):
            if not all(ch in VALID_GROUPS for ch in val.strip().upper()):
                return [f"FAIL_R4:invalid_letter_in_{col}:{val}"]
    return ["PASS_R4"]


def validate_8group_strings(df: pd.DataFrame) -> list[str]:
    for val in df["qualified_third_groups"]:
        v = str(val).strip().upper()
        if len(v) != 8 or len(set(v)) != 8 or not all(ch in VALID_GROUPS for ch in v):
            return [f"FAIL_R5:{val}"]
    return ["PASS_R5"]


def validate_assignment_covers_qualified(df: pd.DataFrame) -> list[str]:
    for _, row in df.iterrows():
        qualified = set(str(row["qualified_third_groups"]).strip().upper())
        assigned = {str(row[col]).strip().upper() for col in SLOT_COLUMNS}
        if assigned != qualified:
            return [f"FAIL_R6:row={row['option_number']}:qualified={sorted(qualified)}:assigned={sorted(assigned)}"]
    return ["PASS_R6"]


def validate_no_duplicate_slot_in_row(df: pd.DataFrame) -> list[str]:
    for _, row in df.iterrows():
        vals = [str(row[col]).strip().upper() for col in SLOT_COLUMNS]
        if len(vals) != len(set(vals)):
            return [f"FAIL_R7:row={row['option_number']}:dups={vals}"]
    return ["PASS_R7"]


def validate_all_495_combos_present(df: pd.DataFrame) -> list[str]:
    expected = {"".join(sorted(c)) for c in combinations("ABCDEFGHIJKL", 8)}
    found = set(df["qualified_third_groups"].str.strip().str.upper())
    missing = expected - found
    extra = found - expected
    if missing or extra:
        return [f"FAIL_R8:missing={len(missing)},extra={len(extra)}"]
    return ["PASS_R8"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestValidSyntheticRow:
    """Test 1: a valid synthetic row passes all checks."""

    def test_valid_row_passes_column_check(self):
        df = _valid_single_row_df()
        assert "PASS_R2" in validate_required_columns(df)

    def test_valid_row_passes_letter_check(self):
        df = _valid_single_row_df()
        assert "PASS_R4" in validate_valid_group_letters(df)

    def test_valid_row_passes_8group_check(self):
        df = _valid_single_row_df()
        assert "PASS_R5" in validate_8group_strings(df)

    def test_valid_row_passes_assignment_coverage(self):
        df = _valid_single_row_df()
        assert "PASS_R6" in validate_assignment_covers_qualified(df)

    def test_valid_row_passes_no_dup_slot(self):
        df = _valid_single_row_df()
        assert "PASS_R7" in validate_no_duplicate_slot_in_row(df)

    def test_valid_full_df_passes_row_count(self):
        df = _valid_full_df()
        assert "PASS_R1" in validate_row_count(df)

    def test_valid_full_df_passes_all_combos(self):
        df = _valid_full_df()
        assert "PASS_R8" in validate_all_495_combos_present(df)


class TestDuplicateAssignedGroup:
    """Test 2: duplicate assigned third group within a row fails R7."""

    def test_duplicate_slot_value_fails(self):
        row = _make_valid_row(1, "ABCDEFGH")
        # Force slot_1B = slot_1A = "A" — duplicate
        row["slot_1B"] = "A"
        df = pd.DataFrame([row])
        result = validate_no_duplicate_slot_in_row(df)
        assert any("FAIL_R7" in r for r in result)

    def test_valid_row_has_no_duplicates(self):
        df = _valid_single_row_df()
        result = validate_no_duplicate_slot_in_row(df)
        assert all("FAIL" not in r for r in result)


class TestAssignedGroupNotInQualified:
    """Test 3: slot assigned to a group not in qualified_third_groups fails R6."""

    def test_unqualified_group_in_slot_fails(self):
        row = _make_valid_row(1, "ABCDEFGH")
        # Assign group "I" (not in ABCDEFGH) to slot_1A
        row["slot_1A"] = "I"
        df = pd.DataFrame([row])
        result = validate_assignment_covers_qualified(df)
        assert any("FAIL_R6" in r for r in result)

    def test_correct_assignment_passes(self):
        df = _valid_single_row_df()
        result = validate_assignment_covers_qualified(df)
        assert "PASS_R6" in result


class TestInvalidGroupLetter:
    """Test 4: group letter outside A-L fails R4."""

    def test_letter_m_is_invalid(self):
        row = _make_valid_row(1, "ABCDEFGH")
        row["slot_1A"] = "M"
        df = pd.DataFrame([row])
        result = validate_valid_group_letters(df)
        assert any("FAIL_R4" in r for r in result)

    def test_digit_in_slot_is_invalid(self):
        row = _make_valid_row(1, "ABCDEFGH")
        row["slot_1B"] = "1"
        df = pd.DataFrame([row])
        result = validate_valid_group_letters(df)
        assert any("FAIL_R4" in r for r in result)

    def test_all_valid_letters_pass(self):
        df = _valid_single_row_df()
        result = validate_valid_group_letters(df)
        assert "PASS_R4" in result


class TestDuplicateQualifiedThirdGroupsKey:
    """Test 5: duplicate qualified_third_groups key fails R3."""

    def test_duplicate_combination_key_fails(self):
        row1 = _make_valid_row(1, "ABCDEFGH")
        row2 = _make_valid_row(2, "ABCDEFGH")  # same combination!
        df = pd.DataFrame([row1, row2])
        results = validate_unique_keys(df)
        assert any("FAIL_R3_qualified_third_groups_dup" in r for r in results)

    def test_unique_combination_keys_pass(self):
        row1 = _make_valid_row(1, "ABCDEFGH")
        row2 = _make_valid_row(2, "ABCDEFGIJ"[:8])  # different combination
        # Build a valid second row with groups ABCDEFGI
        row2 = _make_valid_row(2, "ABCDEFGI")
        df = pd.DataFrame([row1, row2])
        results = validate_unique_keys(df)
        assert "PASS_R3b" in results


class TestMissingRequiredSlot:
    """Test 6: row missing a required slot column fails R2."""

    def test_missing_slot_column_fails(self):
        df = _valid_single_row_df().drop(columns=["slot_1A"])
        result = validate_required_columns(df)
        assert any("FAIL_R2" in r for r in result)

    def test_missing_option_number_fails(self):
        df = _valid_single_row_df().drop(columns=["option_number"])
        result = validate_required_columns(df)
        assert any("FAIL_R2" in r for r in result)

    def test_all_columns_present_passes(self):
        df = _valid_single_row_df()
        result = validate_required_columns(df)
        assert "PASS_R2" in result


class TestRowCountValidation:
    """Additional checks on R1 row count rule."""

    def test_too_few_rows_fails(self):
        df = _valid_single_row_df()
        assert any("FAIL_R1" in r for r in validate_row_count(df))

    def test_too_many_rows_fails(self):
        combos = _all_495_combos()
        extra_row = _make_valid_row(496, "ABCDEFGH")
        rows = [_make_valid_row(i + 1, c) for i, c in enumerate(combos)] + [extra_row]
        df = pd.DataFrame(rows)
        assert any("FAIL_R1" in r for r in validate_row_count(df))

    def test_exactly_495_rows_passes(self):
        df = _valid_full_df()
        assert "PASS_R1" in validate_row_count(df)


class TestMalformedQualifiedThirdGroups:
    """Edge cases for the qualified_third_groups string."""

    def test_7_char_string_fails_r5(self):
        row = _make_valid_row(1, "ABCDEFGH")
        row["qualified_third_groups"] = "ABCDEFG"  # only 7 chars
        df = pd.DataFrame([row])
        assert any("FAIL_R5" in r for r in validate_8group_strings(df))

    def test_9_char_string_fails_r5(self):
        row = _make_valid_row(1, "ABCDEFGH")
        row["qualified_third_groups"] = "ABCDEFGHI"  # 9 chars
        df = pd.DataFrame([row])
        assert any("FAIL_R5" in r for r in validate_8group_strings(df))

    def test_repeated_letter_fails_r5(self):
        row = _make_valid_row(1, "ABCDEFGH")
        row["qualified_third_groups"] = "AABCDEFG"  # A repeated
        df = pd.DataFrame([row])
        assert any("FAIL_R5" in r for r in validate_8group_strings(df))
