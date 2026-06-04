"""Tests for official FIFA Annexe C extraction and lookup."""

from pathlib import Path

import pandas as pd

from src.ingest.fifa_regulations import (
    ANNEX_SLOT_COLUMNS,
    lookup_annex_assignment,
    validate_annex_c,
)


def test_annex_c_table_has_495_rows():
    annex = pd.read_csv(Path("data/reference/third_place_assignment_annex_c.csv"))

    validate_annex_c(annex)
    assert len(annex) == 495


def test_each_annex_c_row_assigns_eight_unique_third_place_groups():
    annex = pd.read_csv(Path("data/reference/third_place_assignment_annex_c.csv"))

    for _, row in annex.iterrows():
        assignments = [row[column] for column in ANNEX_SLOT_COLUMNS]
        groups = [assignment[1] for assignment in assignments]
        qualified = row["qualified_third_groups"].split(",")

        assert len(assignments) == 8
        assert len(set(assignments)) == 8
        assert sorted(groups) == sorted(qualified)


def test_annex_c_lookup_works_for_known_official_option():
    annex = pd.read_csv(Path("data/reference/third_place_assignment_annex_c.csv"))

    row = lookup_annex_assignment(annex, {"E", "F", "G", "H", "I", "J", "K", "L"})

    assert int(row["option_number"]) == 1
    assert row["slot_1A"] == "3E"
    assert row["slot_1L"] == "3K"
