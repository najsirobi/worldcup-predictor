"""Tests for explicit knockout bracket mapping validation."""

import pandas as pd
import pytest

from src.simulation.knockout_bracket import (
    MissingBracketMappingError,
    assign_official_r32_matches,
    assign_r32_teams,
    ensure_manual_bracket_template,
    load_round_of_32_mapping,
    load_third_place_annex,
    load_bracket_mapping,
    validate_bracket_mapping,
)


def synthetic_mapping() -> pd.DataFrame:
    rows = []
    groups = list("ABCDEFGH")
    positions = [f"{group}{rank}" for group in groups for rank in range(1, 5)]
    for idx, source in enumerate(positions, start=1):
        rows.append(
            {
                "slot": f"R32_{idx:02d}",
                "round": "R32",
                "side_of_bracket": "left" if idx <= 16 else "right",
                "source_group_position": source,
                "opponent_slot": f"R32_{idx + 1:02d}" if idx % 2 else f"R32_{idx - 1:02d}",
                "notes": "synthetic test mapping",
            }
        )
    return pd.DataFrame(rows)


def test_bracket_mapping_creates_valid_32_team_r32_when_available():
    mapping = synthetic_mapping()
    source = {row["source_group_position"]: f"Team {idx:02d}" for idx, row in mapping.iterrows()}

    validate_bracket_mapping(mapping)
    assigned = assign_r32_teams(mapping, source)

    assert len(assigned) == 32
    assert assigned["team"].nunique() == 32


def test_missing_bracket_mapping_fails_clearly_and_creates_manual_template(tmp_path):
    missing = tmp_path / "knockout_bracket_mapping.csv"
    manual = tmp_path / "knockout_bracket_mapping_manual.csv"

    with pytest.raises(MissingBracketMappingError):
        load_bracket_mapping(missing, manual_template_path=manual)

    template = pd.read_csv(manual)
    assert len(template) == 32
    assert set(["slot", "round", "source_group_position", "opponent_slot"]).issubset(template.columns)


def test_manual_template_is_idempotent(tmp_path):
    manual = ensure_manual_bracket_template(tmp_path / "manual.csv")
    first = pd.read_csv(manual)
    ensure_manual_bracket_template(manual)
    second = pd.read_csv(manual)

    assert first.equals(second)


def test_official_r32_mapping_has_16_matches():
    mapping = load_round_of_32_mapping("data/reference/round_of_32_mapping.csv")

    assert len(mapping) == 16
    assert set(mapping["match_number"]) == set(range(73, 89))


def test_official_r32_assignment_has_no_duplicate_teams():
    mapping = load_round_of_32_mapping("data/reference/round_of_32_mapping.csv")
    annex = load_third_place_annex("data/reference/third_place_assignment_annex_c.csv")
    source_positions = {
        f"{group}{rank}": f"{group}{rank} Team"
        for group in "ABCDEFGHIJKL"
        for rank in range(1, 5)
    }

    assigned = assign_official_r32_matches(
        mapping,
        source_positions,
        annex,
        {"E", "F", "G", "H", "I", "J", "K", "L"},
    )

    teams = assigned["team_a"].tolist() + assigned["team_b"].tolist()
    assert len(teams) == 32
    assert len(set(teams)) == 32
