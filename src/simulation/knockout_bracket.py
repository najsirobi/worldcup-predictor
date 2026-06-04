"""Knockout bracket mapping utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ingest.fifa_regulations import ANNEX_SLOT_COLUMNS, lookup_annex_assignment

REQUIRED_COLUMNS = [
    "slot",
    "round",
    "side_of_bracket",
    "source_group_position",
    "opponent_slot",
    "notes",
]

ROUND_SIZES = {
    "R32": 32,
    "R16": 16,
    "QF": 8,
    "SF": 4,
    "Final": 2,
    "Winner": 1,
}


class MissingBracketMappingError(FileNotFoundError):
    """Raised when the explicit knockout bracket mapping is not available."""


def manual_template_rows() -> list[dict[str, str]]:
    rows = []
    for idx in range(1, 33):
        slot = f"R32_{idx:02d}"
        opponent = f"R32_{idx + 1:02d}" if idx % 2 else f"R32_{idx - 1:02d}"
        rows.append(
            {
                "slot": slot,
                "round": "R32",
                "side_of_bracket": "",
                "source_group_position": "",
                "opponent_slot": opponent,
                "notes": "Manual input required: official source group/rank/best-third slot not available locally.",
            }
        )
    return rows


def ensure_manual_bracket_template(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        pd.DataFrame(manual_template_rows(), columns=REQUIRED_COLUMNS).to_csv(out, index=False)
    return out


def load_bracket_mapping(path: str | Path, *, manual_template_path: str | Path | None = None) -> pd.DataFrame:
    mapping_path = Path(path)
    if not mapping_path.exists():
        if manual_template_path is not None:
            ensure_manual_bracket_template(manual_template_path)
        raise MissingBracketMappingError(
            f"Missing knockout bracket mapping: {mapping_path}. "
            "Populate data/reference/knockout_bracket_mapping_manual.csv from an official source."
        )
    mapping = pd.read_csv(mapping_path)
    validate_bracket_mapping(mapping)
    return mapping


def validate_bracket_mapping(mapping: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in mapping.columns]
    if missing:
        raise ValueError(f"Bracket mapping missing columns: {missing}")
    r32 = mapping[mapping["round"].eq("R32")].copy()
    if len(r32) != 32:
        raise ValueError(f"Round of 32 mapping must contain 32 slots, found {len(r32)}.")
    if r32["slot"].duplicated().any():
        duplicated = sorted(r32.loc[r32["slot"].duplicated(), "slot"].astype(str).unique())
        raise ValueError(f"Duplicate R32 slots in bracket mapping: {duplicated}")
    if r32["source_group_position"].isna().any() or r32["source_group_position"].eq("").any():
        raise ValueError("Every R32 slot must have an explicit source_group_position.")
    slot_set = set(r32["slot"].astype(str))
    opponent_set = set(r32["opponent_slot"].astype(str))
    if not opponent_set.issubset(slot_set):
        raise ValueError(f"Opponent slots not present in R32 slots: {sorted(opponent_set - slot_set)}")
    for _, row in r32.iterrows():
        opponent = r32.loc[r32["slot"].eq(row["opponent_slot"])]
        if opponent.empty or opponent.iloc[0]["opponent_slot"] != row["slot"]:
            raise ValueError(f"Opponent relationship is not symmetric for slot {row['slot']}.")


def assign_r32_teams(mapping: pd.DataFrame, source_positions: dict[str, str]) -> pd.DataFrame:
    """Assign source positions such as A1/B2/BT1 to R32 slots from an explicit mapping."""
    validate_bracket_mapping(mapping)
    r32 = mapping[mapping["round"].eq("R32")].copy()
    r32["team"] = r32["source_group_position"].map(source_positions)
    if r32["team"].isna().any():
        missing = sorted(r32.loc[r32["team"].isna(), "source_group_position"].astype(str).unique())
        raise ValueError(f"Missing teams for bracket source positions: {missing}")
    if r32["team"].duplicated().any():
        duplicated = sorted(r32.loc[r32["team"].duplicated(), "team"].astype(str).unique())
        raise ValueError(f"Duplicate teams assigned to R32 bracket: {duplicated}")
    return r32


OFFICIAL_R32_COLUMNS = [
    "match_number",
    "team_a_source",
    "team_b_source",
    "team_a_source_type",
    "team_b_source_type",
    "team_a_group",
    "team_a_position",
    "team_b_group_options",
    "team_b_position",
    "source",
    "confidence",
    "notes",
]

OFFICIAL_PROGRESSION_COLUMNS = [
    "round",
    "match_number",
    "team_a_source",
    "team_b_source",
    "winner_to_match",
    "loser_to_match_if_applicable",
    "source",
    "notes",
]


def load_round_of_32_mapping(path: str | Path) -> pd.DataFrame:
    mapping_path = Path(path)
    if not mapping_path.exists():
        raise MissingBracketMappingError(f"Missing official Round-of-32 mapping: {mapping_path}")
    mapping = pd.read_csv(mapping_path)
    validate_round_of_32_official(mapping)
    return mapping


def load_round_progression(path: str | Path) -> pd.DataFrame:
    progression_path = Path(path)
    if not progression_path.exists():
        raise MissingBracketMappingError(f"Missing official knockout progression mapping: {progression_path}")
    progression = pd.read_csv(progression_path)
    validate_round_progression_official(progression)
    return progression


def load_third_place_annex(path: str | Path) -> pd.DataFrame:
    annex_path = Path(path)
    if not annex_path.exists():
        raise MissingBracketMappingError(f"Missing official Annexe C assignment table: {annex_path}")
    annex = pd.read_csv(annex_path)
    missing = {"option_number", "qualified_third_groups", *ANNEX_SLOT_COLUMNS} - set(annex.columns)
    if missing:
        raise ValueError(f"Annexe C mapping missing columns: {sorted(missing)}")
    if len(annex) != 495:
        raise ValueError(f"Annexe C mapping must contain 495 rows, found {len(annex)}")
    return annex


def validate_round_of_32_official(mapping: pd.DataFrame) -> None:
    missing = [column for column in OFFICIAL_R32_COLUMNS if column not in mapping.columns]
    if missing:
        raise ValueError(f"Official R32 mapping missing columns: {missing}")
    if len(mapping) != 16:
        raise ValueError(f"Official R32 mapping must contain 16 matches, found {len(mapping)}")
    match_numbers = set(mapping["match_number"].astype(int))
    if match_numbers != set(range(73, 89)):
        raise ValueError(f"Official R32 mapping must use match numbers 73-88, found {sorted(match_numbers)}")


def validate_round_progression_official(progression: pd.DataFrame) -> None:
    missing = [column for column in OFFICIAL_PROGRESSION_COLUMNS if column not in progression.columns]
    if missing:
        raise ValueError(f"Official progression mapping missing columns: {missing}")
    match_numbers = set(progression["match_number"].astype(int))
    if match_numbers != set(range(89, 105)):
        raise ValueError(f"Official progression mapping must use match numbers 89-104, found {sorted(match_numbers)}")
    if not progression["winner_to_match"].astype(str).eq("Winner").any():
        raise ValueError("Official progression mapping does not reach Winner.")


def fifa_source_to_group_position(source: str) -> str:
    """Convert FIFA source strings like `1A`/`2B`/`3E` to internal `A1`/`B2`/`E3`."""
    value = str(source).strip()
    if len(value) != 2 or value[0] not in {"1", "2", "3"} or not value[1].isalpha():
        raise ValueError(f"Unsupported FIFA bracket source: {source}")
    return f"{value[1]}{value[0]}"


def assign_official_r32_matches(
    r32_mapping: pd.DataFrame,
    source_positions: dict[str, str],
    annex: pd.DataFrame,
    qualified_third_groups: list[str] | set[str],
) -> pd.DataFrame:
    """Fill official M73-M88 Round-of-32 matches from group standings and Annexe C."""
    validate_round_of_32_official(r32_mapping)
    annex_row = lookup_annex_assignment(annex, set(qualified_third_groups))
    third_team_by_group = {
        source[0]: team
        for source, team in source_positions.items()
        if len(source) == 2 and source.endswith("3")
    }
    rows = []
    for _, row in r32_mapping.sort_values("match_number").iterrows():
        team_a_position = fifa_source_to_group_position(row["team_a_source"])
        team_a = source_positions.get(team_a_position)
        if team_a is None:
            raise ValueError(f"Missing team for R32 source position {team_a_position}")

        if row["team_b_source_type"] == "best_third":
            slot_column = f"slot_{row['team_a_source']}"
            if slot_column not in ANNEX_SLOT_COLUMNS:
                raise ValueError(f"Unsupported Annexe C slot column for match {row['match_number']}: {slot_column}")
            assigned_source = str(annex_row[slot_column])
            assigned_group = assigned_source[1]
            team_b = third_team_by_group.get(assigned_group)
            if team_b is None:
                raise ValueError(f"Missing qualified third-place team for group {assigned_group}")
            team_b_position = fifa_source_to_group_position(assigned_source)
        else:
            team_b_position = fifa_source_to_group_position(row["team_b_source"])
            team_b = source_positions.get(team_b_position)
            if team_b is None:
                raise ValueError(f"Missing team for R32 source position {team_b_position}")

        rows.append(
            {
                "match_number": int(row["match_number"]),
                "round": "R32",
                "team_a": team_a,
                "team_b": team_b,
                "team_a_source_position": team_a_position,
                "team_b_source_position": team_b_position,
                "team_a_source": row["team_a_source"],
                "team_b_source": row["team_b_source"],
            }
        )
    assigned = pd.DataFrame(rows)
    teams = assigned["team_a"].tolist() + assigned["team_b"].tolist()
    if len(teams) != 32 or len(set(teams)) != 32:
        duplicated = sorted({team for team in teams if teams.count(team) > 1})
        raise ValueError(f"Official R32 assignment must contain 32 unique teams; duplicates={duplicated}")
    return assigned
