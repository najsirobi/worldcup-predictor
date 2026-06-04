"""Extract official FIFA World Cup 26 bracket tables from the regulations PDF."""

from __future__ import annotations

import itertools
import re
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

SOURCE = "FIFA World Cup 26 Regulations, Articles 12.6-12.11 and Annexe C"
ANNEX_SLOT_COLUMNS = ["slot_1A", "slot_1B", "slot_1D", "slot_1E", "slot_1G", "slot_1I", "slot_1K", "slot_1L"]
VALID_GROUPS = set("ABCDEFGHIJKL")


ROUND_OF_32_ROWS = [
    (73, "2A", "2B", "runner_up", "runner_up", "A", 2, "B", 2),
    (74, "1E", "best 3rd from A/B/C/D/F", "winner", "best_third", "E", 1, "A/B/C/D/F", 3),
    (75, "1F", "2C", "winner", "runner_up", "F", 1, "C", 2),
    (76, "1C", "2F", "winner", "runner_up", "C", 1, "F", 2),
    (77, "1I", "best 3rd from C/D/F/G/H", "winner", "best_third", "I", 1, "C/D/F/G/H", 3),
    (78, "2E", "2I", "runner_up", "runner_up", "E", 2, "I", 2),
    (79, "1A", "best 3rd from C/E/F/H/I", "winner", "best_third", "A", 1, "C/E/F/H/I", 3),
    (80, "1L", "best 3rd from E/H/I/J/K", "winner", "best_third", "L", 1, "E/H/I/J/K", 3),
    (81, "1D", "best 3rd from B/E/F/I/J", "winner", "best_third", "D", 1, "B/E/F/I/J", 3),
    (82, "1G", "best 3rd from A/E/H/I/J", "winner", "best_third", "G", 1, "A/E/H/I/J", 3),
    (83, "2K", "2L", "runner_up", "runner_up", "K", 2, "L", 2),
    (84, "1H", "2J", "winner", "runner_up", "H", 1, "J", 2),
    (85, "1B", "best 3rd from E/F/G/I/J", "winner", "best_third", "B", 1, "E/F/G/I/J", 3),
    (86, "1J", "2H", "winner", "runner_up", "J", 1, "H", 2),
    (87, "1K", "best 3rd from D/E/I/J/L", "winner", "best_third", "K", 1, "D/E/I/J/L", 3),
    (88, "2D", "2G", "runner_up", "runner_up", "D", 2, "G", 2),
]

PROGRESSION_ROWS = [
    ("R16", 89, "W74", "W77", "W89", ""),
    ("R16", 90, "W73", "W75", "W90", ""),
    ("R16", 91, "W76", "W78", "W91", ""),
    ("R16", 92, "W79", "W80", "W92", ""),
    ("R16", 93, "W83", "W84", "W93", ""),
    ("R16", 94, "W81", "W82", "W94", ""),
    ("R16", 95, "W86", "W88", "W95", ""),
    ("R16", 96, "W85", "W87", "W96", ""),
    ("QF", 97, "W89", "W90", "W97", ""),
    ("QF", 98, "W93", "W94", "W98", ""),
    ("QF", 99, "W91", "W92", "W99", ""),
    ("QF", 100, "W95", "W96", "W100", ""),
    ("SF", 101, "W97", "W98", "W101", "L103"),
    ("SF", 102, "W99", "W100", "W102", "L103"),
    ("Third-place", 103, "L101", "L102", "", ""),
    ("Final", 104, "W101", "W102", "Winner", ""),
]


def canonical_groups(groups: list[str] | set[str]) -> str:
    return ",".join(sorted(groups))


def round_of_32_mapping() -> pd.DataFrame:
    rows = []
    for match_number, team_a, team_b, type_a, type_b, group_a, pos_a, group_options_b, pos_b in ROUND_OF_32_ROWS:
        rows.append(
            {
                "match_number": match_number,
                "team_a_source": team_a,
                "team_b_source": team_b,
                "team_a_source_type": type_a,
                "team_b_source_type": type_b,
                "team_a_group": group_a,
                "team_a_position": pos_a,
                "team_b_group_options": group_options_b,
                "team_b_position": pos_b,
                "source": SOURCE,
                "confidence": "official_article_12_6",
                "notes": "Extracted from Article 12.6 fixed Round-of-32 configuration.",
            }
        )
    return pd.DataFrame(rows)


def knockout_round_progression() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "round": round_name,
                "match_number": match_number,
                "team_a_source": team_a,
                "team_b_source": team_b,
                "winner_to_match": winner,
                "loser_to_match_if_applicable": loser,
                "source": SOURCE,
                "notes": "Extracted from Articles 12.7-12.11.",
            }
            for round_name, match_number, team_a, team_b, winner, loser in PROGRESSION_ROWS
        ]
    )


def extract_annex_c_text(pdf_path: str | Path) -> str:
    reader = PdfReader(str(pdf_path))
    return "\n".join((reader.pages[idx].extract_text() or "") for idx in range(79, min(97, len(reader.pages))))


def parse_annex_c_text(text: str) -> pd.DataFrame:
    rows = []
    pattern = re.compile(r"^\s*(\d{1,3})\s+((?:3[A-L]\s+){7}3[A-L])\s*$")
    for line in text.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        option_number = int(match.group(1))
        assignments = match.group(2).split()
        groups = [assignment[1] for assignment in assignments]
        row = {
            "option_number": option_number,
            "qualified_third_groups": canonical_groups(groups),
            "source": SOURCE,
            "notes": "Extracted from Annexe C text table.",
        }
        row.update(dict(zip(ANNEX_SLOT_COLUMNS, assignments, strict=True)))
        rows.append(row)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("option_number").reset_index(drop=True)
    return frame


def validate_round_of_32_mapping(mapping: pd.DataFrame) -> None:
    if len(mapping) != 16:
        raise ValueError(f"R32 fixed mapping must contain 16 matches, found {len(mapping)}")
    expected = set(range(73, 89))
    found = set(mapping["match_number"].astype(int))
    if found != expected:
        raise ValueError(f"R32 match numbers invalid: expected {sorted(expected)}, found {sorted(found)}")


def validate_progression_mapping(progression: pd.DataFrame) -> None:
    expected = set(range(89, 105))
    found = set(progression["match_number"].astype(int))
    if found != expected:
        raise ValueError(f"Progression match numbers invalid: expected {sorted(expected)}, found {sorted(found)}")
    if not progression["winner_to_match"].astype(str).str.contains("Winner").any():
        raise ValueError("Progression mapping does not reach a final Winner.")


def validate_annex_c(annex: pd.DataFrame) -> None:
    if len(annex) != 495:
        raise ValueError(f"Annexe C must contain 495 rows, found {len(annex)}")
    if set(annex["option_number"].astype(int)) != set(range(1, 496)):
        raise ValueError("Annexe C option numbers must be exactly 1-495")
    expected_combinations = {canonical_groups(combo) for combo in itertools.combinations(sorted(VALID_GROUPS), 8)}
    found_combinations = set(annex["qualified_third_groups"])
    if found_combinations != expected_combinations:
        missing = sorted(expected_combinations - found_combinations)[:5]
        extra = sorted(found_combinations - expected_combinations)[:5]
        raise ValueError(f"Annexe C combinations mismatch; missing={missing}, extra={extra}")
    for _, row in annex.iterrows():
        assignments = [row[column] for column in ANNEX_SLOT_COLUMNS]
        groups = [str(value)[1] for value in assignments]
        qualified = str(row["qualified_third_groups"]).split(",")
        if len(set(assignments)) != 8:
            raise ValueError(f"Option {row['option_number']} has duplicate assignment: {assignments}")
        if sorted(groups) != sorted(qualified):
            raise ValueError(
                f"Option {row['option_number']} assignments {sorted(groups)} do not match qualified groups {sorted(qualified)}"
            )
        if any(group not in VALID_GROUPS for group in groups):
            raise ValueError(f"Option {row['option_number']} uses invalid group: {assignments}")


def build_combined_mapping(r32: pd.DataFrame, progression: pd.DataFrame, annex: pd.DataFrame) -> pd.DataFrame:
    frames = []
    frames.append(r32.assign(mapping_section="round_of_32"))
    frames.append(progression.assign(mapping_section="round_progression"))
    frames.append(annex.assign(mapping_section="annex_c"))
    return pd.concat(frames, ignore_index=True, sort=False)


def lookup_annex_assignment(annex: pd.DataFrame, qualified_groups: list[str] | set[str]) -> pd.Series:
    key = canonical_groups(set(qualified_groups))
    matches = annex.loc[annex["qualified_third_groups"].eq(key)]
    if len(matches) != 1:
        raise KeyError(f"Expected one Annexe C row for {key}, found {len(matches)}")
    return matches.iloc[0]
