"""Tests for guarded full-tournament simulation helpers."""

import numpy as np
import pandas as pd
import pytest

from src.simulation.full_tournament import (
    simulate_full_tournament_from_mapping,
    simulate_full_tournament_official,
    validate_no_duplicate_round,
)


def synthetic_mapping() -> pd.DataFrame:
    groups = list("ABCDEFGH")
    positions = [f"{group}{rank}" for group in groups for rank in range(1, 5)]
    return pd.DataFrame(
        [
            {
                "slot": f"R32_{idx:02d}",
                "round": "R32",
                "side_of_bracket": "left" if idx <= 16 else "right",
                "source_group_position": source,
                "opponent_slot": f"R32_{idx + 1:02d}" if idx % 2 else f"R32_{idx - 1:02d}",
                "notes": "synthetic",
            }
            for idx, source in enumerate(positions, start=1)
        ]
    )


def synthetic_group_view() -> pd.DataFrame:
    rows = []
    for group in list("ABCDEFGH"):
        for rank in range(1, 5):
            strength = 5 - rank
            rows.append(
                {
                    "group": group,
                    "team": f"{group}{rank} Team",
                    "suggested_group_standing": rank,
                    "p_finish_1st": 0.7 if rank == 1 else 0.1,
                    "p_top2": 0.9 if rank <= 2 else 0.2,
                    "p_top3": 0.95 if rank <= 3 else 0.3,
                    "p_advance_with_best_thirds": 0.95 if rank <= 2 else 0.4,
                    "expected_points": strength,
                    "expected_goal_difference": strength - 2,
                    "likely_best_third_signal": 0.3 if rank == 3 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def test_no_duplicate_team_in_simulated_knockout_round():
    validate_no_duplicate_round(["A", "B", "C"])
    with pytest.raises(ValueError):
        validate_no_duplicate_round(["A", "B", "A"])


def test_full_tournament_probabilities_are_between_zero_and_one():
    summary = simulate_full_tournament_from_mapping(
        synthetic_mapping(),
        synthetic_group_view(),
        n_sims=50,
        seed=1,
    )

    probability_columns = [column for column in summary.columns if column.startswith("p_")]
    assert len(summary) == 32
    for column in probability_columns:
        assert summary[column].between(0, 1).all()


def test_official_full_simulation_probabilities_are_between_zero_and_one():
    r32 = pd.read_csv("data/reference/round_of_32_mapping.csv")
    progression = pd.read_csv("data/reference/knockout_round_progression.csv")
    annex = pd.read_csv("data/reference/third_place_assignment_annex_c.csv")
    matches = []
    group_view_rows = []
    match_number = 1
    for group in "ABCDEFGHIJKL":
        teams = [f"{group}{rank} Team" for rank in range(1, 5)]
        for rank, team in enumerate(teams, start=1):
            strength = 5 - rank
            group_view_rows.append(
                {
                    "group": group,
                    "team": team,
                    "expected_points": strength,
                    "expected_goal_difference": strength - 2,
                    "p_top2": 0.9 if rank <= 2 else 0.2,
                }
            )
        for idx, team_a in enumerate(teams):
            for team_b in teams[idx + 1 :]:
                matches.append(
                    {
                        "group": group,
                        "match_number": match_number,
                        "team_a": team_a,
                        "team_b": team_b,
                    }
                )
                match_number += 1

    matrices = {row["match_number"]: np.array([[0.25, 0.25], [0.20, 0.30]]) for row in matches}
    summary = simulate_full_tournament_official(
        pd.DataFrame(matches),
        matrices,
        r32,
        progression,
        annex,
        pd.DataFrame(group_view_rows),
        n_sims=10,
        seed=7,
    )

    probability_columns = [column for column in summary.columns if column.startswith("p_")]
    assert len(summary) == 48
    for column in probability_columns:
        assert summary[column].between(0, 1).all()
