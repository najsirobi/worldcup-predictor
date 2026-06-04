"""Tests for live group tables and simulation (Travel Mode, Tasks B & C)."""

import pandas as pd

from src.live.live_simulation import simulate_live
from src.live.scores_override import build_initial_override, update_match
from src.live.tournament_state import compute_group_tables, split_played_remaining


def _predictions():
    return pd.read_csv(
        "outputs/final_candidate_v1/final_group_score_predictions.csv"
    )


def test_live_table_reflects_score():
    frame = build_initial_override()
    # Match 1 is Mexico vs South Africa in Group A.
    frame = update_match(frame, match_number=1, team_a_goals=2, team_b_goals=1)
    tables = compute_group_tables(frame)

    group_a = tables[tables["group"] == "A"].set_index("team")
    assert group_a.loc["Mexico", "points"] == 3
    assert group_a.loc["Mexico", "goals_for"] == 2
    assert group_a.loc["Mexico", "goals_against"] == 1
    assert group_a.loc["Mexico", "goal_difference"] == 1
    assert group_a.loc["South Africa", "points"] == 0
    # Mexico ranks above South Africa.
    assert group_a.loc["Mexico", "rank"] < group_a.loc["South Africa", "rank"]


def test_every_team_has_a_table_row_from_the_start():
    frame = build_initial_override()
    tables = compute_group_tables(frame)
    assert len(tables) == 48  # 12 groups x 4 teams
    assert (tables["played"] == 0).all()
    assert set(tables["group"]) == set("ABCDEFGHIJKL")


def test_split_played_remaining_counts():
    frame = build_initial_override()
    frame = update_match(frame, match_number=1, team_a_goals=1, team_b_goals=0)
    frame = update_match(frame, match_number=2, team_a_goals=0, team_b_goals=0)
    played, remaining = split_played_remaining(frame)
    assert len(played) == 2
    assert len(remaining) == 70


def test_simulation_probabilities_are_valid_and_sum_per_rank():
    frame = build_initial_override()
    summary = simulate_live(frame, _predictions(), n_sims=2000, seed=1)
    assert len(summary) == 48
    # Probabilities in [0, 1].
    for col in ("p_rank1", "p_rank2", "p_rank3", "p_rank4", "p_advance"):
        assert summary[col].between(0, 1).all()
    # Each group has exactly ~1.0 mass on rank1 across its 4 teams.
    for _, sub in summary.groupby("group"):
        assert abs(sub["p_rank1"].sum() - 1.0) < 0.02


def test_played_result_shifts_advancement_probability():
    base = simulate_live(build_initial_override(), _predictions(), n_sims=4000, seed=7)
    # Give South Africa a big win to lift its advancement odds.
    frame = update_match(build_initial_override(), match_number=1, team_a_goals=0, team_b_goals=5)
    shifted = simulate_live(frame, _predictions(), n_sims=4000, seed=7)

    base_sa = base[(base.group == "A") & (base.team == "South Africa")]["p_advance"].iloc[0]
    shift_sa = shifted[(shifted.group == "A") & (shifted.team == "South Africa")]["p_advance"].iloc[0]
    assert shift_sa > base_sa
