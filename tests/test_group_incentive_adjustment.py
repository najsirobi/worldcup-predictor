from pathlib import Path

import numpy as np
import pandas as pd

from src.features.group_incentives import (
    IncentiveAdjustmentConfig,
    adjust_score_matrix_for_incentives,
    build_live_incentive_diagnostics,
    outcome_probabilities_from_matrix,
    poisson_score_matrix,
)
from src.live.scores_override import build_initial_override, update_match
from src.live.submission_guard import sha256_file
from src.simulation.group_stage import simulate_groups


def _one_score_matrix(goals_a: int, goals_b: int, size: int = 4) -> np.ndarray:
    matrix = np.zeros((size, size), dtype=float)
    matrix[goals_a, goals_b] = 1.0
    return matrix


def test_incentive_adjustment_is_capped():
    matrix = poisson_score_matrix(3.0, 0.6)
    state_low = {"low_incentive_flag": True, "high_incentive_flag": False}
    state_normal = {"low_incentive_flag": False, "high_incentive_flag": False}
    config = IncentiveAdjustmentConfig(
        low_xg_factor=0.9,
        max_xg_shift=0.15,
        max_probability_shift=0.05,
    )

    adjusted, meta = adjust_score_matrix_for_incentives(
        matrix,
        3.0,
        0.6,
        state_low,
        state_normal,
        final_group_match=True,
        config=config,
    )

    assert meta["applied"] is True
    assert abs(meta["lambda_a_shift"]) <= 0.15 + 1e-12
    assert abs(meta["lambda_b_shift"]) <= 0.15 + 1e-12
    original_probs = outcome_probabilities_from_matrix(matrix)
    adjusted_probs = outcome_probabilities_from_matrix(adjusted)
    assert float(np.max(np.abs(adjusted_probs - original_probs))) <= 0.05 + 1e-12


def test_simulation_path_updates_incentives_after_simulated_prior_matches():
    matches = pd.DataFrame(
        [
            {"group": "A", "match_number": 1, "date": "2026-06-01", "team_a": "A", "team_b": "B"},
            {"group": "A", "match_number": 2, "date": "2026-06-02", "team_a": "A", "team_b": "C"},
            {"group": "A", "match_number": 3, "date": "2026-06-02", "team_a": "B", "team_b": "D"},
            {"group": "A", "match_number": 4, "date": "2026-06-03", "team_a": "C", "team_b": "D"},
            {"group": "A", "match_number": 5, "date": "2026-06-04", "team_a": "A", "team_b": "D"},
            {"group": "A", "match_number": 6, "date": "2026-06-04", "team_a": "B", "team_b": "C"},
        ]
    )
    matrices = {
        1: _one_score_matrix(1, 0),
        2: _one_score_matrix(1, 0),
        3: _one_score_matrix(0, 0),
        4: _one_score_matrix(1, 0),
        5: poisson_score_matrix(2.0, 0.5, max_goals=3),
        6: _one_score_matrix(1, 0),
    }
    lambdas = {5: (2.0, 0.5)}

    _, diagnostics = simulate_groups(
        matches,
        matrices,
        n_sims=20,
        seed=3,
        incentive_adjustment_config=IncentiveAdjustmentConfig(low_xg_factor=0.07),
        lambdas=lambdas,
        return_diagnostics=True,
    )

    assert diagnostics["adjustment_applications"] > 0
    assert diagnostics["low_incentive_team_counts"].get("A", 0) > 0
    assert diagnostics["adjusted_match_numbers"].get(5, 0) > 0


def test_live_incentive_diagnostics_do_not_modify_frozen_predictions():
    score_file = Path("outputs/final_candidate_v2_auto_science/final_group_score_predictions_auto.csv")
    before = sha256_file(score_file)

    scores = build_initial_override()
    # One actual result is enough to prove diagnostics are read-only; no submitted
    # score file is an input to this helper.
    scores = update_match(scores, match_number=1, team_a_goals=2, team_b_goals=0)
    diagnostics = build_live_incentive_diagnostics(scores)

    assert "matches" in diagnostics
    assert "teams" in diagnostics
    assert sha256_file(score_file) == before
