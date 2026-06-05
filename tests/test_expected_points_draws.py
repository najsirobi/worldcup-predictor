"""Focused tests for draw scoring and expected-points behavior."""

import numpy as np

from src.evaluation.auto_consensus import score_outcome
from src.evaluation.expected_points import expected_points_for_score
from src.evaluation.scoring import match_outcome, score_group_match_prediction


RULES = {
    "group_match_correct_outcome_base_points": 6,
    "group_match_exact_goal_difference_bonus": 2,
    "group_match_exact_score_bonus": 3,
}
EXPECTED_POINTS_ODDS = {"a_win": 2.0, "draw": 4.0, "b_win": 5.0}
SCORING_ODDS = {"home": 2.0, "draw": 4.0, "away": 5.0}


def _certain_matrix(goals_a: int, goals_b: int, n: int = 5) -> np.ndarray:
    matrix = np.zeros((n, n))
    matrix[goals_a, goals_b] = 1.0
    return matrix


def test_one_one_is_recognised_as_draw():
    assert score_outcome("1-1") == "draw"
    assert match_outcome(1, 1) == "draw"


def test_zero_zero_is_recognised_as_draw():
    assert score_outcome("0-0") == "draw"
    assert match_outcome(0, 0) == "draw"


def test_draw_outcome_uses_rate_draw():
    points = score_group_match_prediction(2, 2, 2, 2, SCORING_ODDS, RULES)

    assert points == 6 * SCORING_ODDS["draw"] + 2 + 3


def test_exact_draw_score_stacks_outcome_goal_difference_and_exact_score():
    matrix = _certain_matrix(1, 1)

    expected_points = expected_points_for_score(1, 1, matrix, EXPECTED_POINTS_ODDS, RULES)

    assert expected_points == 6 * EXPECTED_POINTS_ODDS["draw"] + 2 + 3


def test_wrong_outcome_draw_candidate_scores_zero():
    matrix = _certain_matrix(2, 1)

    expected_points = expected_points_for_score(1, 1, matrix, EXPECTED_POINTS_ODDS, RULES)
    deterministic_points = score_group_match_prediction(1, 1, 2, 1, SCORING_ODDS, RULES)

    assert expected_points == 0.0
    assert deterministic_points == 0.0
