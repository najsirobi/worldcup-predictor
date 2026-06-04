"""Tests for the expected-FIF8A-points logic."""
import numpy as np

from src.evaluation.expected_points import (
    expected_points_for_score, outcome_probs_from_matrix, most_probable_score,
)

RULES = {
    "group_match_correct_outcome_base_points": 6,
    "group_match_exact_goal_difference_bonus": 2,
    "group_match_exact_score_bonus": 3,
}
ODDS = {"a_win": 2.0, "draw": 4.0, "b_win": 5.0}


def _matrix_all_on(a, b, n=11):
    M = np.zeros((n, n))
    M[a, b] = 1.0
    return M


def test_exact_score_stacks_outcome_gd_and_exact():
    # actual is certainly 2-1 (a_win). Predict 2-1 -> 6*odd_a + 2 + 3
    M = _matrix_all_on(2, 1)
    ev = expected_points_for_score(2, 1, M, ODDS, RULES)
    assert ev == 6 * 2.0 + 2 + 3


def test_correct_outcome_only_no_exact_no_gd():
    # actual certainly 3-1 (a_win, gd 2). Predict 1-0 (a_win, gd 1) -> only 6*odd_a
    M = _matrix_all_on(3, 1)
    ev = expected_points_for_score(1, 0, M, ODDS, RULES)
    assert ev == 6 * 2.0


def test_correct_outcome_and_gd_no_exact():
    # actual 3-2 (a_win gd1). Predict 1-0 (a_win gd1) -> 6*odd_a + 2
    M = _matrix_all_on(3, 2)
    ev = expected_points_for_score(1, 0, M, ODDS, RULES)
    assert ev == 6 * 2.0 + 2


def test_wrong_outcome_zero():
    M = _matrix_all_on(0, 2)  # b_win
    ev = expected_points_for_score(1, 0, M, ODDS, RULES)  # predicted a_win
    assert ev == 0.0


def test_uses_correct_odds_for_each_outcome():
    # draw certain (1-1). predicting 1-1 must use the DRAW odd, not a/b.
    M = _matrix_all_on(1, 1)
    ev = expected_points_for_score(1, 1, M, ODDS, RULES)
    assert ev == 6 * 4.0 + 2 + 3  # draw odd = 4.0


def test_outcome_probs_and_most_probable_score():
    M = np.zeros((11, 11))
    M[2, 0] = 0.6  # a_win
    M[1, 1] = 0.4  # draw
    p = outcome_probs_from_matrix(M)
    assert abs(p.sum() - 1.0) < 1e-9
    assert most_probable_score(M) == (2, 0)
