"""Tests for the Poisson scoreline distribution."""
import numpy as np

from src.models.baselines import PoissonScoreModel, wdl_from_matrix


def test_score_matrix_sums_to_one():
    m = PoissonScoreModel(max_goals=10)
    for lh, la in [(1.2, 0.9), (2.5, 0.3), (0.5, 0.5), (3.0, 2.0)]:
        M = m.score_matrix(lh, la)
        assert M.shape == (11, 11)
        assert abs(M.sum() - 1.0) < 1e-9
        assert (M >= 0).all()


def test_wdl_from_matrix_sums_to_one_and_orders():
    m = PoissonScoreModel()
    # strong home lambda -> home win most likely
    M = m.score_matrix(3.0, 0.5)
    p = wdl_from_matrix(M)
    assert abs(p.sum() - 1.0) < 1e-9
    assert p[0] > p[2]  # home_win > away_win


def test_higher_away_lambda_favours_away():
    m = PoissonScoreModel()
    p = wdl_from_matrix(m.score_matrix(0.5, 3.0))
    assert p[2] > p[0]  # away_win > home_win
