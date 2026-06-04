"""Tests for the Monte Carlo group-stage simulation + predictions CSV shape."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.simulation.group_stage import simulate_groups

ROOT = Path(__file__).parent.parent


def _round_robin_group():
    teams = ["T1", "T2", "T3", "T4"]
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    rows = []
    for i, (a, b) in enumerate(pairs, start=1):
        rows.append({"group": "A", "match_number": i, "team_a": teams[a], "team_b": teams[b]})
    return pd.DataFrame(rows), teams


def _uniform_matrix(n=4):
    M = np.ones((n, n))
    return M / M.sum()


def test_simulation_produces_valid_ranks_and_probs():
    gm, teams = _round_robin_group()
    matrices = {i: _uniform_matrix() for i in gm["match_number"]}
    summary = simulate_groups(gm, matrices, n_sims=500, seed=1)

    assert len(summary) == 4
    # ranks are within 1..4
    assert set(summary["most_likely_group_rank"]).issubset({1, 2, 3, 4})
    # finish-position probabilities for the group sum to ~1 per position
    for col in ["p_finish_1st", "p_finish_2nd", "p_finish_3rd", "p_finish_4th"]:
        assert abs(summary[col].sum() - 1.0) < 0.02
    # p_top2 consistent and bounded
    assert ((summary["p_top2"] >= 0) & (summary["p_top2"] <= 1)).all()
    np.testing.assert_allclose(
        summary["p_top2"], summary["p_finish_1st"] + summary["p_finish_2nd"], atol=1e-9)


def test_each_team_appears_once():
    gm, teams = _round_robin_group()
    matrices = {i: _uniform_matrix() for i in gm["match_number"]}
    summary = simulate_groups(gm, matrices, n_sims=300, seed=2)
    assert sorted(summary["team"]) == sorted(teams)


def test_predictions_csv_has_72_rows_when_template_has_72():
    template = pd.read_csv(ROOT / "data" / "reference" / "fif8a_group_stage_template.csv")
    pred_path = ROOT / "outputs" / "predictions" / "fif8a_group_stage_predictions.csv"
    if not pred_path.exists():
        pytest.skip("predictions CSV not generated yet (run generate_group_stage_predictions.py)")
    pred = pd.read_csv(pred_path)
    assert len(pred) == len(template) == 72
