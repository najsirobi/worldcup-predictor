"""Tests for Team A / Team B score-orientation helpers."""

import numpy as np
import pandas as pd

from src.evaluation.group_stage_predictions import (
    add_score_columns,
    most_probable_score_for_outcome,
    parse_score,
    score_display,
)


def test_most_probable_score_for_outcome_keeps_team_a_team_b_goal_order():
    M = np.zeros((4, 4))
    M[1, 1] = 0.21
    M[2, 1] = 0.20
    M[0, 1] = 0.19
    M[1, 0] = 0.18

    assert most_probable_score_for_outcome(M, "a_win") == (2, 1)
    assert most_probable_score_for_outcome(M, "draw") == (1, 1)
    assert most_probable_score_for_outcome(M, "b_win") == (0, 1)


def test_add_score_columns_builds_explicit_goal_columns_and_displays():
    pred = pd.DataFrame(
        [
            {
                "team_a": "Qatar",
                "team_b": "Switzerland",
                "recommended_score_safe": "0-2",
                "recommended_score_ev": "1-2",
            }
        ]
    )

    out = add_score_columns(pred)

    assert parse_score(out.loc[0, "recommended_score_safe"]) == (0, 2)
    assert out.loc[0, "recommended_team_a_goals_safe"] == 0
    assert out.loc[0, "recommended_team_b_goals_safe"] == 2
    assert out.loc[0, "recommended_team_a_goals_ev"] == 1
    assert out.loc[0, "recommended_team_b_goals_ev"] == 2
    assert out.loc[0, "recommended_score_safe_display"] == score_display("Qatar", "Switzerland", (0, 2))
    assert out.loc[0, "recommended_score_ev_display"] == score_display("Qatar", "Switzerland", (1, 2))
