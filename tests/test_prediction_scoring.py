"""Tests for prediction-vs-actual scoring (Travel Mode, Task E).

Verifies the RULES_AND_SCORING.md group-stage scoring: correct outcome pays
6 x template odd, +2 flat for exact goal difference, +3 flat for exact score,
both bonuses only on a correct outcome.
"""

import pandas as pd

from src.live.prediction_scoring import (
    ScoringRules,
    load_scoring_rules,
    load_template_odds,
    score_match,
    score_predictions_vs_actuals,
    summarise,
)

RULES = ScoringRules(base=6, gd_bonus=2, exact_bonus=3)
# team_a win odd 2.0, draw odd 3.5, team_b win odd 4.0.
ODDS = {"team_a": 2.0, "draw": 3.5, "team_b": 4.0}


def test_wrong_outcome_earns_zero_points():
    # Predicted team_a win (1-0), actual team_b win (0-1).
    r = score_match((1, 0), (0, 1), ODDS, RULES)
    assert r["outcome_correct"] is False
    assert r["outcome_points"] == 0
    assert r["goal_difference_bonus"] == 0
    assert r["exact_score_bonus"] == 0
    assert r["total_points"] == 0


def test_correct_outcome_earns_six_times_odd():
    # Predicted 1-0, actual 3-1: team_a win correct, GD wrong, score wrong.
    r = score_match((1, 0), (3, 1), ODDS, RULES)
    assert r["outcome_correct"] is True
    assert r["outcome_points"] == 6 * 2.0
    assert r["goal_difference_bonus"] == 0
    assert r["exact_score_bonus"] == 0
    assert r["total_points"] == 12.0


def test_correct_goal_difference_adds_two():
    # Predicted 1-0, actual 2-1: team_a win + same GD (+1), but not exact score.
    r = score_match((1, 0), (2, 1), ODDS, RULES)
    assert r["goal_difference_correct"] is True
    assert r["exact_score_correct"] is False
    assert r["goal_difference_bonus"] == 2
    assert r["exact_score_bonus"] == 0
    assert r["total_points"] == 6 * 2.0 + 2


def test_exact_score_adds_three():
    # Predicted 2-1, actual 2-1: outcome + GD + exact all correct.
    r = score_match((2, 1), (2, 1), ODDS, RULES)
    assert r["exact_score_correct"] is True
    assert r["goal_difference_bonus"] == 2
    assert r["exact_score_bonus"] == 3
    assert r["total_points"] == 6 * 2.0 + 2 + 3


def test_draw_scoring_uses_rate_draw():
    # Predicted 1-1, actual 1-1: draw, exact -> 6*draw_odd + 2 + 3.
    r = score_match((1, 1), (1, 1), ODDS, RULES)
    assert r["predicted_outcome"] == "draw"
    assert r["applicable_odd"] == 3.5
    assert r["outcome_points"] == 6 * 3.5
    assert r["total_points"] == 6 * 3.5 + 2 + 3


def test_max_possible_uses_actual_outcome_odd():
    # Predicted team_a win but actual is a team_b win: max possible is the perfect
    # call on the actual (team_b) outcome odd plus both bonuses.
    r = score_match((1, 0), (0, 2), ODDS, RULES)
    assert r["max_possible_points_for_match"] == 6 * 4.0 + 2 + 3
    assert r["points_missed"] == r["max_possible_points_for_match"]


def test_repo_scoring_rules_match_spec():
    rules = load_scoring_rules()
    assert rules.base == 6
    assert rules.gd_bonus == 2
    assert rules.exact_bonus == 3


def test_score_frame_and_summary_only_uses_played():
    predictions = pd.DataFrame(
        {
            "match_number": [1, 2],
            "group": ["A", "A"],
            "team_a": ["X", "Y"],
            "team_b": ["P", "Q"],
            "final_recommended_score": ["1-0", "1-0"],
        }
    )
    scores = pd.DataFrame(
        {
            "match_number": [1, 2],
            "group": ["A", "A"],
            "team_a": ["X", "Y"],
            "team_b": ["P", "Q"],
            "team_a_goals": [2, None],
            "team_b_goals": [1, None],
            "status": ["played", "scheduled"],
        }
    )
    odds = {1: ODDS, 2: ODDS}
    detail = score_predictions_vs_actuals(predictions, scores, odds, RULES)
    assert len(detail) == 1  # only the played match scored
    summary = summarise(detail)
    assert summary["played_matches"] == 1
    assert summary["outcomes_correct"] == 1
    assert summary["goal_differences_correct"] == 1
    assert "A" in summary["total_by_group"]


def test_template_odds_load_for_all_matches():
    odds = load_template_odds()
    assert len(odds) == 72
    assert set(odds[1]) == {"team_a", "draw", "team_b"}
