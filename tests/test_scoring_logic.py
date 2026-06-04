"""Test FIF8A scoring logic with tiny synthetic examples."""
import pytest

from src.evaluation.scoring import (
    score_group_match_prediction,
    expected_group_match_points,
    score_group_standing_prediction,
    score_last8_prediction,
    score_knockout_match_prediction,
)

ODDS = {"home": 2.0, "draw": 3.0, "away": 5.0}


# --- group match scoring ---

def test_exact_score_stacks_all_three_bonuses():
    # predict 2-1 (home win), actual 2-1 -> 6*2 + 2 (gd) + 3 (exact) = 17
    pts = score_group_match_prediction(2, 1, 2, 1, ODDS)
    assert pts == 6 * 2.0 + 2 + 3


def test_correct_outcome_only_earns_base_times_odd():
    # predict 1-0 (home win, gd=1), actual 3-1 (home win, gd=2) -> only 6*odd
    pts = score_group_match_prediction(1, 0, 3, 1, ODDS)
    assert pts == 6 * 2.0


def test_correct_outcome_with_goal_diff_no_exact():
    # predict 2-1 (gd=1), actual 3-2 (gd=1, home win) -> 6*odd + 2, no exact
    pts = score_group_match_prediction(2, 1, 3, 2, ODDS)
    assert pts == 6 * 2.0 + 2


def test_wrong_outcome_earns_nothing():
    # predict home win, actual away win -> 0 (no gd/exact bonus on wrong outcome)
    pts = score_group_match_prediction(2, 1, 0, 1, ODDS)
    assert pts == 0.0


def test_draw_uses_draw_odd():
    pts = score_group_match_prediction(1, 1, 1, 1, ODDS)
    assert pts == 6 * 3.0 + 2 + 3  # exact draw


def test_expected_points_of_fixed_prediction():
    # predict 1-0. actual is 1-0 w.p. 0.5 (->17... here odds home=2: 6*2+2+3=17),
    # away win 2-0 w.p. 0.5 (->0). expected = 8.5
    dist = {(1, 0): 0.5, (0, 2): 0.5}
    ep = expected_group_match_points(1, 0, dist, ODDS)
    assert ep == pytest.approx(0.5 * (6 * 2.0 + 2 + 3))


# --- group standing scoring ---

def test_exact_standing_earns_30_plus_60():
    order = ["A", "B", "C", "D"]
    assert score_group_standing_prediction(order, order) == 90


def test_correct_top2_any_order_earns_30():
    # same top-2 set {A,B} but swapped order, and 3rd/4th differ -> 30 only
    pts = score_group_standing_prediction(["B", "A", "D", "C"], ["A", "B", "C", "D"])
    assert pts == 30


def test_wrong_top2_earns_zero():
    pts = score_group_standing_prediction(["C", "D", "A", "B"], ["A", "B", "C", "D"])
    assert pts == 0


# --- last-8 progression block ---

def test_last8_block_sums_correctly():
    pred = {
        "quarter_finalists": ["A", "B", "C", "D", "E", "F", "G", "H"],
        "semi_finalists": ["A", "B", "C", "D"],
        "finalists": ["A", "B"],
        "winner": "A",
    }
    actual = {
        # 6 of 8 QF correct, 3 of 4 SF correct, 1 of 2 finalists, winner correct
        "quarter_finalists": ["A", "B", "C", "D", "E", "F", "X", "Y"],
        "semi_finalists": ["A", "B", "C", "Z"],
        "finalists": ["A", "Q"],
        "winner": "A",
    }
    # 6*20 + 3*40 + 1*60 + 100 = 120 + 120 + 60 + 100 = 400
    assert score_last8_prediction(pred, actual) == 400


def test_last8_winner_wrong():
    pred = {"quarter_finalists": [], "semi_finalists": [], "finalists": [], "winner": "A"}
    actual = {"quarter_finalists": [], "semi_finalists": [], "finalists": [], "winner": "B"}
    assert score_last8_prediction(pred, actual) == 0


# --- knockout match scoring (later phase) ---

def test_knockout_correct_team_and_bonuses():
    pts = score_knockout_match_prediction(
        "France", "France", exact_score_correct=True, penalty_shootout_correct=True, odd=2.0
    )
    assert pts == 6 * 2.0 + 2 + 2


def test_knockout_wrong_team_no_base():
    pts = score_knockout_match_prediction(
        "France", "Spain", exact_score_correct=False, penalty_shootout_correct=False, odd=2.0
    )
    assert pts == 0.0
