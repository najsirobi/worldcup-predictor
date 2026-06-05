"""Tests for draw candidate generation and draw-aware selection policy."""

from __future__ import annotations

import pandas as pd

from src.evaluation.auto_consensus import (
    AutoPolicyConfig,
    collect_candidate_scores,
    select_final_scores,
)
from src.evaluation.draw_aware_policy import choose_draw_aware_hybrid_score, is_draw_score


def _base_prediction(
    safe: str = "1-0",
    ev: str = "1-1",
    most_probable: str = "1-1",
    ev_max: str = "1-1",
    ep_safe: float = 7.0,
    ep_ev: float = 7.4,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_number": 1,
                "group": "A",
                "team_a": "Team A",
                "team_b": "Team B",
                "model_p_a_win": 0.35,
                "model_p_draw": 0.33,
                "model_p_b_win": 0.32,
                "recommended_score_safe": safe,
                "recommended_score_ev": ev,
                "most_probable_score": most_probable,
                "ev_max_score": ev_max,
                "expected_points_safe": ep_safe,
                "expected_points_ev": ep_ev,
                "notes": "",
            }
        ]
    )


def _decision(uplift: float = 0.4, high_variance: bool = False, contrarian: bool = False) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_number": 1,
                "ev_uplift": uplift,
                "high_variance_flag": high_variance,
                "contrarian_flag": contrarian,
            }
        ]
    )


def _final_v1() -> pd.DataFrame:
    return pd.DataFrame([{"match_number": 1, "manual_review_flag": False}])


def test_auto_consensus_candidate_generation_includes_draw_scorelines():
    candidates, skipped = collect_candidate_scores(_base_prediction(), _decision(), None)

    assert skipped == ["match 1: ensemble score unavailable"]
    assert candidates["candidate_score"].map(is_draw_score).any()
    assert set(candidates.loc[candidates["candidate_score"].map(is_draw_score), "candidate_source"]) == {
        "ev_score",
        "most_probable_score",
        "expected_points_max_score",
    }


def test_draw_aware_policy_can_select_draw_in_synthetic_close_match():
    row = {
        "model_p_a_win": 0.35,
        "model_p_draw": 0.31,
        "model_p_b_win": 0.34,
    }

    selected, reason = choose_draw_aware_hybrid_score(
        row=row,
        current_score="1-0",
        best_draw_score="1-1",
        expected_points_current=7.0,
        expected_points_best_draw=7.05,
        modal_score="1-1",
    )

    assert selected == "1-1"
    assert reason == "draw_probability_close_to_selected_outcome"


def test_auto_policy_does_not_hard_code_zero_draws_when_gates_pass():
    predictions = _base_prediction(ep_safe=7.0, ep_ev=7.5)
    candidates = pd.DataFrame(
        [
            {"match_number": 1, "candidate_score": "1-1", "expected_points": 7.5},
            {"match_number": 1, "candidate_score": "1-1", "expected_points": 7.5},
            {"match_number": 1, "candidate_score": "1-0", "expected_points": 7.0},
        ]
    )

    output = select_final_scores(
        predictions,
        _final_v1(),
        _decision(uplift=0.5, high_variance=False, contrarian=False),
        candidates,
        config=AutoPolicyConfig(min_ev_uplift_to_override_safe=0.25),
    )

    assert output.loc[0, "final_recommended_score"] == "1-1"
    assert output.loc[0, "auto_policy_decision"] == "ev_override_accepted"
