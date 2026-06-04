"""Tests for deterministic auto-consensus score policy."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evaluation.auto_consensus import (
    AutoPolicyConfig,
    collect_candidate_scores,
    select_final_scores,
)


def _base_prediction(safe: str = "1-0", ev: str = "1-1", ep_safe: float = 7.0, ep_ev: float = 7.4) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_number": 1,
                "group": "A",
                "team_a": "Team A",
                "team_b": "Team B",
                "model_p_a_win": 0.5,
                "model_p_draw": 0.3,
                "model_p_b_win": 0.2,
                "recommended_score_safe": safe,
                "recommended_score_ev": ev,
                "most_probable_score": safe,
                "ev_max_score": ev,
                "expected_points_safe": ep_safe,
                "expected_points_ev": ep_ev,
                "notes": "",
            }
        ]
    )


def _decision(high_variance: bool = False, contrarian: bool = False, uplift: float = 0.4) -> pd.DataFrame:
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


def _final_v1(manual: bool = True) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_number": 1,
                "manual_review_flag": manual,
            }
        ]
    )


def test_auto_policy_resolves_all_manual_review_rows():
    pred = _base_prediction()
    candidates, _ = collect_candidate_scores(pred, _decision(), None)

    out = select_final_scores(pred, _final_v1(True), _decision(), candidates)

    assert bool(out.loc[0, "manual_review_flag_original"]) is True
    assert bool(out.loc[0, "manual_review_resolved_auto"]) is True


def test_ev_score_only_overrides_safe_when_threshold_and_variance_rules_pass():
    pred = _base_prediction(safe="1-0", ev="1-1", ep_safe=7.0, ep_ev=7.5)
    candidates = pd.DataFrame(
        [
            {"match_number": 1, "candidate_score": "1-1", "expected_points": 7.5},
            {"match_number": 1, "candidate_score": "1-1", "expected_points": 7.5},
            {"match_number": 1, "candidate_score": "1-0", "expected_points": 7.0},
        ]
    )

    allowed = select_final_scores(pred, _final_v1(), _decision(False, False, 0.5), candidates)
    blocked = select_final_scores(pred, _final_v1(), _decision(True, False, 0.5), candidates)
    low_uplift = select_final_scores(pred, _final_v1(), _decision(False, False, 0.1), candidates)

    assert allowed.loc[0, "final_recommended_score"] == "1-1"
    assert allowed.loc[0, "auto_policy_decision"] == "ev_override_accepted"
    assert blocked.loc[0, "final_recommended_score"] == "1-0"
    assert low_uplift.loc[0, "final_recommended_score"] == "1-0"


def test_tie_falls_back_to_highest_expected_points():
    pred = _base_prediction(safe="1-0", ev="3-0", ep_safe=7.0, ep_ev=7.1)
    decisions = _decision(False, False, 0.1)
    candidates = pd.DataFrame(
        [
            {"match_number": 1, "candidate_score": "1-0", "expected_points": 7.0},
            {"match_number": 1, "candidate_score": "2-0", "expected_points": 7.1},
        ]
    )

    out = select_final_scores(pred, _final_v1(), decisions, candidates)

    assert out.loc[0, "auto_consensus_score"] == "2-0"
    assert out.loc[0, "final_recommended_score"] == "2-0"


def test_final_fallback_is_safe_score():
    pred = _base_prediction(safe="1-0", ev="3-0", ep_safe=7.0, ep_ev=7.0)
    decisions = _decision(False, False, 0.0)
    candidates = pd.DataFrame(
        [
            {"match_number": 1, "candidate_score": "1-0", "expected_points": 7.0},
            {"match_number": 1, "candidate_score": "2-0", "expected_points": 7.0},
        ]
    )

    out = select_final_scores(pred, _final_v1(), decisions, candidates, config=AutoPolicyConfig())

    assert out.loc[0, "final_recommended_score"] == "1-0"
    assert "safe_score" in out.loc[0, "reason"]


def test_auto_score_output_has_72_rows_when_generated():
    path = Path("outputs/predictions/final_group_score_predictions_auto.csv")
    assert path.exists()
    scores = pd.read_csv(path)

    assert len(scores) == 72
    assert scores["final_recommended_score"].notna().all()
