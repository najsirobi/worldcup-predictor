"""Tests for coach feature leakage controls."""

import pandas as pd

from src.features.coach_features import build_coach_match_features


def test_coach_tenure_and_winrate_use_only_previous_matches():
    appearances = pd.DataFrame(
        [
            {
                "tournament_name": "T",
                "team_name": "A",
                "match_date": "2022-01-01",
                "match_id": "m1",
                "coach_name": "Coach One",
                "goals_for": 2,
                "goals_against": 0,
                "is_win": True,
            },
            {
                "tournament_name": "T",
                "team_name": "A",
                "match_date": "2022-01-10",
                "match_id": "m2",
                "coach_name": "Coach One",
                "goals_for": 0,
                "goals_against": 1,
                "is_win": False,
            },
        ]
    )

    out = build_coach_match_features(appearances).sort_values("match_date")

    first = out.iloc[0]
    second = out.iloc[1]
    assert first["coach_matches_before_match"] == 0
    assert pd.isna(first["coach_winrate_before_match"])
    assert second["coach_matches_before_match"] == 1
    assert second["coach_winrate_before_match"] == 1.0
    assert second["coach_goal_diff_per_match_before_match"] == 2.0
    assert second["coach_tenure_days"] == 9


def test_recent_coach_change_flag_uses_previous_team_match():
    appearances = pd.DataFrame(
        [
            {"tournament_name": "T", "team_name": "A", "match_date": "2022-01-01", "match_id": "m1", "coach_name": "Old", "goals_for": 0, "goals_against": 0, "is_win": False},
            {"tournament_name": "T", "team_name": "A", "match_date": "2022-01-05", "match_id": "m2", "coach_name": "New", "goals_for": 1, "goals_against": 0, "is_win": True},
        ]
    )

    out = build_coach_match_features(appearances).sort_values("match_date")

    assert not out.iloc[0]["recent_coach_change_flag"]
    assert out.iloc[1]["recent_coach_change_flag"]
