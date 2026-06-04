"""Tests for plus model matrix joins."""

import pandas as pd

from scripts.build_model_matrix_plus import build_plus_matrix


def test_plus_model_matrix_preserves_baseline_rows_and_targets():
    baseline = pd.DataFrame(
        [
            {
                "date": "2018-06-01",
                "match_year": 2018,
                "tournament": "FIFA World Cup",
                "home_team": "A",
                "away_team": "B",
                "home_score": 1,
                "away_score": 0,
                "result_label": "home_win",
                "home_goals": 1,
                "away_goals": 0,
            },
            {
                "date": "2018-06-02",
                "match_year": 2018,
                "tournament": "Friendly",
                "home_team": "C",
                "away_team": "D",
                "home_score": 0,
                "away_score": 0,
                "result_label": "draw",
                "home_goals": 0,
                "away_goals": 0,
            },
        ]
    )
    squad = pd.DataFrame(
        [
            {"tournament_name": "2018 FIFA World Cup", "team": "A", "squad_player_count": 23, "squad_avg_age": 26, "has_squad_features": True, "has_attacker_features": False, "has_wc2026_squad_features": False},
            {"tournament_name": "2018 FIFA World Cup", "team": "B", "squad_player_count": 22, "squad_avg_age": 27, "has_squad_features": True, "has_attacker_features": False, "has_wc2026_squad_features": False},
        ]
    )
    coach = pd.DataFrame(
        [
            {"tournament_name": "2018 FIFA World Cup", "team": "A", "match_date": "2018-06-01", "coach_name": "A Coach", "coach_tenure_days": 0, "coach_matches_before_match": 0, "coach_winrate_before_match": pd.NA, "has_coach_features": True},
            {"tournament_name": "2018 FIFA World Cup", "team": "B", "match_date": "2018-06-01", "coach_name": "B Coach", "coach_tenure_days": 0, "coach_matches_before_match": 0, "coach_winrate_before_match": pd.NA, "has_coach_features": True},
        ]
    )

    plus = build_plus_matrix(baseline, squad, coach)

    assert len(plus) == len(baseline)
    for column in ["home_score", "away_score", "result_label", "home_goals", "away_goals"]:
        assert column in plus.columns
    assert plus.loc[0, "has_squad_features"]
    assert plus.loc[0, "has_coach_features"]
    assert not plus.loc[1, "has_squad_features"]
    assert "squad_player_count_diff" in plus.columns
    assert plus.loc[0, "squad_player_count_diff"] == 1
