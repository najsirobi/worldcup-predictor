"""Tests for WC2026 squad overlay helpers."""

import pandas as pd

from scripts.apply_wc2026_squad_overlay import CONTEXT_COLUMNS, attach_team_features


def test_overlay_never_changes_prediction_row_count():
    predictions = pd.DataFrame(
        [
            {"match_number": 1, "team_a": "A", "team_b": "B", "model_p_a_win": 0.5},
            {"match_number": 2, "team_a": "C", "team_b": "D", "model_p_a_win": 0.4},
        ]
    )
    features = pd.DataFrame(
        [
            {"team": team, **{column: 1 for column in CONTEXT_COLUMNS}}
            for team in ["A", "B", "C", "D"]
        ]
    )

    out = attach_team_features(predictions, features)

    assert len(out) == len(predictions)
    assert out["match_number"].tolist() == [1, 2]
    assert out["team_a_squad_player_count"].notna().all()
    assert out["team_b_squad_player_count"].notna().all()
