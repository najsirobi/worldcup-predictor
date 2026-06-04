"""Tests for Last-8 recommendation selection."""

import pandas as pd

from scripts.generate_last8_recommendations import build_last8_recommendations


def test_last8_recommendations_have_required_stage_counts():
    summary = pd.DataFrame(
        {
            "team": [f"Team {idx:02d}" for idx in range(12)],
            "p_reach_qf": [1 - idx * 0.04 for idx in range(12)],
            "p_reach_sf": [1 - idx * 0.05 for idx in range(12)],
            "p_reach_final": [1 - idx * 0.06 for idx in range(12)],
            "p_win_world_cup": [1 - idx * 0.07 for idx in range(12)],
        }
    )

    rec = build_last8_recommendations(summary)

    assert len(rec["quarter_finalists"]) == 8
    assert len(rec["semi_finalists"]) == 4
    assert len(rec["finalists"]) == 2
    assert rec["winner"] == "Team 00"
    assert rec["expected_points_estimate"] > 0
