"""Tests for auto standings derived from selected scores."""

from pathlib import Path

import pandas as pd

from scripts.build_group_standings_from_auto_scores import compute_group_standings


def test_group_standings_computed_from_scores_have_12_groups_and_4_teams_each():
    scores_path = Path("outputs/predictions/final_group_score_predictions_auto.csv")
    standings_path = Path("outputs/predictions/final_group_standing_predictions_auto.csv")
    template_path = Path("data/reference/fif8a_group_stage_template.csv")
    assert scores_path.exists()
    assert standings_path.exists()

    scores = pd.read_csv(scores_path)
    standings = pd.read_csv(standings_path)
    template = pd.read_csv(template_path)

    recomputed, _ = compute_group_standings(scores, template)

    assert len(standings) == 12
    assert set(standings["group"]) == set("ABCDEFGHIJKL")
    assert standings[["rank_1", "rank_2", "rank_3", "rank_4"]].notna().all().all()
    for _, row in standings.iterrows():
        ranked = [row["rank_1"], row["rank_2"], row["rank_3"], row["rank_4"]]
        assert len(set(ranked)) == 4
    pd.testing.assert_frame_equal(standings, recomputed)


def test_team_a_team_b_orientation_remains_correct():
    scores = pd.read_csv("outputs/predictions/final_group_score_predictions_auto.csv")
    template = pd.read_csv("data/reference/fif8a_group_stage_template.csv")

    left = scores.sort_values("match_number")[["match_number", "team_a", "team_b"]].reset_index(drop=True)
    right = template.sort_values("match_number")[["match_number", "team_a", "team_b"]].reset_index(drop=True)

    pd.testing.assert_frame_equal(left, right)
