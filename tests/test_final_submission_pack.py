"""Tests for final submission pack outputs."""

from pathlib import Path

import pandas as pd


def test_final_submission_pack_includes_72_group_stage_matches():
    path = Path("outputs/predictions/final_group_score_predictions.csv")
    assert path.exists()
    scores = pd.read_csv(path)

    assert len(scores) == 72
    assert scores["match_number"].is_unique
    assert scores["final_recommended_score"].notna().all()


def test_group_standing_predictions_include_12_groups():
    path = Path("outputs/predictions/final_group_standing_predictions.csv")
    assert path.exists()
    standings = pd.read_csv(path)

    assert len(standings) == 12
    assert set(standings["group"]) == set("ABCDEFGHIJKL")
    assert standings[["rank_1", "rank_2", "rank_3", "rank_4"]].notna().all().all()


def test_final_pack_has_populated_last8_when_bracket_mapping_validates():
    path = Path("outputs/predictions/final_last8_predictions.csv")
    assert path.exists()
    last8 = pd.read_csv(path)

    assert len(last8) == 15
    assert set(last8["stage"]) == {"quarter_finalist", "semi_finalist", "finalist", "winner"}
    assert (last8.groupby("stage").size().to_dict()) == {
        "finalist": 2,
        "quarter_finalist": 8,
        "semi_finalist": 4,
        "winner": 1,
    }
    assert last8["team"].notna().all()
