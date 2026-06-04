"""Tests for frozen auto-science final candidate artifacts."""

from pathlib import Path

import pandas as pd


REQUIRED_FILES = {
    "final_group_score_predictions_auto.csv",
    "final_group_standing_predictions_auto.csv",
    "final_last8_predictions_auto.csv",
    "final_submission_pack_auto.csv",
    "auto_score_candidates.csv",
    "final_group_score_auto_policy.md",
    "final_group_standing_auto_report.md",
    "auto_consensus_seed_stability_report.md",
    "README.md",
}


def test_final_candidate_v2_auto_science_contains_all_required_files():
    candidate_dir = Path("outputs/final_candidate_v2_auto_science")
    assert candidate_dir.exists()

    present = {path.name for path in candidate_dir.iterdir() if path.is_file()}

    assert REQUIRED_FILES <= present


def test_auto_candidate_pack_has_expected_sections_and_last8_unchanged():
    pack = pd.read_csv("outputs/predictions/final_submission_pack_auto.csv")
    last8_auto = pd.read_csv("outputs/predictions/final_last8_predictions_auto.csv")
    last8_v1 = pd.read_csv("outputs/final_candidate_v1/final_last8_predictions.csv")

    assert (pack["section"] == "group_score").sum() == 72
    assert (pack["section"] == "group_standing").sum() == 12
    assert (pack["section"] == "last8").sum() == 15
    pd.testing.assert_frame_equal(last8_auto, last8_v1)


def test_auto_final_scores_resolve_manual_review_flags():
    scores = pd.read_csv("outputs/predictions/final_group_score_predictions_auto.csv")

    assert scores["manual_review_flag_original"].sum() >= 1
    assert scores["manual_review_resolved_auto"].all()
