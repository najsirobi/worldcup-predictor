"""Tests that live actual results do not rewrite frozen submitted predictions."""

from pathlib import Path

import pandas as pd

from scripts import update_live_tournament_state
from src.live.active_candidate import load_active_candidate
from src.live.prediction_scoring import ScoringRules, score_predictions_vs_actuals
from src.live.scores_override import build_initial_override, update_match, write_override
from src.live.submission_guard import sha256_file, verify_manifest


RULES = ScoringRules(base=6, gd_bonus=2, exact_bonus=3)
ODDS = {"team_a": 2.0, "draw": 3.5, "team_b": 4.0}


def _played_override(tmp_path: Path) -> Path:
    frame = build_initial_override()
    frame = update_match(frame, match_number=1, team_a_goals=2, team_b_goals=1)
    path = tmp_path / "scores_override.csv"
    write_override(frame, path)
    return path


def test_travel_mode_result_updates_live_table_not_submitted_scores(tmp_path, monkeypatch):
    candidate = load_active_candidate()
    before_hash = sha256_file(candidate.score_predictions_path)
    verify_manifest()

    scores_path = _played_override(tmp_path)
    live_dir = tmp_path / "live"
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(update_live_tournament_state, "OVERRIDE_PATH", scores_path)
    monkeypatch.setattr(update_live_tournament_state, "LIVE_DIR", live_dir)
    monkeypatch.setattr(update_live_tournament_state, "REPORTS_DIR", reports_dir)

    update_live_tournament_state.main()

    assert sha256_file(candidate.score_predictions_path) == before_hash
    verify_manifest()

    tables = pd.read_csv(live_dir / "live_group_tables.csv")
    group_a = tables[tables["group"].eq("A")].set_index("team")
    assert group_a.loc["Mexico", "points"] == 3
    assert group_a.loc["Mexico", "goals_for"] == 2
    assert group_a.loc["South Africa", "goals_against"] == 2


def test_prediction_vs_actual_uses_submitted_score_and_actual_score():
    predictions = pd.DataFrame(
        {
            "match_number": [1],
            "group": ["A"],
            "team_a": ["Mexico"],
            "team_b": ["South Africa"],
            "final_recommended_score": ["1-0"],
        }
    )
    scores = pd.DataFrame(
        {
            "match_number": [1],
            "group": ["A"],
            "team_a": ["Mexico"],
            "team_b": ["South Africa"],
            "team_a_goals": [2],
            "team_b_goals": [1],
            "status": ["played"],
        }
    )

    detail = score_predictions_vs_actuals(predictions, scores, {1: ODDS}, RULES)
    row = detail.iloc[0]

    assert row["status"] == "locked/submitted"
    assert row["submitted_score"] == "1-0"
    assert row["actual_score"] == "2-1"
    assert row["points_earned"] == row["total_points"]
    assert row["points_earned"] == 14.0
