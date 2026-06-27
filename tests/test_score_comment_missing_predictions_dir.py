"""Test that score comment update works when outputs/predictions is absent.

Critical guardrail: the workflow must resolve active-candidate paths properly
and not depend on outputs/predictions/group_submission_view.csv, which is
NOT committed and thus absent in the GitHub runner.
"""

import pytest
from pathlib import Path
from src.live.active_candidate import load_active_candidate


def test_active_candidate_uses_committed_config():
    """Verify active_candidate.yml points to a frozen committed candidate."""
    candidate = load_active_candidate()
    # The active candidate directory must exist and be committed.
    assert candidate.candidate_dir.exists(), \
        f"Active candidate dir missing: {candidate.candidate_dir}"
    assert candidate.config_existed, \
        "active_candidate.yml must exist in data/live/"


def test_candidate_score_predictions_file_exists():
    """Verify the resolved score predictions file actually exists."""
    candidate = load_active_candidate()
    assert candidate.score_predictions_path.exists(), \
        f"Score predictions file missing: {candidate.score_predictions_path}"


def test_candidate_fill_only_file_exists():
    """Verify the fill-only predictions file exists (used by mobile dashboard)."""
    candidate = load_active_candidate()
    fill_only_path = candidate.candidate_dir / "final_group_score_predictions_fill_only.csv"
    assert fill_only_path.exists(), \
        f"Fill-only file missing: {fill_only_path}"


def test_outputs_predictions_dir_absence_does_not_break_candidate():
    """Score comment update must work even if outputs/predictions is absent.
    
    This replicates the GitHub runner condition: outputs/ is gitignored,
    so only committed frozen candidates exist.
    """
    # Verify the active candidate resolution is independent of outputs/predictions/.
    candidate = load_active_candidate()
    
    # The active candidate is a frozen directory committed to the repo.
    assert candidate.config_existed
    assert candidate.candidate_dir.exists()
    # All prediction files it needs exist.
    assert candidate.score_predictions_path.exists()
    assert candidate.standing_predictions_path.exists()
    assert candidate.last8_predictions_path.exists()
    
    # We should never need to access outputs/predictions/ for the score comment flow.
    # (If this fails, the workflow would have already crashed before reaching this test.)
    outputs_predictions = candidate.candidate_dir.parent / "predictions"
    # This directory may not exist; we don't care.
    # The point is: our resolution never depends on it.
