"""Tests for the country-context backtest guardrails and promotion gate (Task F)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "outputs" / "reports" / "country_context_feature_lift_results.csv"
DASHBOARD = ROOT / "outputs" / "reports" / "wc2026_country_context_dashboard_note.md"
V2_MANIFEST = ROOT / "outputs" / "final_candidate_v2_auto_science" / "FROZEN_MANIFEST.json"
V3_DIR = ROOT / "outputs" / "final_candidate_v3_country_context"


def test_v2_auto_science_frozen_files_unchanged():
    """Requirement 7: frozen v2 submission files are byte-for-byte unchanged."""
    manifest = json.loads(V2_MANIFEST.read_text())
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        assert path.exists(), f"missing frozen file {entry['path']}"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == entry["sha256"], f"frozen file changed: {entry['path']}"


def test_v2_auto_science_no_git_diff():
    """Requirement 7: nothing under the v2 candidate dir is modified in the worktree."""
    result = subprocess.run(
        ["git", "diff", "--name-only", "--", "outputs/final_candidate_v2_auto_science"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == ""


@pytest.mark.skipif(not RESULTS.exists(), reason="backtest not run yet")
def test_v3_created_only_if_promotion_passes():
    """Requirement 8: v3 exists iff the documented promotion gate passed."""
    from scripts.train_country_context_models import evaluate_gate

    res = pd.read_csv(RESULTS)
    gate = evaluate_gate(res, "cc_core_all")
    if gate["passed"]:
        assert V3_DIR.exists(), "promotion gate passed but v3 dir is missing"
    else:
        assert not V3_DIR.exists(), (
            "v3 dir exists but the country-context promotion gate failed "
            "(see country_context_policy_recommendation.md)"
        )


@pytest.mark.skipif(not RESULTS.exists(), reason="backtest not run yet")
def test_dashboard_note_present_and_context_only_when_not_promoted():
    from scripts.train_country_context_models import evaluate_gate

    res = pd.read_csv(RESULTS)
    gate = evaluate_gate(res, "cc_core_all")
    if not gate["passed"]:
        assert DASHBOARD.exists()
        text = DASHBOARD.read_text()
        assert "Context only" in text
        assert "not used in final prediction" in text.lower()
