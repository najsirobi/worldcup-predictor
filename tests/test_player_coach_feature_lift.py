"""Tests for Phase 5 promotion guardrails."""

from pathlib import Path


def test_feature_lift_report_keeps_baseline_when_wc2026_coverage_missing():
    report = Path("outputs/reports/player_coach_feature_lift_report.md")
    if not report.exists():
        return

    text = report.read_text()
    assert "Do not promote plus features" in text
    assert "WC2026 squad coverage is 0/48 teams" in text
