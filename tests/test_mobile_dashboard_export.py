"""Tests for the mobile dashboard export and workflow guardrails (Tasks H/E/K)."""

import json
from pathlib import Path

from scripts import build_mobile_dashboard

SCRIPT_DIR = Path("scripts")
LIVE_DIR = Path("outputs/live")

# Travel Mode scripts that the GitHub Actions workflows run.
WORKFLOW_SCRIPTS = [
    "init_scores_override.py",
    "update_score_override.py",
    "apply_scores_batch_update.py",
    "apply_score_comment.py",
    "update_live_tournament_state.py",
    "recalculate_live_simulations.py",
    "score_predictions_vs_actuals.py",
    "build_mobile_dashboard.py",
]

# Tokens that would indicate (re)training or network/API access sneaking in.
FORBIDDEN_TOKENS = [
    "train_",
    ".fit(",
    "requests.get",
    "requests.post",
    "urllib",
    "kagglehub",
    "BeautifulSoup",
]


def test_render_html_produces_self_contained_page():
    payload = build_mobile_dashboard.build_payload()
    html = build_mobile_dashboard.render_html(payload)
    assert html.lstrip().startswith("<!DOCTYPE html>")
    assert "</html>" in html
    # Data embedded inline (works offline, no server fetch needed).
    assert "application/json" in html
    # Key components are present.
    assert 'id="app"' in html
    assert "function render" in html
    # All required functionality markers are present.
    for marker in [
        "Overview",
        "Submit Scores",
        "Scores to fill in",
        "Group Standings",
        "Last-8",
        "Live Results",
        "Prediction vs Actual",
        "Score input instructions",  # 📱 Score input instructions
        "Played",  # 📋 Played
        "Points",  # 🏅 Points
        "Live group tables",  # 📊 Live group tables
        "Remaining",  # ⏭️ Remaining
    ]:
        assert marker in html, f"missing marker: {marker}"


def test_dashboard_includes_prediction_vs_actual_section():
    payload = build_mobile_dashboard.build_payload()
    assert "prediction_vs_actual" in payload
    assert "scoring_summary" in payload
    assert "submission_score_predictions" in payload
    assert "submission_summary" in payload
    assert payload["submission_summary"]["manual_review_rows_auto_resolved"] == 21
    assert payload["submission_summary"]["ev_overrides_accepted"] == 0
    assert payload["submission_summary"]["ev_overrides_rejected"] == 27
    assert payload["submission_summary"]["safe_scores_kept"] == 72
    assert len(payload["submission_score_predictions"]) == 72
    html = build_mobile_dashboard.render_html(payload)
    assert "Prediction vs Actual" in html


def test_dashboard_displays_active_candidate_name():
    payload = build_mobile_dashboard.build_payload()
    assert payload["active_candidate"]["name"] == "final_candidate_v2_auto_science"
    html = build_mobile_dashboard.render_html(payload)
    # The candidate name is embedded (in the payload JSON and rendered into the header).
    assert "final_candidate_v2_auto_science" in html
    assert "Active candidate" in html


def test_build_main_creates_html_and_json():
    build_mobile_dashboard.main()
    live_html = LIVE_DIR / "mobile_dashboard.html"
    data_path = LIVE_DIR / "mobile_dashboard_data.json"
    docs_html = Path("docs/index.html")
    docs_json = Path("docs/mobile_dashboard_data.json")
    assert live_html.exists()
    assert data_path.exists()
    assert docs_html.exists()
    assert docs_json.exists()
    data = json.loads(data_path.read_text())
    assert "generated_at" in data
    assert "advancement" in data
    assert "final_group_standings" in data
    assert "active_candidate" in data
    assert "scoring_summary" in data
    assert json.loads(docs_json.read_text()) == data
    html = live_html.read_text()
    assert "Scores to fill in" in html
    assert "Group standings to fill in" in html
    assert "Last-8 / progression picks to fill in" in html
    assert "No matches played yet" in html
    assert "No scoring summary yet" in html
    assert "Loaded keys" in html
    assert "manual decision required" not in html.lower()
    assert "manual review required" not in html.lower()
    assert "try {" in html


def test_workflow_scripts_do_not_call_training_or_apis():
    for name in WORKFLOW_SCRIPTS:
        source = (SCRIPT_DIR / name).read_text()
        for token in FORBIDDEN_TOKENS:
            assert token not in source, f"{name} contains forbidden token {token!r}"


def test_live_modules_do_not_call_training_or_apis():
    for path in Path("src/live").glob("*.py"):
        source = path.read_text()
        for token in FORBIDDEN_TOKENS:
            assert token not in source, f"{path} contains forbidden token {token!r}"


def test_workflow_yaml_has_no_training_steps():
    for wf_name in ("travel_mode_update.yml", "score_comment_update.yml"):
        text = (Path(".github/workflows") / wf_name).read_text()
        assert "train_" not in text, f"{wf_name} references training"
        assert "fetch_" not in text, f"{wf_name} references a fetch script"
    # The single-match workflow is still dispatch-triggered.
    assert "workflow_dispatch" in (Path(".github/workflows/travel_mode_update.yml")).read_text()
    # The comment workflow triggers on issue comments.
    assert "issue_comment" in (Path(".github/workflows/score_comment_update.yml")).read_text()
