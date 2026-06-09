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
    "build_knockout_predictions.py",
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
    assert 'data-theme="light"' in html
    assert "Light / Dark" in html
    # All required functionality markers are present.
    for marker in [
        "Overview",
        "Submitted Scores",
        "Submitted prediction",
        "Group Standings",
        "Last-8",
        "Live Results",
        "Prediction vs Actual",
        "Score input instructions",  # 📱 Score input instructions
        "Played",  # 📋 Played
        "Actual result",
        "Points earned",
        "Live group table",
        "Projected future bracket",
        "Future recommendation",
        "Remaining",  # ⏭️ Remaining
    ]:
        assert marker in html, f"missing marker: {marker}"


def test_dashboard_presents_locked_submitted_score_per_match():
    payload = build_mobile_dashboard.build_payload()
    html = build_mobile_dashboard.render_html(payload)
    # The frozen submitted score is clearly labelled.
    assert "Submitted prediction" in html
    assert "Actual result" in html
    assert "Points earned" in html
    assert "locked/submitted" in html
    # Copy-friendly line format is present verbatim.
    assert "1. Mexico 1-0 South Africa" in html
    # Alternatives are labelled as secondary/audit, not as final picks.
    assert "Safe alternative" in html
    assert "EV alternative" in html
    assert "Consensus/modal score" in html
    # Audit alternatives live behind a collapsible "Why? / Audit details" block.
    assert "Why? / Audit details" in html
    assert 'class="audit-details"' in html
    # The bare "safe:" / "EV:" inline pills (old equal-pick presentation) are gone.
    assert ">safe: " not in html
    assert ">EV: " not in html
    # Copy blocks for all scores and per group are present.
    assert "All submitted predictions" in html
    assert "Per-group copy blocks" in html
    # Banned post-submission wording.
    assert "updated prediction" not in html.lower()
    assert "new group-stage pick" not in html.lower()
    assert "actual results update submitted picks" not in html.lower()
    # No manual-review action text.
    assert "manual review required" not in html.lower()
    assert "manual decision required" not in html.lower()


def test_dashboard_fill_in_sections_and_alignment():
    """Practical submission dashboard: fill-in sections, copy lines, aligned tables."""
    import pandas as pd

    payload = build_mobile_dashboard.build_payload()
    html = build_mobile_dashboard.render_html(payload)

    # Active candidate.
    assert payload["active_candidate"]["name"] == "final_candidate_v4_recent_rollforward"
    # Fill-in section headings (practical, not audit-first).
    assert "Scores to fill in" in html
    assert "Group standings to fill in" in html
    assert "Last-8 to fill in" in html
    # Copy-friendly line format present verbatim (unadjusted match).
    assert "1. Mexico 1-0 South Africa" in html
    # The active score-to-fill-in source is the active candidate's fill-only file.
    # v4 is now active (rolling-forward with June 3-8 form update, R1 rule applied).
    fill_path = (
        "outputs/final_candidate_v4_recent_rollforward/"
        "final_group_score_predictions_fill_only.csv"
    )
    fill_only = pd.read_csv(fill_path)
    assert len(fill_only) == 72
    for line in fill_only["copy_text"].astype(str):
        assert line in html, f"missing fill-only line: {line}"
    # Aligned table classes are used so columns line up on mobile/desktop.
    assert 'class="aligned"' in html
    assert 'class="aligned standings-table"' in html
    # Audit/policy clutter is not a main visible section.
    assert "Science-only policy notes" not in html
    assert "manual review required" not in html.lower()
    assert "manual decision required" not in html.lower()
    # Strict separation wording from the frozen-guard work is preserved.
    assert "Submitted prediction" in html
    assert "Actual result" in html
    # Round-by-round knockout exact-score predictions are present.
    assert "Knockout predictions" in html
    assert "All knockout predictions" in html
    assert "Round of 32" in html
    assert "Compare gambles" in html
    knockout = payload["knockout_predictions"]
    assert len(knockout["matches"]) == 32
    # "Next round to predict" shows the full next unresolved round.
    assert "Next round to predict" in html
    assert "next match to predict" not in html.lower()
    assert "Projected matchup" in html
    assert "Current recommendation" in html
    assert knockout["next_round"] == "R32"
    assert len(knockout["next_round_matches"]) == 16  # full round, multiple matches
    # Copy-friendly next-round lines use the "— adv …" format.
    assert "— adv" in html
    assert " — adv " in knockout["next_round_matches"][0]["copy_text"]
    # JSON payload remains strict-valid JSON.
    json.loads(build_mobile_dashboard._payload_json(payload))


def test_dashboard_is_server_side_prerendered_without_js():
    """The page must show real content even if the client JavaScript never runs."""
    payload = build_mobile_dashboard.build_payload()
    html = build_mobile_dashboard.render_html(payload)

    # No permanent "Loading" placeholder remains after build.
    assert "Loading" not in html
    # #app is server-rendered, not an empty shell filled only by JS.
    assert "<main id=\"app\"><!--APP_PLACEHOLDER-->" not in html
    assert "<main id=\"app\"><details" in html

    # Every key section exists as a real collapsible <details> element (not JS-only).
    for section_id in [
        "overview",
        "submitted-scores",
        "group-standings",
        "last8",
        "knockout-predictions",
        "live-results",
        "prediction-vs-actual",
        "live-group-tables",
        "advancement",
    ]:
        assert f'<details class="dashboard-section" id="{section_id}"' in html, section_id

    # The static body (before the inline JSON payload) carries the real content,
    # so the strings exist directly in the HTML and not only inside the payload.
    body = html.split('<script id="payload"', 1)[0]
    for marker in [
        "Scores to fill in",
        "1. Mexico 1-0 South Africa",
        "Group standings to fill in",
        "Last-8 to fill in",
        "Next round to predict",
        "Live group tables",
    ]:
        assert marker in body, f"missing from server-rendered body: {marker}"

    # JS remains as progressive enhancement with non-destructive error handling.
    assert "Loaded keys" in html  # error banner is appended, never blanks #app
    assert "insertAdjacentHTML" in html


def test_dashboard_collapsible_sections_and_theme_contract():
    payload = build_mobile_dashboard.build_payload()
    html = build_mobile_dashboard.render_html(payload)

    assert 'data-theme="light"' in html
    assert 'data-theme-choice="light"' in html
    assert 'data-theme-choice="dark"' in html
    assert "localStorage" in html
    assert "travelModeTheme" in html
    assert "--background:" in html
    assert "--card-background:" in html
    assert "--text:" in html
    assert "--muted-text:" in html
    assert "--border:" in html
    assert "--accent:" in html
    assert "--success:" in html
    assert "--warning:" in html
    assert "--danger:" in html

    for section_id in [
        "overview",
        "submitted-scores",
        "group-standings",
        "last8",
        "knockout-predictions",
        "live-results",
        "prediction-vs-actual",
        "live-group-tables",
        "advancement",
    ]:
        assert f'<details class="dashboard-section" id="{section_id}"' in html
        assert f'<details class="dashboard-section" id="{section_id}"' in html.split('<script id="payload"', 1)[0]

    for section_id in ["overview", "submitted-scores", "knockout-predictions"]:
        assert f'<details class="dashboard-section" id="{section_id}" open>' in html

    for section_id in [
        "group-standings",
        "last8",
        "live-results",
        "prediction-vs-actual",
        "live-group-tables",
        "advancement",
    ]:
        assert f'<details class="dashboard-section" id="{section_id}">' in html


def test_dashboard_includes_prediction_vs_actual_section():
    payload = build_mobile_dashboard.build_payload()
    assert "prediction_vs_actual" in payload
    assert "scoring_summary" in payload
    assert "submission_score_predictions" in payload
    assert "submission_summary" in payload
    assert payload["submission_summary"]["manual_review_rows_auto_resolved"] == 21
    assert payload["submission_summary"]["ev_overrides_accepted"] == 0
    assert payload["submission_summary"]["ev_overrides_rejected"] == 26
    assert payload["submission_summary"]["safe_scores_kept"] == 68
    assert len(payload["submission_score_predictions"]) == 72
    first = payload["submission_score_predictions"][0]
    assert first["status"] == "locked/submitted"
    assert first["submitted_score"] == "1-0"
    assert "actual_score" in first
    assert "points_earned" in first
    html = build_mobile_dashboard.render_html(payload)
    assert "Prediction vs Actual" in html


def test_dashboard_displays_active_candidate_name():
    payload = build_mobile_dashboard.build_payload()
    assert payload["active_candidate"]["name"] == "final_candidate_v4_recent_rollforward"
    html = build_mobile_dashboard.render_html(payload)
    # The candidate name is embedded (in the payload JSON and rendered into the header).
    assert "final_candidate_v4_recent_rollforward" in html
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
    docs_html_text = docs_html.read_text()
    assert "Submitted prediction" in html
    assert "Actual result" in html
    assert "Points earned" in html
    assert "Projected future bracket" in html
    assert "No matches played yet" in html
    assert "No scoring summary yet" in html
    assert "Loaded keys" in html
    assert "manual decision required" not in html.lower()
    assert "manual review required" not in html.lower()
    assert "try {" in html
    assert "Loading" not in docs_html_text
    assert "Light / Dark" in docs_html_text
    assert 'data-theme="light"' in docs_html_text
    assert '<details class="dashboard-section" id="overview" open>' in docs_html_text
    assert '<details class="dashboard-section" id="group-standings">' in docs_html_text
    assert "1. Mexico 1-0 South Africa" in docs_html_text


def test_dashboard_shows_knockout_human_overlay_review():
    payload = build_mobile_dashboard.build_payload()
    html = build_mobile_dashboard.render_html(payload)
    # The dedicated knockout overlay subsection is present.
    assert "Knockout human-overlay review" in html
    # The human-upside analyst section exists and is collapsed by default.
    assert '<details class="dashboard-section" id="human-upside-overlay">' in html
    assert (
        build_mobile_dashboard.SECTION_DEFAULT_OPEN.get("human-upside-overlay") is False
    )


def test_dashboard_labels_human_overlay_not_used_in_final():
    payload = build_mobile_dashboard.build_payload()
    html = build_mobile_dashboard.render_html(payload)
    assert "Context only — not used in final prediction." in html
    assert "Objective residual review" in html
    assert "manually approved" not in html.lower()
    assert "manual approval" not in html.lower()
    assert "Broad human-overlay suggestions (audit-only, hidden by default)" in html


def test_human_overlay_does_not_modify_frozen_candidate():
    import hashlib
    from src.overlay.human_upside_overlay import V2_SCORES_PATH

    before = hashlib.sha256(V2_SCORES_PATH.read_bytes()).hexdigest()
    build_mobile_dashboard.build_payload()
    build_mobile_dashboard.render_html(build_mobile_dashboard.build_payload())
    assert hashlib.sha256(V2_SCORES_PATH.read_bytes()).hexdigest() == before


def test_dashboard_uses_objective_residual_score_when_promoted():
    """When v3 promotion passes, the score-to-fill-in is the adjusted score (Part D)."""
    payload = build_mobile_dashboard.build_payload()
    objective = payload["objective_residual"]
    if not objective["promotion_gate_passed"]:
        return  # gate failed -> v2 stays active; covered by other tests
    rows = {r["match_number"]: r for r in payload["submission_score_predictions"]}
    adjusted = {int(a["match_number"]): a for a in objective["adjustments"]}
    assert adjusted, "promotion passed but no adjusted matches recorded"
    for mn, adj in adjusted.items():
        row = rows[mn]
        assert row["objective_residual_applied"] is True
        assert row["submitted_score"] == adj["objective_residual_score"]
        assert row["base_v2_score"] == adj["v2_score"]
        assert row["objective_rule_triggered"] == "R1_only_diff_5_0"
    # Unadjusted matches keep their v2 score and are not flagged.
    for mn, row in rows.items():
        if mn not in adjusted:
            assert row["objective_residual_applied"] is False
            assert row["submitted_score"] == row["base_v2_score"]


def test_dashboard_labels_base_and_objective_residual_candidate():
    """Dashboard labels the v2 baseline and the objective-residual candidate (Part D)."""
    payload = build_mobile_dashboard.build_payload()
    html = build_mobile_dashboard.render_html(payload)
    # Active candidate is now v4 (rolling-forward with June 3-8 form update).
    assert payload["active_candidate"]["name"] == "final_candidate_v4_recent_rollforward"
    assert "Base model: v2_auto_science" in html
    assert "Adjusted candidate: final_candidate_v3_objective_residual" in html
    assert "Post-model objective residual rule applied:" in html
    # The v2 candidate files are never mutated by the dashboard build.
    assert payload["objective_residual"]["base_model_byte_identical"] is True


def test_dashboard_does_not_use_manual_approval_language():
    """Part D / Part F: no 'manual approval' framing anywhere in the dashboard."""
    payload = build_mobile_dashboard.build_payload()
    html = build_mobile_dashboard.render_html(payload)
    assert "manual approval" not in html.lower()
    assert "manually approved" not in html.lower()
    # The deterministic-only statement is present instead.
    assert "No manual sign-off or subjective override used." in html


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


def test_workflows_run_knockout_prediction_build():
    # Both Travel Mode workflows must build knockout predictions as part of the
    # refresh, after scoring and before the dashboard build.
    for wf_name in ("travel_mode_update.yml", "score_comment_update.yml"):
        text = (Path(".github/workflows") / wf_name).read_text()
        assert "scripts/build_knockout_predictions.py" in text, wf_name
        ko = text.index("scripts/build_knockout_predictions.py")
        dash = text.index("scripts/build_mobile_dashboard.py")
        assert ko < dash, f"{wf_name}: knockout build must run before the dashboard"
        # The dashboard's knockout artifacts are published to docs/.
        assert "docs/knockout_predictions.csv" in text, wf_name


def test_workflow_yaml_has_no_training_steps():
    for wf_name in ("travel_mode_update.yml", "score_comment_update.yml"):
        text = (Path(".github/workflows") / wf_name).read_text()
        assert "train_" not in text, f"{wf_name} references training"
        assert "fetch_" not in text, f"{wf_name} references a fetch script"
    # The single-match workflow is still dispatch-triggered.
    assert "workflow_dispatch" in (Path(".github/workflows/travel_mode_update.yml")).read_text()
    # The comment workflow triggers on issue comments.
    assert "issue_comment" in (Path(".github/workflows/score_comment_update.yml")).read_text()
