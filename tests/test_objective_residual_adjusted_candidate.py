"""Tests for the deterministic objective-residual adjusted candidate (v3).

The v3 candidate is a deterministic post-model layer over
``final_candidate_v2_auto_science`` driven solely by the WC2022-validated rule
``R1_only_diff_5_0``. It must never mutate the frozen v2 submission, must be sparse,
must change scores by at most one goal, and must use no chemistry-only / key-absence-
only / fame-only / mid-table / manual-approval logic.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import build_objective_residual_candidate as build

ROOT = Path(__file__).resolve().parents[1]
V2_DIR = ROOT / "outputs" / "final_candidate_v2_auto_science"
V3_DIR = ROOT / "outputs" / "final_candidate_v3_objective_residual"
OVERLAY_CSV = ROOT / "data" / "reference" / "wc2026_human_upside_overlay.csv"

STRONG = {"elite_upside", "positive"}
WEAK = {"fragile", "low_upside"}


@pytest.fixture(scope="module")
def result():
    return build.main()


@pytest.fixture(scope="module")
def adjustments(result):
    return result["adjustments"]


def _parse(score: str) -> tuple[int, int]:
    a, b = str(score).split("-")
    return int(a), int(b)


def test_v2_auto_science_files_unchanged(result):
    """1. The frozen v2 submission files are byte-identical to their manifest."""
    assert result["v2_unchanged"] is True
    manifest = json.loads((V2_DIR / "FROZEN_MANIFEST.json").read_text())
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == entry["sha256"], entry["path"]


def test_v3_candidate_generated_separately(result):
    """2. v3 is a separate candidate directory with its own artifacts."""
    assert V3_DIR.is_dir()
    assert V3_DIR != V2_DIR
    for name in [
        "final_group_score_predictions_auto.csv",
        "final_group_score_predictions_fill_only.csv",
        "objective_residual_adjustments.csv",
        "objective_residual_adjustment_report.md",
        "FROZEN_MANIFEST.json",
    ]:
        assert (V3_DIR / name).exists(), name


def test_adjusted_candidate_has_72_group_matches(adjustments):
    """3. The adjusted candidate covers all 72 group matches."""
    assert len(adjustments) == 72
    auto = pd.read_csv(V3_DIR / "final_group_score_predictions_auto.csv")
    fill = pd.read_csv(V3_DIR / "final_group_score_predictions_fill_only.csv")
    assert len(auto) == 72
    assert len(fill) == 72


def test_changed_count_is_sparse(adjustments, result):
    """4. Changed scores <= 8 (warn band), and the gate fails only above 16."""
    n_changed = int(adjustments["changed"].sum())
    assert n_changed <= build.FAIL_CHANGES
    if n_changed > build.WARN_CHANGES:
        assert result["gate"]["warning"] is True
    # Current expected sparse set.
    assert n_changed == 4


def test_every_change_is_max_one_goal(adjustments):
    """5. Every changed score differs from v2 by at most one goal."""
    changed = adjustments[adjustments["changed"]]
    for _, r in changed.iterrows():
        ga, gb = _parse(r["v2_score"])
        na, nb = _parse(r["objective_residual_score"])
        assert abs(ga - na) + abs(gb - nb) <= 1, r["match_number"]


def test_every_change_triggered_by_r1(adjustments):
    """6. Every changed score is triggered by R1_only_diff_5_0."""
    changed = adjustments[adjustments["changed"]]
    assert (changed["rule_triggered"] == "R1_only_diff_5_0").all()
    assert (changed["source_backtest_rule"].str.contains("R1_only_diff_5_0")).all()
    assert (changed["deterministic_yes_no"] == "yes").all()


def test_no_midtable_vs_midtable_adjustment(adjustments):
    """7. No mid-table (useful_context/high_variance) opponent is adjusted."""
    changed = adjustments[adjustments["changed"]]
    for _, r in changed.iterrows():
        strong_cat = r["team_a_category"] if r["overlay_diff"] > 0 else r["team_b_category"]
        weak_cat = r["team_b_category"] if r["overlay_diff"] > 0 else r["team_a_category"]
        assert strong_cat in STRONG, r["match_number"]
        assert weak_cat in WEAK, r["match_number"]
        assert "useful_context" not in {strong_cat, weak_cat}
        assert "high_variance" not in {strong_cat, weak_cat}


def test_no_chemistry_only_adjustment(adjustments):
    """8. No change relies on a chemistry-only component."""
    changed = adjustments[adjustments["changed"]]
    for comp in changed["rule_components"].astype(str):
        assert "chemistry" not in comp.lower()


def test_no_key_absence_only_adjustment(adjustments):
    """9. No change relies on a key-absence-only component."""
    changed = adjustments[adjustments["changed"]]
    for comp in changed["rule_components"].astype(str):
        assert "absence" not in comp.lower()
        assert "absent" not in comp.lower()


def test_no_fame_only_adjustment(adjustments):
    """10. No change relies on a fame/star-only component."""
    changed = adjustments[adjustments["changed"]]
    for comp in changed["rule_components"].astype(str):
        assert "fame" not in comp.lower()
        assert "star" not in comp.lower()
    # Each change is grounded in overlay diff + category gate, not fame.
    for _, r in changed.iterrows():
        assert abs(r["overlay_diff"]) >= build.DIFF_THRESHOLD


def test_expected_adjusted_matches(adjustments):
    """The four current qualifying matches are adjusted as specified."""
    changed = adjustments[adjustments["changed"]]
    got = {
        int(r["match_number"]): (r["v2_score"], r["objective_residual_score"])
        for _, r in changed.iterrows()
    }
    assert got == {
        21: ("1-0", "2-0"),  # England vs Croatia
        23: ("1-0", "2-0"),  # Portugal vs Congo DR
        50: ("1-0", "2-0"),  # Morocco vs Haiti
        64: ("0-1", "1-1"),  # Egypt vs IR Iran (narrow fragile win -> draw)
    }


def test_fill_only_uses_objective_residual_score(adjustments):
    """11. The v3 fill-only export carries the objective-residual adjusted scores."""
    fill = pd.read_csv(V3_DIR / "final_group_score_predictions_fill_only.csv")
    fill_map = dict(zip(fill["match_number"], fill["score_to_fill_in"].astype(str)))
    for _, r in adjustments.iterrows():
        assert fill_map[int(r["match_number"])] == str(r["objective_residual_score"])


def test_promotion_gate_passes(result):
    """The promotion gate passes for the current sparse, deterministic change set."""
    gate = result["gate"]
    assert gate["passed"] is True
    assert gate["checks"]["v2_byte_identical"] is True
    assert gate["checks"]["every_change_one_goal"] is True
    assert gate["checks"]["deterministic_from_R1"] is True
    manifest = json.loads((V3_DIR / "FROZEN_MANIFEST.json").read_text())
    assert manifest["promotion_gate_passed"] is True
    assert manifest["manual_approval_used"] is False
    assert manifest["is_recommended_score_fill_candidate"] is True


def test_no_submitted_predictions_overwritten(result):
    """18. The build does not overwrite the frozen v2 submission files."""
    assert result["v2_unchanged"] is True


# --- Part E: late-news residual audit -------------------------------------------------
LATE_NEWS_CSV = ROOT / "outputs" / "predictions" / "wc2026_late_news_residual_audit.csv"
LATE_NEWS_MD = ROOT / "outputs" / "reports" / "wc2026_late_news_residual_audit.md"
FRIENDLY_POLICY_MD = ROOT / "outputs" / "reports" / "friendly_lineup_distortion_policy.md"


@pytest.fixture(scope="module")
def late_news(result):
    return pd.read_csv(LATE_NEWS_CSV)


def test_karl_late_news_row_exists(late_news):
    """14. A Germany/Karl late-news audit row exists."""
    row = late_news[(late_news["team"] == "Germany") & (late_news["player"] == "Lennart Karl")]
    assert len(row) == 1
    r = row.iloc[0]
    assert r["case_type"] == "late_squad_absence"
    assert float(r["recommended_residual"]) == -0.25
    assert float(r["max_penalty"]) == -0.25
    assert bool(r["auto_changes_score"]) is False


def test_messi_late_fitness_row_exists_not_key_absence(late_news):
    """15/16. Messi late-fitness row exists, is not a key absence, default residual -0.25."""
    row = late_news[(late_news["team"] == "Argentina") & (late_news["player"] == "Lionel Messi")]
    assert len(row) == 1
    r = row.iloc[0]
    assert r["case_type"] == "in_squad_fitness_risk"
    assert bool(r["is_key_absence"]) is False  # 15
    assert float(r["recommended_residual"]) == -0.25  # 16
    assert float(r["max_penalty"]) == -0.50  # escalation cap only


def test_friendly_lineup_distortion_policy_exists():
    """17. The friendly-lineup-distortion policy exists and defines the tag."""
    assert FRIENDLY_POLICY_MD.exists()
    text = FRIENDLY_POLICY_MD.read_text().lower()
    assert "lineup" in text
    assert "distort" in text
    assert "friendly" in text


def test_late_news_does_not_change_scores(adjustments):
    """Germany and Argentina scores are unchanged: no objective rule qualifies for them."""
    changed = adjustments[adjustments["changed"]]
    teams_changed = set(changed["team_a"]).union(set(changed["team_b"]))
    assert "Germany" not in teams_changed
    assert "Argentina" not in teams_changed


def test_policy_and_promotion_reports_exist():
    assert (ROOT / "outputs" / "reports" / "objective_residual_adjustment_policy.md").exists()
    assert (
        ROOT / "outputs" / "reports" / "objective_residual_candidate_promotion_report.md"
    ).exists()
    assert LATE_NEWS_MD.exists()
