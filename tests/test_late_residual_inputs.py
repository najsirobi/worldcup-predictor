"""Tests for the late-residual reference data scaffolding (Tasks A–E).

These tests verify:
1. All four reference files exist with required columns.
2. The validation script runs and passes.
3. Messi appears as active_monitoring / in_squad_fitness_risk (not key absence).
4. Karl appears as ruled_out / late squad absence.
5. All penalties are within allowed bounds.
6. No row sets applies_to_score_candidate=true unless a promoted objective rule exists.
7. v2 candidate is unchanged.
8. v3 objective-residual candidate is unchanged.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]

LATE_AVAIL = ROOT / "data" / "reference" / "late_availability_events.csv"
EXPECTED_XI = ROOT / "data" / "reference" / "expected_xi_status.csv"
GK_STATUS = ROOT / "data" / "reference" / "goalkeeper_status.csv"
FRIENDLY = ROOT / "data" / "reference" / "friendly_lineup_distortion_events.csv"

V2_MANIFEST = ROOT / "outputs" / "final_candidate_v2_auto_science" / "FROZEN_MANIFEST.json"
V3_MANIFEST = ROOT / "outputs" / "final_candidate_v3_objective_residual" / "FROZEN_MANIFEST.json"

LATE_AVAIL_REQUIRED = [
    "event_id", "team", "player", "player_position", "event_type", "status",
    "event_date", "source_date", "source_url", "source_name", "source_reliability",
    "timing_category", "role_tier", "role_importance_0_5",
    "already_reflected_score_0_5", "raw_penalty", "recommended_residual_penalty",
    "applies_to_score_candidate", "reason", "notes", "needs_review",
]
EXPECTED_XI_REQUIRED = [
    "team", "player", "position", "expected_xi_status", "role_tier",
    "role_importance_0_5", "evidence_source", "source_date", "source_url",
    "notes", "needs_review",
]
GK_REQUIRED = [
    "team", "goalkeeper", "expected_rank", "availability_status",
    "caps_or_experience_note", "club_2026", "source_date", "source_url",
    "crisis_flag", "crisis_reason", "needs_review",
]
FRIENDLY_REQUIRED = [
    "match_date", "team", "opponent", "friendly_match_id_if_known",
    "distortion_type", "affected_players", "estimated_strength_distortion",
    "should_downweight_for_form", "source_date", "source_url", "notes", "needs_review",
]


def _parse_bool(v) -> bool:
    import math
    if isinstance(v, bool):
        return v
    if isinstance(v, float) and math.isnan(v):
        return False
    return str(v).strip().lower() in {"true", "1", "yes"}


# ── 1. All four reference files exist with required columns ───────────────────

def test_late_availability_events_exists_with_required_columns():
    assert LATE_AVAIL.exists()
    df = pd.read_csv(LATE_AVAIL)
    for col in LATE_AVAIL_REQUIRED:
        assert col in df.columns, f"missing column: {col}"


def test_expected_xi_status_exists_with_required_columns():
    assert EXPECTED_XI.exists()
    df = pd.read_csv(EXPECTED_XI)
    for col in EXPECTED_XI_REQUIRED:
        assert col in df.columns, f"missing column: {col}"


def test_goalkeeper_status_exists_with_required_columns():
    assert GK_STATUS.exists()
    df = pd.read_csv(GK_STATUS)
    for col in GK_REQUIRED:
        assert col in df.columns, f"missing column: {col}"


def test_friendly_lineup_distortion_exists_with_required_columns():
    assert FRIENDLY.exists()
    df = pd.read_csv(FRIENDLY)
    for col in FRIENDLY_REQUIRED:
        assert col in df.columns, f"missing column: {col}"


# ── 2. Validation script runs without errors ──────────────────────────────────

def test_validation_script_runs():
    from scripts import validate_late_residual_inputs as v

    issues = v.main()
    errors = [i for i in issues if i["rule"] != 6]
    assert errors == [], f"validation errors: {errors}"


# ── 3. Messi row: active_monitoring, not key absence ─────────────────────────

def test_messi_is_active_monitoring_not_key_absence():
    df = pd.read_csv(LATE_AVAIL)
    messi = df[
        (df["team"] == "Argentina") & (df["player"] == "Lionel Messi")
    ]
    assert len(messi) == 1, "Messi row must exist in late_availability_events.csv"
    row = messi.iloc[0]
    assert row["status"] == "active_monitoring", (
        f"Messi status should be 'active_monitoring', got '{row['status']}'"
    )
    assert row["event_type"] not in {"withdrawn", "ruled_out"}, (
        f"Messi event_type must not be 'withdrawn' or 'ruled_out'; got '{row['event_type']}'"
    )
    # Must not be a key absence row.
    assert row["event_type"] != "withdrawn"
    combined = (
        str(row.get("notes", "")).lower()
        + " "
        + str(row.get("reason", "")).lower()
    )
    assert "key absence" not in combined.replace("not a key absence", ""), (
        "Messi row must not claim key absence"
    )
    # The reason or notes must signal this is an in-squad fitness risk.
    assert "fitness" in combined or "in_squad" in combined, (
        "Messi row must reference fitness or in_squad_fitness_risk"
    )


# ── 4. Karl row: ruled_out, late squad absence ────────────────────────────────

def test_karl_is_ruled_out():
    df = pd.read_csv(LATE_AVAIL)
    karl = df[
        (df["team"] == "Germany") & (df["player"] == "Lennart Karl")
    ]
    assert len(karl) == 1, "Karl row must exist in late_availability_events.csv"
    row = karl.iloc[0]
    assert row["status"] == "ruled_out", (
        f"Karl status should be 'ruled_out', got '{row['status']}'"
    )
    assert row["event_type"] == "ruled_out", (
        f"Karl event_type should be 'ruled_out', got '{row['event_type']}'"
    )
    assert row["timing_category"] == "late"


# ── 5. All penalties within allowed bounds ────────────────────────────────────

def test_penalties_within_bounds():
    df = pd.read_csv(LATE_AVAIL)
    for col in ("raw_penalty", "recommended_residual_penalty"):
        for idx, v in df[col].items():
            if pd.notna(v):
                p = float(v)
                assert -1.25 <= p <= 0, (
                    f"row {idx} {col}={p} outside [-1.25, 0]"
                )


# ── 6. No row applies to score candidate unless promoted rule exists ───────────

def test_no_row_applies_to_score_candidate_unless_rule_promoted():
    df = pd.read_csv(LATE_AVAIL)
    v3_promoted = False
    if V3_MANIFEST.exists():
        v3_promoted = bool(
            json.loads(V3_MANIFEST.read_text()).get("promotion_gate_passed")
        )
    for idx, row in df.iterrows():
        if _parse_bool(row.get("applies_to_score_candidate")):
            assert v3_promoted, (
                f"row {idx}: applies_to_score_candidate=true but no promoted v3 rule"
            )


# ── 7. v2 candidate unchanged ────────────────────────────────────────────────

def test_v2_candidate_files_unchanged():
    assert V2_MANIFEST.exists()
    manifest = json.loads(V2_MANIFEST.read_text())
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        assert path.exists(), f"v2 file missing: {entry['path']}"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == entry["sha256"], f"v2 file modified: {entry['path']}"


# ── 8. v3 objective-residual candidate unchanged ─────────────────────────────

def test_v3_candidate_files_unchanged():
    assert V3_MANIFEST.exists()
    manifest = json.loads(V3_MANIFEST.read_text())
    for entry in manifest.get("files", []):
        path = ROOT / entry["path"]
        assert path.exists(), f"v3 file missing: {entry['path']}"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == entry["sha256"], f"v3 file modified: {entry['path']}"
    # Sanity: the v3 manifest still reports a deterministic, non-manual-approval run.
    assert manifest["deterministic"] is True
    assert manifest["manual_approval_used"] is False
    assert manifest["is_recommended_score_fill_candidate"] is True
    assert manifest["n_adjusted_scores"] == 4


# ── Extra: readiness report and validation outputs exist ─────────────────────

def test_readiness_report_exists():
    assert (ROOT / "outputs" / "reports" / "late_residual_data_readiness.md").exists()


def test_validation_report_exists():
    assert (ROOT / "outputs" / "reports" / "late_residual_input_validation_report.md").exists()
    assert (
        ROOT / "outputs" / "predictions" / "late_residual_input_validation_issues.csv"
    ).exists()


def test_inventory_classifies_new_files_as_late_news():
    """The inventory catalog assigns the four new files to the late-news family."""
    catalog = pd.read_csv(ROOT / "outputs" / "predictions" / "data_inventory_catalog.csv")
    for fname in [
        "late_availability_events.csv",
        "expected_xi_status.csv",
        "goalkeeper_status.csv",
        "friendly_lineup_distortion_events.csv",
    ]:
        match = catalog[catalog["file_path"].str.contains(fname, na=False)]
        assert len(match) >= 1, f"{fname} not found in catalog"
        family = match.iloc[0]["family"]
        assert family == "8. Late-news / residual overlay data", (
            f"{fname} family is '{family}', expected '8. Late-news / residual overlay data'"
        )
