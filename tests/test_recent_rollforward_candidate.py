"""Tests for the v4 recent-rollforward candidate pipeline (Parts B–F).

Verifies:
1.  Updated model matrix (matches_with_ratings_recent_rollforward) exists and has
    expected row count (49296 + 35 = 49331)
2.  Updated model matrix has the same columns as the original
3.  Rolling form date range extended to 2026-06-08 in updated matrix
4.  Updated model matrix baseline (model_matrix_baseline_recent_rollforward) exists
5.  v4 candidate directory exists with required files
6.  v4 fill-only predictions has 72 rows
7.  All v4 fill-only scores are valid X-Y format
8.  v4 FROZEN_MANIFEST.json has required metadata fields
9.  v4 frozen candidate flags are set correctly
10. R1 adjustments file has exactly 72 rows
11. Every R1 change moves at most one goal
12. Delta vs v3 file covers all 72 matches
13. v2_auto_science remains byte-identical (re-verified after Part B)
14. v3_objective_residual remains byte-identical (re-verified after Part B)
15. active_candidate.yml points to v4
16. Dashboard JSON updated to reference v4
"""
import hashlib
import json
from pathlib import Path
import re

import pandas as pd
import pytest

REPO = Path(__file__).parent.parent

# ── Paths ────────────────────────────────────────────────────────────────────
MWR_V4          = REPO / "data/interim/matches_with_ratings_recent_rollforward.parquet"
MM_V4           = REPO / "data/interim/model_matrix_baseline_recent_rollforward.parquet"
MATCHES_CLEAN   = REPO / "data/interim/matches_clean.parquet"
MM_BASELINE     = REPO / "data/processed/model_matrix_baseline.parquet"
MM_REPORT       = REPO / "outputs/reports/recent_rollforward_model_matrix_report.md"

V4_DIR          = REPO / "outputs/final_candidate_v4_recent_rollforward"
V4_AUTO         = V4_DIR / "final_group_score_predictions_auto.csv"
V4_FILL         = V4_DIR / "final_group_score_predictions_fill_only.csv"
V4_ADJ          = V4_DIR / "recent_rollforward_adjustments.csv"
V4_DELTA        = V4_DIR / "recent_rollforward_delta_vs_v3.csv"
V4_REPORT       = V4_DIR / "recent_rollforward_candidate_report.md"
V4_MANIFEST     = V4_DIR / "FROZEN_MANIFEST.json"
PROMO_REPORT    = REPO / "outputs/reports/recent_rollforward_candidate_promotion_report.md"

V2_MANIFEST     = REPO / "outputs/final_candidate_v2_auto_science/FROZEN_MANIFEST.json"
V3_MANIFEST     = REPO / "outputs/final_candidate_v3_objective_residual/FROZEN_MANIFEST.json"

ACTIVE_YML      = REPO / "data/live/active_candidate.yml"
DASHBOARD_JSON  = REPO / "outputs/live/mobile_dashboard_data.json"

SCORE_RE        = re.compile(r"^\d{1,2}-\d{1,2}$")

# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def mwr_v4():
    return pd.read_parquet(MWR_V4)

@pytest.fixture(scope="module")
def mm_v4():
    return pd.read_parquet(MM_V4)

@pytest.fixture(scope="module")
def v4_adj():
    return pd.read_csv(V4_ADJ)

@pytest.fixture(scope="module")
def v4_fill():
    return pd.read_csv(V4_FILL)

@pytest.fixture(scope="module")
def v4_delta():
    return pd.read_csv(V4_DELTA)

@pytest.fixture(scope="module")
def v4_manifest():
    return json.loads(V4_MANIFEST.read_text())


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ── Test 1: updated ratings join parquet exists and has correct row count ─────
def test_matches_with_ratings_v4_row_count(mwr_v4):
    original = pd.read_parquet(MATCHES_CLEAN)
    expected = len(original) + 35
    assert len(mwr_v4) == expected, (
        f"Expected {expected} rows (original {len(original)} + 35 new); "
        f"got {len(mwr_v4)}"
    )


# ── Test 2: updated ratings join parquet has same columns as original ─────────
def test_matches_with_ratings_v4_columns_superset(mwr_v4):
    original = pd.read_parquet(MATCHES_CLEAN)
    missing = set(original.columns) - set(mwr_v4.columns)
    assert not missing, f"Columns missing from v4 ratings parquet: {missing}"


# ── Test 3: updated model matrix date range extends to 2026-06-08 ─────────────
def test_model_matrix_v4_date_range(mm_v4):
    max_date = pd.to_datetime(mm_v4["date"]).max().date()
    assert str(max_date) == "2026-06-08", (
        f"Expected max date 2026-06-08 in updated model matrix; got {max_date}"
    )


# ── Test 4: model matrix baseline v4 parquet exists ──────────────────────────
def test_model_matrix_v4_exists():
    assert MM_V4.exists(), f"Missing: {MM_V4}"
    assert MM_V4.stat().st_size > 0


# ── Test 5: v4 candidate directory has required files ────────────────────────
def test_v4_candidate_required_files():
    required = [
        "final_group_score_predictions_auto.csv",
        "final_group_score_predictions_fill_only.csv",
        "recent_rollforward_adjustments.csv",
        "recent_rollforward_delta_vs_v3.csv",
        "recent_rollforward_candidate_report.md",
        "FROZEN_MANIFEST.json",
        "final_group_standing_predictions_auto.csv",
        "final_last8_predictions_auto.csv",
        "final_submission_pack_auto.csv",
    ]
    missing = [f for f in required if not (V4_DIR / f).exists()]
    assert not missing, f"Missing v4 candidate files: {missing}"


# ── Test 6: v4 fill-only predictions has 72 rows ─────────────────────────────
def test_v4_fill_only_row_count(v4_fill):
    assert len(v4_fill) == 72, f"Expected 72 rows in fill-only; got {len(v4_fill)}"


# ── Test 7: all v4 fill-only scores are valid X-Y format ─────────────────────
def test_v4_fill_only_score_format(v4_fill):
    bad = [s for s in v4_fill["score_to_fill_in"] if not SCORE_RE.match(str(s))]
    assert not bad, f"Invalid score values in v4 fill-only: {bad[:5]}"


# ── Test 8: v4 FROZEN_MANIFEST has required metadata fields ──────────────────
def test_v4_manifest_required_fields(v4_manifest):
    required = [
        "schema_version", "candidate_dir", "candidate_name",
        "deterministic", "broad_human_overlay_used", "manual_approval_used",
        "subjective_override_used", "rolling_forward_update",
        "promotion_gate_passed", "files",
    ]
    missing = [f for f in required if f not in v4_manifest]
    assert not missing, f"Missing manifest fields: {missing}"


# ── Test 9: v4 frozen candidate flags are set correctly ──────────────────────
def test_v4_manifest_flags(v4_manifest):
    assert v4_manifest["broad_human_overlay_used"] is False, "broad_human_overlay_used must be false"
    assert v4_manifest["manual_approval_used"] is False, "manual_approval_used must be false"
    assert v4_manifest["subjective_override_used"] is False, "subjective_override_used must be false"
    assert v4_manifest["rolling_forward_update"] is True, "rolling_forward_update must be true"
    assert v4_manifest["deterministic"] is True, "deterministic must be true"


# ── Test 10: R1 adjustments file covers exactly 72 matches ───────────────────
def test_v4_adjustments_row_count(v4_adj):
    assert len(v4_adj) == 72, f"Expected 72 rows in adjustments; got {len(v4_adj)}"


# ── Test 11: every R1 change moves at most one goal ──────────────────────────
def test_v4_r1_changes_at_most_one_goal(v4_adj):
    def parse(s):
        a, b = str(s).split("-")
        return int(a), int(b)

    changed = v4_adj[v4_adj["changed"]]
    for _, r in changed.iterrows():
        ga, gb = parse(r["v4_auto_score"])
        na, nb = parse(r["final_score"])
        delta = abs(ga - na) + abs(gb - nb)
        assert delta <= 1, (
            f"match {r['match_number']}: R1 change moved more than 1 goal "
            f"({r['v4_auto_score']} -> {r['final_score']})"
        )


# ── Test 12: delta vs v3 covers all 72 matches ───────────────────────────────
def test_v4_delta_row_count(v4_delta):
    assert len(v4_delta) == 72, f"Expected 72 rows in delta vs v3; got {len(v4_delta)}"


# ── Test 13: v2_auto_science unchanged after rollforward build ────────────────
def test_v2_auto_science_unchanged_after_v4():
    assert V2_MANIFEST.exists(), f"Missing: {V2_MANIFEST}"
    manifest = json.loads(V2_MANIFEST.read_text())
    for entry in manifest.get("files", []):
        path = REPO / entry["path"]
        expected = entry["sha256"]
        assert path.exists(), f"v2 file missing: {path}"
        actual = sha256_file(path)
        assert actual == expected, (
            f"v2 file MODIFIED after v4 build: {path.name}\n"
            f"  Expected: {expected}\n  Actual:   {actual}"
        )


# ── Test 14: v3_objective_residual unchanged after rollforward build ──────────
def test_v3_objective_residual_unchanged_after_v4():
    assert V3_MANIFEST.exists(), f"Missing: {V3_MANIFEST}"
    manifest = json.loads(V3_MANIFEST.read_text())
    for entry in manifest.get("files", []):
        path = REPO / entry["path"]
        expected = entry["sha256"]
        assert path.exists(), f"v3 file missing: {path}"
        actual = sha256_file(path)
        assert actual == expected, (
            f"v3 file MODIFIED after v4 build: {path.name}\n"
            f"  Expected: {expected}\n  Actual:   {actual}"
        )


# ── Test 15: active_candidate.yml points to v4 ────────────────────────────────
def test_active_candidate_points_to_v4():
    assert ACTIVE_YML.exists(), f"Missing: {ACTIVE_YML}"
    content = ACTIVE_YML.read_text()
    assert "final_candidate_v4_recent_rollforward" in content, (
        "active_candidate.yml does not point to v4 after promotion gate passed"
    )


# ── Test 16: dashboard JSON references v4 (updated by build_mobile_dashboard) ─
def test_dashboard_json_references_v4():
    assert DASHBOARD_JSON.exists(), f"Missing: {DASHBOARD_JSON}"
    data = json.loads(DASHBOARD_JSON.read_text())
    candidate_name = (
        data.get("meta", {}).get("candidate_name", "") or
        data.get("candidate_name", "") or
        str(data)
    )
    assert "v4" in candidate_name or "rollforward" in candidate_name, (
        f"Dashboard JSON does not reference v4 candidate; "
        f"candidate_name found: {candidate_name[:200]}"
    )
