"""Tests for the post-cutoff recent international results update pipeline.

Verifies:
1.  Cutoff audit report exists
2.  Raw recent results file exists
3.  Clean recent results parquet exists
4.  All clean included rows are senior men's full internationals
5.  No needs_review=True row in clean parquet
6.  All clean rows have source provenance
7.  match_uid uniqueness
8.  Team mapping audit exists
9.  Lineup distortion audit exists
10. v2_auto_science candidate unchanged (hash verified)
11. v3_objective_residual candidate unchanged (hash verified)
12. fill_only prediction file unchanged (hash verified)
"""
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).parent.parent

# ── Paths ────────────────────────────────────────────────────────────────────
CUTOFF_AUDIT    = REPO / "outputs/reports/recent_international_cutoff_audit.md"
RAW_CSV         = REPO / "data/raw/manual/recent_senior_mens_international_results_since_cutoff_raw.csv"
CLEAN_PARQUET   = REPO / "data/interim/recent_senior_mens_international_results_since_cutoff.parquet"
PRED_CSV        = REPO / "outputs/predictions/recent_senior_mens_international_results_since_cutoff.csv"
TEAM_MAP_AUDIT  = REPO / "outputs/reports/recent_international_team_mapping_audit.md"
DISTORT_AUDIT   = REPO / "outputs/predictions/recent_international_lineup_distortion_audit.csv"

V2_MANIFEST     = REPO / "outputs/final_candidate_v2_auto_science/FROZEN_MANIFEST.json"
V3_MANIFEST     = REPO / "outputs/final_candidate_v3_objective_residual/FROZEN_MANIFEST.json"

# ── Fixtures ─────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def clean_df():
    return pd.read_parquet(CLEAN_PARQUET)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


# ── Test 1: Cutoff audit exists ──────────────────────────────────────────────
def test_cutoff_audit_exists():
    assert CUTOFF_AUDIT.exists(), f"Missing: {CUTOFF_AUDIT}"


# ── Test 2: Raw results file exists ─────────────────────────────────────────
def test_raw_recent_results_exists():
    assert RAW_CSV.exists(), f"Missing: {RAW_CSV}"


# ── Test 3: Clean parquet exists ────────────────────────────────────────────
def test_clean_recent_results_parquet_exists():
    assert CLEAN_PARQUET.exists(), f"Missing: {CLEAN_PARQUET}"


# ── Test 4: All clean included rows → senior_mens_full_international=True ───
def test_all_included_rows_are_senior_mens_full_internationals(clean_df):
    included = clean_df[clean_df["include_in_clean_update"]]
    bad = included[included["senior_mens_full_international"] != True]
    assert len(bad) == 0, (
        f"{len(bad)} included rows have senior_mens_full_international != True: "
        f"{bad['match_uid'].tolist()}"
    )


# ── Test 5: No needs_review=True row in clean parquet ───────────────────────
def test_no_needs_review_in_clean_parquet(clean_df):
    bad = clean_df[clean_df["needs_review"] == True]
    assert len(bad) == 0, (
        f"{len(bad)} rows in clean parquet have needs_review=True: "
        f"{bad['match_uid'].tolist()}"
    )


# ── Test 6: All clean rows have source provenance ───────────────────────────
def test_all_clean_rows_have_source_provenance(clean_df):
    included = clean_df[clean_df["include_in_clean_update"]]
    no_source = included[
        (included["source_primary"].fillna("") == "") &
        (included["source_url_primary"].fillna("") == "")
    ]
    assert len(no_source) == 0, (
        f"{len(no_source)} included rows have no source_primary or source_url_primary: "
        f"{no_source['match_uid'].tolist()}"
    )


# ── Test 7: match_uid uniqueness ────────────────────────────────────────────
def test_match_uid_uniqueness(clean_df):
    dupes = clean_df[clean_df["match_uid"].duplicated()]
    assert len(dupes) == 0, (
        f"{len(dupes)} duplicate match_uid rows: {dupes['match_uid'].tolist()}"
    )


# ── Test 8: Team mapping audit exists ───────────────────────────────────────
def test_team_mapping_audit_exists():
    assert TEAM_MAP_AUDIT.exists(), f"Missing: {TEAM_MAP_AUDIT}"


# ── Test 9: Lineup distortion audit exists ───────────────────────────────────
def test_lineup_distortion_audit_exists():
    assert DISTORT_AUDIT.exists(), f"Missing: {DISTORT_AUDIT}"
    df = pd.read_csv(DISTORT_AUDIT)
    assert len(df) > 0, "Lineup distortion audit is empty"


# ── Test 10: v2_auto_science candidate files unchanged ──────────────────────
def test_v2_auto_science_unchanged():
    assert V2_MANIFEST.exists(), f"Missing: {V2_MANIFEST}"
    manifest = json.loads(V2_MANIFEST.read_text())
    for entry in manifest.get("files", []):
        path = REPO / entry["path"]
        expected_sha = entry["sha256"]
        assert path.exists(), f"v2 file missing: {path}"
        actual_sha = sha256_file(path)
        assert actual_sha == expected_sha, (
            f"v2_auto_science file MODIFIED: {path.name}\n"
            f"  Expected SHA256: {expected_sha}\n"
            f"  Actual   SHA256: {actual_sha}"
        )


# ── Test 11: v3_objective_residual candidate files unchanged ────────────────
def test_v3_objective_residual_unchanged():
    assert V3_MANIFEST.exists(), f"Missing: {V3_MANIFEST}"
    manifest = json.loads(V3_MANIFEST.read_text())
    for entry in manifest.get("files", []):
        path = REPO / entry["path"]
        expected_sha = entry["sha256"]
        assert path.exists(), f"v3 file missing: {path}"
        actual_sha = sha256_file(path)
        assert actual_sha == expected_sha, (
            f"v3_objective_residual file MODIFIED: {path.name}\n"
            f"  Expected SHA256: {expected_sha}\n"
            f"  Actual   SHA256: {actual_sha}"
        )


# ── Test 12: fill_only prediction unchanged (v3) ────────────────────────────
def test_fill_only_prediction_unchanged():
    """Confirm the active fill_only prediction file has not been modified."""
    fill_only = REPO / "outputs/final_candidate_v3_objective_residual/final_group_score_predictions_fill_only.csv"
    assert fill_only.exists(), f"Missing: {fill_only}"
    manifest = json.loads(V3_MANIFEST.read_text())
    for entry in manifest.get("files", []):
        if "fill_only" in entry["path"]:
            expected_sha = entry["sha256"]
            actual_sha = sha256_file(fill_only)
            assert actual_sha == expected_sha, (
                f"fill_only file MODIFIED — active predictions changed without authorisation!\n"
                f"  Expected SHA256: {expected_sha}\n"
                f"  Actual   SHA256: {actual_sha}"
            )
            return
    pytest.skip("fill_only not found in v3 manifest")


# ── Sanity: dataset has expected size and date range ─────────────────────────
def test_dataset_size_and_date_range(clean_df):
    """Post-cutoff dataset must have exactly 35 rows covering June 3–8, 2026."""
    assert len(clean_df) == 35, f"Expected 35 rows, got {len(clean_df)}"
    dates = pd.to_datetime(clean_df["match_date"])
    assert str(dates.min().date()) == "2026-06-03", f"Min date: {dates.min().date()}"
    assert str(dates.max().date()) == "2026-06-08", f"Max date: {dates.max().date()}"


def test_all_goals_non_negative(clean_df):
    assert (clean_df["team_a_goals"] >= 0).all()
    assert (clean_df["team_b_goals"] >= 0).all()


def test_no_self_match(clean_df):
    bad = clean_df[clean_df["team_a_normalized"] == clean_df["team_b_normalized"]]
    assert len(bad) == 0


def test_wc2026_relevance_values(clean_df):
    valid = {"both_wc2026", "one_wc2026", "none_wc2026"}
    bad = clean_df[~clean_df["wc2026_match_relevance"].isin(valid)]
    assert len(bad) == 0, f"Invalid wc2026_match_relevance values: {bad['wc2026_match_relevance'].unique()}"
