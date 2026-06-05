"""Comprehensive QA tests for WC2026 match-to-venue reference table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ingest.wc2026_venues import altitude_bucket

ROOT = Path(__file__).resolve().parents[1]
MATCH_FILE = ROOT / "data" / "reference" / "wc2026_match_venues.csv"
ENRICHED_FILE = ROOT / "data" / "reference" / "wc2026_match_venues_enriched.csv"
VENUE_FILE = ROOT / "data" / "reference" / "wc2026_venues.csv"
GROUP_TEMPLATE_FILE = ROOT / "data" / "reference" / "fif8a_group_stage_template.csv"


def _match_df() -> pd.DataFrame:
    return pd.read_csv(MATCH_FILE)


def _enriched_df() -> pd.DataFrame:
    return pd.read_csv(ENRICHED_FILE)


def _venue_df() -> pd.DataFrame:
    return pd.read_csv(VENUE_FILE)


def _group_template_df() -> pd.DataFrame:
    return pd.read_csv(GROUP_TEMPLATE_FILE)


# Task B: Match venue file QA tests

def test_exactly_104_match_rows():
    """Verify exactly 104 match rows."""
    df = _match_df()
    assert len(df) == 104


def test_match_number_unique():
    """Verify match_number is unique."""
    df = _match_df()
    assert df["match_number"].is_unique


def test_match_number_range_1_to_104():
    """Verify match_number range is 1–104."""
    df = _match_df()
    assert set(df["match_number"]) == set(range(1, 105))


def test_exactly_72_group_matches():
    """Verify exactly 72 group stage matches."""
    df = _match_df()
    group_matches = df[df["match_number"].between(1, 72)]
    assert len(group_matches) == 72


def test_exactly_32_knockout_matches():
    """Verify exactly 32 knockout matches."""
    df = _match_df()
    knockout_matches = df[df["match_number"].between(73, 104)]
    assert len(knockout_matches) == 32


def test_group_matches_have_group_label():
    """Verify group matches (1–72) have group label."""
    df = _match_df()
    group_matches = df[df["match_number"].between(1, 72)]
    assert group_matches["group"].notna().all()


def test_knockout_matches_have_null_group():
    """Verify knockout matches (73–104) have null group."""
    df = _match_df()
    knockout_matches = df[df["match_number"].between(73, 104)]
    assert knockout_matches["group"].isna().all()


def test_group_match_alignment():
    """Verify group matches 1–72 align with template and predictions."""
    match_df = _match_df().query("match_number <= 72").sort_values("match_number").reset_index(drop=True)
    template_df = _group_template_df().sort_values("match_number").reset_index(drop=True)

    # Merge on match_number to compare
    merged = match_df[["match_number", "group", "date", "team_a", "team_b"]].merge(
        template_df[["match_number", "group", "date", "team_a", "team_b"]],
        on="match_number",
        suffixes=("_match", "_template"),
        validate="one_to_one",
    )

    for field in ["group", "date", "team_a", "team_b"]:
        col_match = f"{field}_match"
        col_template = f"{field}_template"
        assert merged[col_match].equals(merged[col_template]), f"Mismatch on {field}"


def test_every_match_has_required_venue_fields():
    """Verify every match has venue assignment fields."""
    df = _match_df()
    assert df["venue_id"].notna().all()
    assert df["stadium"].notna().all()
    assert df["host_city"].notna().all()
    assert df["host_country"].notna().all()


def test_every_match_has_source_documentation():
    """Verify every match has source/extraction_method."""
    df = _match_df()
    assert df["source"].notna().all()
    assert df["extraction_method"].notna().all()


def test_no_knockout_row_invents_participants():
    """Verify knockout rows don't invent participants."""
    df = _match_df()
    knockout_matches = df[df["match_number"].between(73, 104)]

    # Knockout team_a/team_b should be slot notation or winner references
    valid_prefixes = ("1", "2", "best", "W", "L")
    for _, row in knockout_matches.iterrows():
        team_a = str(row["team_a"])
        team_b = str(row["team_b"])
        # Should start with valid knockout notation
        assert any(team_a.startswith(p) for p in valid_prefixes) or "rd from" in team_a
        assert any(team_b.startswith(p) for p in valid_prefixes) or "rd from" in team_b


def test_needs_review_false_means_source_backed():
    """Verify needs_review=false rows have source documentation."""
    df = _match_df()
    source_backed = df[~df["needs_review"].fillna(False)]
    assert (source_backed["source"].notna() & (source_backed["source"] != "")).all()


# Task E: Enriched join QA tests

def test_enriched_has_104_rows():
    """Verify enriched file has 104 rows."""
    df = _enriched_df()
    assert len(df) == 104


def test_enriched_match_numbers_unique():
    """Verify enriched file has unique match numbers."""
    df = _enriched_df()
    assert df["match_number"].nunique() == 104


def test_enriched_every_match_has_venue_id():
    """Verify every enriched match has venue_id."""
    df = _enriched_df()
    assert df["venue_id"].notna().all()


def test_enriched_every_match_joins_to_one_venue():
    """Verify every match joins to exactly one venue."""
    match_df = _match_df()
    enriched_df = _enriched_df()
    venue_df = _venue_df()

    # Count venue rows per match
    for _, match_row in match_df.iterrows():
        enriched_row = enriched_df[enriched_df["match_number"] == match_row["match_number"]]
        assert len(enriched_row) == 1
        # Venue should exist
        assert match_row["venue_id"] in venue_df["venue_id"].values


def test_enriched_all_16_venues_represented():
    """Verify all 16 venues represented in enriched file."""
    df = _enriched_df()
    assert df["venue_id"].nunique() == 16


def test_enriched_geocode_fields_populated():
    """Verify enriched file has geocode fields populated."""
    df = _enriched_df()
    assert df["latitude"].notna().all()
    assert df["longitude"].notna().all()
    assert df["elevation_m"].notna().all()


def test_enriched_altitude_bucket_populated():
    """Verify enriched file has altitude_bucket populated."""
    df = _enriched_df()
    assert df["altitude_bucket"].notna().all()
    assert set(df["altitude_bucket"].unique()).issubset({"low", "moderate", "high", "very_high"})


def test_enriched_preserves_match_structure():
    """Verify enrichment doesn't alter match numbers or team orientation."""
    match_df = _match_df().sort_values("match_number")
    enriched_df = _enriched_df().sort_values("match_number")

    for _, (match_row, enriched_row) in enumerate(zip(match_df.itertuples(), enriched_df.itertuples())):
        assert match_row.match_number == enriched_row.match_number
        assert match_row.team_a == enriched_row.team_a
        assert match_row.team_b == enriched_row.team_b


# Task F: Altitude match QA tests

def test_altitude_bucket_matches_elevation():
    """Verify altitude_bucket matches elevation_m."""
    df = _enriched_df()

    for _, row in df.iterrows():
        elevation = float(row["elevation_m"])
        bucket = row["altitude_bucket"]
        expected_bucket = altitude_bucket(elevation)
        assert bucket == expected_bucket, f"M{int(row['match_number'])}: elevation={elevation}m, bucket={bucket}, expected={expected_bucket}"


def test_13_moderate_high_altitude_matches():
    """Verify exactly 13 moderate/high altitude matches."""
    df = _enriched_df()
    altitude_matches = df[df["altitude_bucket"].isin(["moderate", "high", "very_high"])]
    assert len(altitude_matches) == 13


def test_9_high_altitude_matches():
    """Verify exactly 9 high altitude matches."""
    df = _enriched_df()
    high_altitude = df[df["altitude_bucket"] == "high"]
    assert len(high_altitude) == 9


def test_4_moderate_altitude_matches():
    """Verify exactly 4 moderate altitude matches."""
    df = _enriched_df()
    moderate_altitude = df[df["altitude_bucket"] == "moderate"]
    assert len(moderate_altitude) == 4


def test_mexico_city_is_high_altitude():
    """Verify Mexico City matches are high altitude."""
    df = _enriched_df()
    mexico_city_matches = df[df["host_city"] == "Mexico City"]
    assert (mexico_city_matches["altitude_bucket"] == "high").all()
    assert (mexico_city_matches["elevation_m"] > 2000).all()


def test_guadalajara_is_high_altitude():
    """Verify Guadalajara matches are high altitude."""
    df = _enriched_df()
    guadalajara_matches = df[df["host_city"] == "Guadalajara"]
    assert (guadalajara_matches["altitude_bucket"] == "high").all()
    assert (guadalajara_matches["elevation_m"] > 1500).all()


def test_monterrey_is_moderate_altitude():
    """Verify Monterrey matches are moderate altitude."""
    df = _enriched_df()
    monterrey_matches = df[df["host_city"] == "Monterrey"]
    assert (monterrey_matches["altitude_bucket"] == "moderate").all()
    assert (monterrey_matches["elevation_m"].between(500, 1500)).all()


def test_no_very_high_altitude_matches():
    """Verify no matches at very high altitude (≥2500m)."""
    df = _enriched_df()
    very_high = df[df["altitude_bucket"] == "very_high"]
    assert len(very_high) == 0


# Task G/H: Downstream readiness tests

def test_venue_id_foreign_key_integrity():
    """Verify all match venue_ids reference valid venues."""
    match_df = _match_df()
    venue_df = _venue_df()
    valid_venue_ids = set(venue_df["venue_id"])

    for _, row in match_df.iterrows():
        assert row["venue_id"] in valid_venue_ids


def test_all_16_host_cities_in_matches():
    """Verify all 16 official host cities represented in matches."""
    match_df = _match_df()
    venue_df = _venue_df()

    expected_cities = set(venue_df["official_host_city"])
    actual_cities = set(match_df["host_city"])

    assert actual_cities == expected_cities


def test_group_matches_dont_use_knockout_notation():
    """Verify group matches (1–72) use actual team names."""
    df = _match_df()
    group_matches = df[df["match_number"].between(1, 72)]

    # Group matches should have real team names, not slot notation
    for _, row in group_matches.iterrows():
        assert not str(row["team_a"]).startswith("W"), f"M{int(row['match_number'])}: team_a has winner notation"
        assert not str(row["team_b"]).startswith("W"), f"M{int(row['match_number'])}: team_b has winner notation"


def test_no_matches_flagged_for_manual_review():
    """Verify no match rows are flagged needs_review=true."""
    df = _match_df()
    assert not (df["needs_review"] == True).any()


def test_enriched_no_matches_flagged_for_manual_review():
    """Verify no enriched rows are flagged needs_review=true."""
    df = _enriched_df()
    assert not (df["needs_review"] == True).any()
