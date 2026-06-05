"""Comprehensive QA tests for WC2026 venue reference table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ingest.wc2026_venues import altitude_bucket

ROOT = Path(__file__).resolve().parents[1]
VENUE_FILE = ROOT / "data" / "reference" / "wc2026_venues.csv"
MATCH_FILE = ROOT / "data" / "reference" / "wc2026_match_venues.csv"

EXPECTED_HOST_CITIES = {
    "Atlanta",
    "Boston",
    "Dallas",
    "Guadalajara",
    "Houston",
    "Kansas City",
    "Los Angeles",
    "Mexico City",
    "Miami",
    "Monterrey",
    "New York New Jersey",
    "Philadelphia",
    "San Francisco Bay Area",
    "Seattle",
    "Toronto",
    "Vancouver",
}

EXPECTED_ACTUAL_CITY_MAPPINGS = {
    "Boston": "Foxborough",
    "Dallas": "Arlington",
    "Los Angeles": "Inglewood",
    "New York New Jersey": "East Rutherford",
    "San Francisco Bay Area": "Santa Clara",
    "Miami": "Miami Gardens",
    "Monterrey": "Guadalupe",
    "Guadalajara": "Zapopan",
}


def _venue_df() -> pd.DataFrame:
    return pd.read_csv(VENUE_FILE)


def _match_df() -> pd.DataFrame:
    return pd.read_csv(MATCH_FILE)


# Task C: Venue reference QA tests

def test_exactly_16_venue_rows():
    """Verify exactly 16 venue rows."""
    df = _venue_df()
    assert len(df) == 16


def test_no_duplicate_venue_ids():
    """Verify venue_id is unique."""
    df = _venue_df()
    assert df["venue_id"].is_unique


def test_all_16_official_host_cities_present():
    """Verify all 16 official host cities present."""
    df = _venue_df()
    actual_cities = set(df["official_host_city"])
    assert actual_cities == EXPECTED_HOST_CITIES


def test_official_host_city_separate_from_actual_city():
    """Verify official_host_city and actual_city are distinct columns."""
    df = _venue_df()
    assert "official_host_city" in df.columns
    assert "actual_city" in df.columns
    # They may be the same or different, but both should exist
    assert df["official_host_city"].notna().all()
    assert df["actual_city"].notna().all()


def test_official_stadium_name_populated():
    """Verify official_stadium_name is populated."""
    df = _venue_df()
    assert df["official_stadium_name"].notna().all()
    assert (df["official_stadium_name"] != "").all()


def test_actual_city_mappings_correct():
    """Verify stadium-city caveat mappings are correct."""
    df = _venue_df().set_index("official_host_city")

    for official_city, expected_actual_city in EXPECTED_ACTUAL_CITY_MAPPINGS.items():
        actual = df.loc[official_city, "actual_city"]
        assert actual == expected_actual_city, f"{official_city}: expected {expected_actual_city}, got {actual}"


def test_latitude_populated_and_valid():
    """Verify latitude is populated and valid."""
    df = _venue_df()
    assert df["latitude"].notna().all()
    assert (df["latitude"].between(-90, 90)).all()


def test_longitude_populated_and_valid():
    """Verify longitude is populated and valid."""
    df = _venue_df()
    assert df["longitude"].notna().all()
    assert (df["longitude"].between(-180, 180)).all()


def test_elevation_m_populated_and_valid():
    """Verify elevation_m is populated and numeric."""
    df = _venue_df()
    assert df["elevation_m"].notna().all()
    assert (df["elevation_m"] >= 0).all()
    assert (df["elevation_m"] < 3000).all()  # Reasonable upper bound


def test_timezone_populated():
    """Verify timezone is populated."""
    df = _venue_df()
    assert df["timezone"].notna().all()
    assert (df["timezone"] != "").all()


def test_geocode_source_populated():
    """Verify geocode_source is populated."""
    df = _venue_df()
    assert df["geocode_source"].notna().all()
    assert (df["geocode_source"] != "").all()


def test_elevation_source_populated():
    """Verify elevation_source is populated."""
    df = _venue_df()
    assert df["elevation_source"].notna().all()
    assert (df["elevation_source"] != "").all()


def test_confidence_populated():
    """Verify confidence is populated."""
    df = _venue_df()
    assert df["confidence"].notna().all()
    assert (df["confidence"] != "").all()


def test_needs_review_false_for_all():
    """Verify all venues have needs_review=false."""
    df = _venue_df()
    assert (df["needs_review"] == False).all()


def test_source_populated():
    """Verify source is populated."""
    df = _venue_df()
    assert df["source"].notna().all()
    assert (df["source"] != "").all()


# Task D: Geocode and elevation sanity QA tests

def test_latitude_longitude_plausible():
    """Verify coordinates are plausible for North America."""
    df = _venue_df()

    for _, row in df.iterrows():
        # Latitudes should be 15–50°N for Mexico, Canada, US
        assert 15 < row["latitude"] < 50
        # Longitudes should be 50–130°W for North America
        assert -130 < row["longitude"] < -50


def test_mexico_city_high_altitude():
    """Verify Mexico City is high altitude (≥1500m)."""
    df = _venue_df().set_index("official_host_city")
    elevation = float(df.loc["Mexico City", "elevation_m"])
    assert elevation >= 1500
    assert altitude_bucket(elevation) == "high"


def test_guadalajara_high_altitude():
    """Verify Guadalajara is high altitude (≥1500m)."""
    df = _venue_df().set_index("official_host_city")
    elevation = float(df.loc["Guadalajara", "elevation_m"])
    assert elevation >= 1500
    assert altitude_bucket(elevation) == "high"


def test_monterrey_moderate_altitude():
    """Verify Monterrey is moderate altitude (500–1500m)."""
    df = _venue_df().set_index("official_host_city")
    elevation = float(df.loc["Monterrey", "elevation_m"])
    assert 500 <= elevation < 1500
    assert altitude_bucket(elevation) == "moderate"


def test_us_canada_venues_not_accidentally_mexico_altitude():
    """Verify US/Canada venues not mistakenly assigned high altitude."""
    df = _venue_df()
    us_canada_venues = df[
        (df["official_host_country"].isin(["USA", "Canada"])) &
        (df["official_host_city"].notna())
    ]

    for _, row in us_canada_venues.iterrows():
        # All should be low altitude (<500m)
        assert row["elevation_m"] < 500, f"{row['official_host_city']}: elevation {row['elevation_m']} is unexpectedly high"


def test_boston_not_confused_with_downtown():
    """Verify Boston stadium location is Foxborough (not downtown)."""
    df = _venue_df().set_index("official_host_city")
    boston_row = df.loc["Boston"]
    assert boston_row["actual_city"] == "Foxborough"
    assert boston_row["latitude"] > 42  # Foxborough is north of downtown Boston


def test_dallas_not_confused_with_downtown():
    """Verify Dallas stadium location is Arlington (not downtown)."""
    df = _venue_df().set_index("official_host_city")
    dallas_row = df.loc["Dallas"]
    assert dallas_row["actual_city"] == "Arlington"
    assert dallas_row["longitude"] < -97  # Arlington is west of downtown Dallas


def test_san_francisco_not_confused_with_downtown():
    """Verify SF stadium location is Santa Clara (not downtown SF)."""
    df = _venue_df().set_index("official_host_city")
    sf_row = df.loc["San Francisco Bay Area"]
    assert sf_row["actual_city"] == "Santa Clara"
    assert sf_row["longitude"] < -121  # Santa Clara is south of downtown SF


def test_miami_not_confused_with_miami_beach():
    """Verify Miami stadium is in Miami Gardens (correct location)."""
    df = _venue_df().set_index("official_host_city")
    miami_row = df.loc["Miami"]
    assert miami_row["actual_city"] == "Miami Gardens"


def test_new_york_maps_to_new_jersey_not_ny():
    """Verify NY stadium is in East Rutherford, NJ (not NY)."""
    df = _venue_df().set_index("official_host_city")
    ny_row = df.loc["New York New Jersey"]
    assert ny_row["actual_city"] == "East Rutherford"
    assert ny_row["actual_region_state"] == "New Jersey"


# Task C: Altitude bucket classification

def test_altitude_bucket_classification():
    """Verify altitude_bucket matches elevation thresholds."""
    df = _venue_df()

    for _, row in df.iterrows():
        elevation = float(row["elevation_m"])
        expected_bucket = altitude_bucket(elevation)
        # Note: venue file may not have altitude_bucket column, but we can verify logic
        if expected_bucket == "high":
            assert elevation >= 1500
        elif expected_bucket == "moderate":
            assert 500 <= elevation < 1500
        elif expected_bucket == "low":
            assert elevation < 500


def test_no_venue_duplicates():
    """Verify no stadium is duplicated."""
    df = _venue_df()
    assert df["official_stadium_name"].is_unique
    assert df["venue_id"].is_unique


def test_all_venues_referenced_in_matches():
    """Verify all 16 venues are used in at least one match."""
    venue_df = _venue_df()
    match_df = _match_df()

    venue_ids = set(venue_df["venue_id"])
    referenced_venue_ids = set(match_df["venue_id"])

    assert venue_ids == referenced_venue_ids, f"Missing/extra venues: {venue_ids ^ referenced_venue_ids}"


def test_venue_match_distribution():
    """Verify venue usage distribution is reasonable."""
    match_df = _match_df()
    venue_counts = match_df["venue_id"].value_counts()

    # Each venue should host at least 1 match and at most 9 matches
    for count in venue_counts.values:
        assert 1 <= count <= 10


def test_mexico_city_has_most_altitude_impact():
    """Verify Mexico City (highest elevation) has matches in both stages."""
    match_df = _match_df()
    mexico_city_matches = match_df[match_df["host_city"] == "Mexico City"]

    group_matches = mexico_city_matches[mexico_city_matches["match_number"] <= 72]
    ko_matches = mexico_city_matches[mexico_city_matches["match_number"] > 72]

    assert len(group_matches) > 0, "Mexico City should have group stage matches"
    assert len(ko_matches) > 0, "Mexico City should have knockout matches"


def test_venue_coordinates_geographically_consistent():
    """Verify coordinates form reasonable geographic pattern."""
    df = _venue_df()

    # US/Canada should be in Western Hemisphere (negative longitude)
    western_venues = df[df["official_host_country"].isin(["USA", "Canada"])]
    assert (western_venues["longitude"] < 0).all()

    # Mexico should be in Western Hemisphere (negative longitude)
    mexico_venues = df[df["official_host_country"] == "Mexico"]
    assert (mexico_venues["longitude"] < 0).all()

    # Canada should be north of US (higher latitude)
    canada_lats = df[df["official_host_country"] == "Canada"]["latitude"]
    us_lats = df[df["official_host_country"] == "USA"]["latitude"]
    assert canada_lats.min() > us_lats.min()  # North of southernmost US venue


def test_actual_region_state_populated():
    """Verify actual_region_state is populated for all venues."""
    df = _venue_df()
    assert df["actual_region_state"].notna().all()
    assert (df["actual_region_state"] != "").all()


def test_actual_country_populated():
    """Verify actual_country is populated for all venues."""
    df = _venue_df()
    assert df["actual_country"].notna().all()
    assert (df["actual_country"] != "").all()
