"""Tests for historical World Cup squad ingestion (Phase 5C, Task B/J)."""

import numpy as np
import pandas as pd

from src.ingest.historical_squads import STANDARD_COLUMNS, _completed_age


def test_standard_schema_columns_present():
    expected = {
        "tournament_year", "team", "raw_team_name", "canonical_team_name",
        "country_code", "player_name", "position", "date_of_birth",
        "age_at_tournament_start", "club", "club_country", "height_cm",
        "coach_name", "source", "source_file", "source_quality", "parse_notes",
    }
    assert set(STANDARD_COLUMNS) == expected


def test_completed_age_is_floor_of_years():
    birth = pd.Series(["1990-06-01", "2000-01-01", None])
    ref = pd.Series(["2018-06-14", "2018-06-14", "2018-06-14"])
    age = _completed_age(birth, ref)
    assert age.iloc[0] == 28.0  # 28 completed years
    assert age.iloc[1] == 18.0
    assert pd.isna(age.iloc[2])  # missing DOB stays null, not zero


def test_unavailable_fields_stay_null_not_zero():
    # Simulate the standard-schema output: club/club_country/height must be null.
    row = {c: pd.NA for c in STANDARD_COLUMNS}
    row.update({"club": pd.NA, "club_country": pd.NA, "height_cm": pd.NA})
    df = pd.DataFrame([row])
    for col in ["club", "club_country", "height_cm"]:
        assert df[col].isna().all()
        assert not (df[col] == 0).any()
