"""Tests for WC2026 country-context data and feature outputs."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd

from src.features.country_context import latest_value_before_year

ROOT = Path(__file__).parent.parent
FEATURES = ROOT / "data" / "interim" / "country_context_features.parquet"


def test_latest_value_before_year_uses_strict_pre_tournament_cutoff():
    frame = pd.DataFrame(
        {
            "country_code": ["AAA", "AAA", "AAA"],
            "year": [2024, 2025, 2026],
            "is_aggregate": [False, False, False],
            "SP.POP.TOTL": [1.0, 2.0, 3.0],
        }
    )

    value, year = latest_value_before_year(frame, "AAA", "SP.POP.TOTL", 2026)

    assert value == 2.0
    assert year == 2025


def test_country_context_feature_rows_cover_all_48_teams():
    frame = pd.read_parquet(FEATURES)
    assert len(frame) == 48
    assert frame["team"].nunique() == 48


def test_missing_values_have_matching_missing_flags():
    frame = pd.read_parquet(FEATURES)

    assert frame["log_gdp"].isna().eq(frame["gdp_current_usd_missing"]).all()
    assert frame["log_population"].isna().eq(frame["population_total_missing"]).all()
    assert frame["log_gdp_per_capita"].isna().eq(frame["gdp_per_capita_current_usd_missing"]).all()
    assert frame["education_spend_pct_gdp"].isna().eq(frame["education_spend_pct_gdp_missing"]).all()
    assert frame["rd_spend_pct_gdp"].isna().eq(frame["rd_spend_pct_gdp_missing"]).all()
    assert frame["urbanisation_pct"].isna().eq(frame["urbanisation_pct_missing"]).all()
    assert frame["life_expectancy"].isna().eq(frame["life_expectancy_missing"]).all()


def test_value_years_are_strictly_before_2026():
    frame = pd.read_parquet(FEATURES)
    for column in [
        "gdp_value_year",
        "gdp_per_capita_value_year",
        "population_value_year",
        "education_spend_pct_gdp_value_year",
        "rd_spend_pct_gdp_value_year",
        "urbanisation_value_year",
        "life_expectancy_value_year",
    ]:
        series = frame[column].dropna()
        assert (series < 2026).all(), column


def test_final_candidate_v2_auto_science_is_not_modified():
    result = subprocess.run(
        ["git", "diff", "--name-only", "--", "outputs/final_candidate_v2_auto_science"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == ""
