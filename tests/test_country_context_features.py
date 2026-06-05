"""Tests for match-level, leakage-safe country-context features (Task F)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.country_context_match import (
    PRIMARY_FEATURES,
    SECONDARY_FEATURES,
    add_country_context_features,
    core_diff_features,
    feature_columns,
)

ROOT = Path(__file__).parent.parent
MATRIX = ROOT / "data" / "processed" / "model_matrix_country_context.parquet"


@pytest.fixture
def mapping() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "canonical_team": ["Brazil", "Argentina", "England", "Scotland"],
            "world_bank_code": ["BRA", "ARG", "GBR", "GBR"],
            "is_proxy_mapping": [False, False, True, True],
        }
    )


@pytest.fixture
def wb() -> pd.DataFrame:
    rows = []
    # Distinct values per year so we can detect which year was selected.
    series = {
        "BRA": {2016: 1.0e12, 2017: 2.0e12, 2018: 9.9e12},
        "ARG": {2016: 5.0e11, 2017: 6.0e11, 2018: 9.0e12},
        "GBR": {2016: 3.0e12, 2017: 3.1e12, 2018: 9.5e12},
    }
    for code, by_year in series.items():
        for year, gdp in by_year.items():
            rows.append(
                {
                    "country_code": code,
                    "year": year,
                    "is_aggregate": False,
                    "NY.GDP.MKTP.CD": gdp,
                    "NY.GDP.PCAP.CD": gdp / 1e6,
                    "SP.POP.TOTL": 1.0e8,
                    "SP.URB.TOTL.IN.ZS": 80.0,
                    "SP.DYN.LE00.IN": 75.0,
                    "SE.XPD.TOTL.GD.ZS": 4.0,
                    "GB.XPD.RSDV.GD.ZS": 1.0,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def matches() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2018-06-01", "2018-06-02", "2018-06-03"]),
            "match_year": [2018, 2018, 2018],
            "tournament": ["FIFA World Cup"] * 3,
            "home_team": ["Brazil", "England", "Narnia"],
            "away_team": ["Argentina", "Brazil", "Brazil"],
        }
    )


def test_latest_pre_tournament_value_is_used(mapping, wb, matches):
    out = add_country_context_features(matches, mapping, wb)
    # 2018 match must use the 2017 value (latest strictly before 2018), not 2016.
    assert out.loc[0, "home_cc_log_gdp"] == pytest.approx(np.log10(2.0e12))


def test_post_tournament_values_are_not_used(mapping, wb, matches):
    out = add_country_context_features(matches, mapping, wb)
    # The 2018 (same-year / post) value must never be selected.
    assert out.loc[0, "home_cc_log_gdp"] != pytest.approx(np.log10(9.9e12))


def test_missing_values_create_missing_flags(mapping, wb, matches):
    out = add_country_context_features(matches, mapping, wb)
    # Narnia is unmapped -> no context, null feature, missing flag set.
    assert not out.loc[2, "home_cc_has_context"]
    assert pd.isna(out.loc[2, "home_cc_log_gdp"])
    assert bool(out.loc[2, "home_cc_log_gdp_missing"]) is True
    # Mapped team has a value and a False missing flag.
    assert bool(out.loc[0, "home_cc_log_gdp_missing"]) is False


def test_england_scotland_flagged_as_proxy(mapping, wb, matches):
    out = add_country_context_features(matches, mapping, wb)
    assert out.loc[1, "home_cc_is_proxy"]  # England
    assert out.loc[1, "home_cc_world_bank_code"] == "GBR"
    assert out.loc[1, "any_proxy_mapping_in_match"] == 1


def test_proxy_missing_variant_drops_england_context(mapping, wb, matches):
    out = add_country_context_features(matches, mapping, wb)
    # All-with-proxy: England has a GBR value -> diff defined.
    assert not pd.isna(out.loc[1, "log_gdp_diff"])
    # Direct-only / proxy-missing: England dropped -> diff is null.
    assert pd.isna(out.loc[1, "direct_log_gdp_diff"])
    assert not out.loc[1, "has_country_context_features_direct"]
    # Non-proxy match keeps the direct diff.
    assert not pd.isna(out.loc[0, "direct_log_gdp_diff"])


def test_secondary_indicators_not_primary_by_default():
    assert "education_spend_pct_gdp" not in PRIMARY_FEATURES
    assert "rd_spend_pct_gdp" not in PRIMARY_FEATURES
    assert set(SECONDARY_FEATURES) == {"education_spend_pct_gdp", "rd_spend_pct_gdp"}
    # The core (primary) diff feature set excludes the sparse secondary indicators.
    core = set(core_diff_features("all"))
    assert "education_spend_pct_gdp_diff" not in core
    assert "rd_spend_pct_gdp_diff" not in core


@pytest.mark.skipif(not MATRIX.exists(), reason="country-context matrix not built")
def test_built_matrix_preserves_rows_and_flags():
    matrix = pd.read_parquet(MATRIX)
    baseline = pd.read_parquet(ROOT / "data" / "processed" / "model_matrix_baseline.parquet")
    assert len(matrix) == len(baseline)
    for col in ["home_score", "away_score", "result_label", "home_goals", "away_goals"]:
        assert col in matrix.columns
    for col in feature_columns():
        assert col in matrix.columns
    # Missing flags must match null-ness of their feature on the all-with-proxy side.
    assert matrix["home_cc_log_gdp"].isna().eq(matrix["home_cc_log_gdp_missing"]).all()
