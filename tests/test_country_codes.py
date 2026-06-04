"""Test country code mapping and normalization."""
import pandas as pd
import pytest

from src.ingest.country_codes import (
    create_country_code_normalizer,
    normalize_country_name,
    get_country_code,
)


@pytest.fixture
def sample_country_mapping():
    """Create sample country code mapping for testing."""
    return pd.DataFrame({
        "source": ["world_bank", "world_bank", "world_bank"],
        "raw_country_name": ["United Kingdom", "France", "United States"],
        "canonical_country_name": ["England", "France", "USA"],
        "fifa_code": ["ENG", "FRA", "USA"],
        "iso2": ["GB", "FR", "US"],
        "iso3": ["GBR", "FRA", "USA"],
        "world_bank_code": ["GBR", "FRA", "USA"],
        "notes": ["test", "test", "test"],
    })


def test_create_normalizer(sample_country_mapping):
    """Test creating a country code normalizer."""
    normalizer = create_country_code_normalizer(sample_country_mapping)

    assert "world_bank" in normalizer
    assert "United Kingdom" in normalizer["world_bank"]
    assert normalizer["world_bank"]["United Kingdom"] == "England"


def test_normalize_valid_country(sample_country_mapping):
    """Test normalizing a valid country name."""
    normalizer = create_country_code_normalizer(sample_country_mapping)
    result = normalize_country_name("United Kingdom", "world_bank", normalizer)

    assert result == "England"


def test_normalize_invalid_source(sample_country_mapping):
    """Test that invalid source raises error."""
    normalizer = create_country_code_normalizer(sample_country_mapping)

    with pytest.raises(ValueError, match="Source.*not found"):
        normalize_country_name("France", "invalid_source", normalizer)


def test_normalize_unknown_country(sample_country_mapping):
    """Test that unknown country raises error."""
    normalizer = create_country_code_normalizer(sample_country_mapping)

    with pytest.raises(ValueError, match="not found in mapping"):
        normalize_country_name("UnknownCountry", "world_bank", normalizer)


def test_conflicting_country_mapping_raises():
    """Duplicate raw_country_name with different canonical values must raise."""
    df = pd.DataFrame({
        "source": ["world_bank", "world_bank"],
        "raw_country_name": ["Korea", "Korea"],
        "canonical_country_name": ["South Korea", "North Korea"],
        "fifa_code": ["KOR", "PRK"],
        "iso2": ["KR", "KP"],
        "iso3": ["KOR", "PRK"],
        "world_bank_code": ["KOR", "PRK"],
        "notes": ["", ""],
    })

    with pytest.raises(ValueError, match="Conflicting country mapping"):
        create_country_code_normalizer(df)


def test_get_fifa_code(sample_country_mapping):
    """Test getting FIFA code for a country."""
    code = get_country_code("England", "fifa_code", sample_country_mapping)
    assert code == "ENG"


def test_get_iso2_code(sample_country_mapping):
    """Test getting ISO2 code for a country."""
    code = get_country_code("France", "iso2", sample_country_mapping)
    assert code == "FR"


def test_get_code_missing_country(sample_country_mapping):
    """Test that missing country raises error."""
    with pytest.raises(ValueError, match="not found in mapping"):
        get_country_code("UnknownCountry", "fifa_code", sample_country_mapping)
