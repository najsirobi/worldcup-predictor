"""Test country context data loading and validation."""
import pandas as pd
import pytest

from src.ingest.country_context import validate_country_context_schema


def test_validate_country_context_with_country_column():
    """Test validation with 'country' column."""
    df = pd.DataFrame({
        "country": ["England", "France"],
        "population": [56000000, 67000000],
    })

    # Should not raise
    validate_country_context_schema(df, "world_bank")


def test_validate_country_context_with_country_code():
    """Test validation with 'country_code' column."""
    df = pd.DataFrame({
        "country_code": ["ENG", "FRA"],
        "population": [56000000, 67000000],
    })

    # Should not raise
    validate_country_context_schema(df, "world_bank")


def test_validate_country_context_with_country_name():
    """Test validation with 'country_name' column."""
    df = pd.DataFrame({
        "country_name": ["England", "France"],
        "gdp": [1000, 2000],
    })

    # Should not raise
    validate_country_context_schema(df, "world_bank")


def test_validate_country_context_missing_country_identifier():
    """Test validation fails without country identifier."""
    df = pd.DataFrame({
        "population": [56000000, 67000000],
        "gdp": [1000, 2000],
    })

    with pytest.raises(ValueError, match="country"):
        validate_country_context_schema(df, "world_bank")


def test_validate_country_context_empty_dataframe():
    """Test validation with empty dataframe (warning, not error)."""
    df = pd.DataFrame()

    # Should log warning but not raise
    validate_country_context_schema(df, "world_bank")
