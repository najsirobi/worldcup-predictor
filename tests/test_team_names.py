"""Test team name mapping and normalization."""
import pandas as pd
import pytest

from src.ingest.team_names import (
    create_team_name_normalizer,
    normalize_team_name,
)


@pytest.fixture
def sample_team_mapping():
    """Create sample team name mapping for testing."""
    return pd.DataFrame({
        "source": ["international_results", "international_results", "international_results"],
        "raw_name": ["England", "France", "USA"],
        "canonical_team_name": ["England", "France", "USA"],
        "country_code": ["ENG", "FRA", "USA"],
        "notes": ["test", "test", "test"],
    })


def test_create_normalizer(sample_team_mapping):
    """Test creating a team name normalizer."""
    normalizer = create_team_name_normalizer(sample_team_mapping)

    assert "international_results" in normalizer
    assert "England" in normalizer["international_results"]
    assert normalizer["international_results"]["England"] == "England"


def test_normalize_valid_team_name(sample_team_mapping):
    """Test normalizing a valid team name."""
    normalizer = create_team_name_normalizer(sample_team_mapping)
    result = normalize_team_name("England", "international_results", normalizer)

    assert result == "England"


def test_normalize_invalid_source(sample_team_mapping):
    """Test that invalid source raises error."""
    normalizer = create_team_name_normalizer(sample_team_mapping)

    with pytest.raises(ValueError, match="Source.*not found"):
        normalize_team_name("England", "invalid_source", normalizer)


def test_normalize_unknown_team(sample_team_mapping):
    """Test that unknown team name raises error (no silent normalization)."""
    normalizer = create_team_name_normalizer(sample_team_mapping)

    with pytest.raises(ValueError, match="not found in mapping"):
        normalize_team_name("UnknownTeam", "international_results", normalizer)


def test_conflicting_mapping_raises():
    """Duplicate raw_name with different canonical values must not pass silently."""
    df = pd.DataFrame({
        "source": ["international_results", "international_results"],
        "raw_name": ["Korea", "Korea"],
        "canonical_team_name": ["South Korea", "North Korea"],
        "country_code": ["KOR", "PRK"],
        "notes": ["", ""],
    })

    with pytest.raises(ValueError, match="Conflicting team-name mapping"):
        create_team_name_normalizer(df)


def test_duplicate_identical_mapping_allowed():
    """An exact duplicate row is harmless and should not raise."""
    df = pd.DataFrame({
        "source": ["international_results", "international_results"],
        "raw_name": ["England", "England"],
        "canonical_team_name": ["England", "England"],
        "country_code": ["ENG", "ENG"],
        "notes": ["", ""],
    })

    normalizer = create_team_name_normalizer(df)
    assert normalizer["international_results"]["England"] == "England"


def test_multiple_sources(sample_team_mapping):
    """Test normalizer with multiple sources."""
    df = pd.concat([
        sample_team_mapping,
        pd.DataFrame({
            "source": ["fifa_ranking"],
            "raw_name": ["ENG"],
            "canonical_team_name": ["England"],
            "country_code": ["ENG"],
            "notes": ["FIFA code"],
        }),
    ], ignore_index=True)

    normalizer = create_team_name_normalizer(df)

    # Test both sources work
    assert normalize_team_name("England", "international_results", normalizer) == "England"
    assert normalize_team_name("ENG", "fifa_ranking", normalizer) == "England"
