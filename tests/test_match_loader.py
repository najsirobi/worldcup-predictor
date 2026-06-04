"""Test match data loading and validation."""
import pandas as pd
import pytest

from src.ingest.matches import (
    validate_match_schema,
    get_matches_before_date,
    get_recent_form,
    coerce_match_types,
    filter_played_matches,
    add_match_targets,
    validate_clean_matches,
    BACKBONE_COLUMNS,
)


@pytest.fixture
def raw_backbone_sample():
    """Tiny synthetic raw international-results table (incl. an unplayed fixture)."""
    return pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2026-12-31"],
        "home_team": ["England", "France", "Spain", "Brazil"],
        "away_team": ["France", "Germany", "Spain", "Argentina"],
        "home_score": [2, 1, 0, None],   # row 2 same team test handled separately; row 3 unplayed
        "away_score": [1, 1, 0, None],
        "tournament": ["Friendly", "Friendly", "Friendly", "FIFA World Cup"],
        "city": ["London", "Paris", "Madrid", "Dallas"],
        "country": ["England", "France", "Spain", "USA"],
        "neutral": ["FALSE", "FALSE", "TRUE", "TRUE"],
    })


def test_coerce_match_types(raw_backbone_sample):
    out = coerce_match_types(raw_backbone_sample)
    assert pd.api.types.is_datetime64_any_dtype(out["date"])
    assert out["neutral"].dtype == bool
    assert out["neutral"].tolist() == [False, False, True, True]


def test_filter_played_drops_unplayed(raw_backbone_sample):
    typed = coerce_match_types(raw_backbone_sample)
    played, dropped = filter_played_matches(typed)
    assert len(played) == 3
    assert len(dropped) == 1
    assert dropped.iloc[0]["tournament"] == "FIFA World Cup"


def test_add_match_targets_values(raw_backbone_sample):
    typed = coerce_match_types(raw_backbone_sample)
    played, _ = filter_played_matches(typed)
    out = add_match_targets(played).reset_index(drop=True)
    # row0: 2-1 home win, row1: 1-1 draw, row2: 0-0 draw
    assert out.loc[0, "result_label"] == "home_win"
    assert out.loc[0, "home_points"] == 3 and out.loc[0, "away_points"] == 0
    assert out.loc[1, "result_label"] == "draw"
    assert out.loc[1, "home_points"] == 1 and out.loc[1, "away_points"] == 1
    assert out.loc[0, "goal_diff"] == 1 and out.loc[0, "total_goals"] == 3
    assert out["home_score"].dtype == int and out["away_score"].dtype == int


def test_add_match_targets_away_win():
    df = pd.DataFrame({c: [] for c in BACKBONE_COLUMNS})
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01"]),
        "home_team": ["A"], "away_team": ["B"],
        "home_score": [0.0], "away_score": [3.0],
        "tournament": ["Friendly"], "city": ["X"], "country": ["Y"], "neutral": [False],
    })
    out = add_match_targets(df)
    assert out.loc[0, "result_label"] == "away_win"
    assert out.loc[0, "home_points"] == 0 and out.loc[0, "away_points"] == 3
    assert out.loc[0, "goal_diff"] == -3


def test_validate_clean_matches_rejects_same_team():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01"]),
        "home_team": ["Spain"], "away_team": ["Spain"],
        "home_score": [1], "away_score": [0],
        "tournament": ["Friendly"], "city": ["Madrid"], "country": ["Spain"], "neutral": [False],
    })
    with pytest.raises(ValueError, match="home_team == away_team"):
        validate_clean_matches(df)


def test_validate_clean_matches_rejects_negative_score():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01"]),
        "home_team": ["A"], "away_team": ["B"],
        "home_score": [-1], "away_score": [0],
        "tournament": ["Friendly"], "city": ["X"], "country": ["Y"], "neutral": [False],
    })
    with pytest.raises(ValueError, match="non-negative"):
        validate_clean_matches(df)


@pytest.fixture
def sample_match_data():
    """Create sample international match data."""
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=10, freq="D"),
        "home_team": ["England", "France", "Germany", "Spain", "Italy"] * 2,
        "away_team": ["France", "Germany", "Spain", "Italy", "England"] * 2,
        "home_score": [2, 1, 3, 0, 1] * 2,
        "away_score": [1, 1, 2, 2, 0] * 2,
    })


def test_validate_match_schema_valid(sample_match_data):
    """Test validation of correct match schema."""
    # Should not raise
    validate_match_schema(sample_match_data)


def test_validate_match_schema_missing_column():
    """Test validation fails with missing required column."""
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=3),
        "home_team": ["A", "B", "C"],
        "away_team": ["X", "Y", "Z"],
        # Missing home_score and away_score
    })

    with pytest.raises(ValueError, match="required columns"):
        validate_match_schema(df)


def test_validate_match_schema_wrong_dtype():
    """Test validation fails with wrong data type."""
    df = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],  # string, not datetime
        "home_team": ["A", "B", "C"],
        "away_team": ["X", "Y", "Z"],
        "home_score": [1, 2, 3],
        "away_score": [0, 1, 2],
    })

    with pytest.raises(ValueError, match="datetime"):
        validate_match_schema(df)


def test_get_matches_before_date(sample_match_data):
    """Test filtering matches before a date."""
    cutoff = "2024-01-05"
    result = get_matches_before_date(sample_match_data, cutoff)

    assert len(result) == 4
    assert (result["date"] < cutoff).all()


def test_get_matches_before_date_no_matches(sample_match_data):
    """Test filtering when no matches are before date."""
    cutoff = "2023-01-01"
    result = get_matches_before_date(sample_match_data, cutoff)

    assert len(result) == 0


def test_get_matches_before_date_non_iso_string_dates():
    """Dates are compared as timestamps, not lexicographically.

    Lexicographic comparison breaks on non-zero-padded dates: the string
    '2024-1-9' sorts after '2024-1-10'. Real timestamp parsing handles it.
    """
    df = pd.DataFrame({
        "date": ["2024-1-9", "2024-1-10", "2024-1-11"],
        "home_team": ["A", "B", "C"],
        "away_team": ["X", "Y", "Z"],
        "home_score": [1, 2, 3],
        "away_score": [0, 1, 2],
    })

    result = get_matches_before_date(df, "2024-01-10")
    # Only Jan 9 is strictly before Jan 10.
    assert list(result["home_team"]) == ["A"]


def test_get_recent_form_orders_by_date_not_position():
    """Recent form uses most-recent-by-date even if rows are unordered."""
    df = pd.DataFrame({
        "date": ["2024-03-01", "2024-01-01", "2024-02-01"],
        "home_team": ["England", "England", "England"],
        "away_team": ["X", "Y", "Z"],
        "home_score": [1, 2, 3],
        "away_score": [0, 1, 2],
    })

    result = get_recent_form(df, "England", "2024-04-01", periods=2)
    # All three are before the cutoff; home appearances counted, capped at periods.
    assert result["recent_home_matches"] == 2


def test_get_recent_form(sample_match_data):
    """Test getting recent form for a team."""
    result = get_recent_form(sample_match_data, "England", "2024-01-10")

    assert result["team"] == "England"
    assert result["date"] == "2024-01-10"
    assert "recent_home_matches" in result
    assert "recent_away_matches" in result
