"""Test rating as-of-date rules (no future leakage)."""
import pandas as pd
import pytest

from src.ingest.ratings import get_rating_asof_date, asof_rating_join


# ---------------------------------------------------------------------------
# Vectorized as-of join (merge_asof, strict <) — used by the ratings build
# ---------------------------------------------------------------------------

@pytest.fixture
def asof_ratings():
    return pd.DataFrame({
        "canonical_team_name": ["England", "England", "England", "France"],
        "rating_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-01-01"]),
        "elo_rating": [1800.0, 1810.0, 1820.0, 1750.0],
    })


def _matches(rows):
    return pd.DataFrame(rows)


def test_asof_uses_latest_strictly_before(asof_ratings):
    m = _matches([{"date": pd.Timestamp("2024-02-15"), "team": "England"}])
    res = asof_rating_join(m, asof_ratings, "team", ["elo_rating"], "home")
    # latest before 2024-02-15 is the 2024-02-01 rating (1810), not 2024-03-01
    assert res["home_elo_rating"].iloc[0] == 1810.0
    assert res["home_rating_date"].iloc[0] == pd.Timestamp("2024-02-01")


def test_asof_does_not_use_rating_on_match_date(asof_ratings):
    # match exactly on 2024-02-01 -> must use 2024-01-01 (1800), NOT same-day 1810
    m = _matches([{"date": pd.Timestamp("2024-02-01"), "team": "England"}])
    res = asof_rating_join(m, asof_ratings, "team", ["elo_rating"], "home")
    assert res["home_elo_rating"].iloc[0] == 1800.0
    assert res["home_rating_date"].iloc[0] == pd.Timestamp("2024-01-01")


def test_asof_does_not_use_future_rating(asof_ratings):
    # match before any rating -> NaN (no future leakage)
    m = _matches([{"date": pd.Timestamp("2023-12-01"), "team": "England"}])
    res = asof_rating_join(m, asof_ratings, "team", ["elo_rating"], "home")
    assert pd.isna(res["home_elo_rating"].iloc[0])
    assert pd.isna(res["home_rating_date"].iloc[0])


def test_asof_missing_team_yields_nan_not_guess(asof_ratings):
    # team not present in ratings -> NaN, never substituted/guessed
    m = _matches([{"date": pd.Timestamp("2024-06-01"), "team": "Atlantis"}])
    res = asof_rating_join(m, asof_ratings, "team", ["elo_rating"], "home")
    assert pd.isna(res["home_elo_rating"].iloc[0])


def test_asof_duplicate_team_date_raises(asof_ratings):
    dup = pd.concat([asof_ratings, asof_ratings.iloc[[0]]], ignore_index=True)
    m = _matches([{"date": pd.Timestamp("2024-06-01"), "team": "England"}])
    with pytest.raises(ValueError, match="duplicate"):
        asof_rating_join(m, dup, "team", ["elo_rating"], "home")


def test_asof_preserves_match_order(asof_ratings):
    # rows intentionally out of date order; result must align to input order
    m = _matches([
        {"date": pd.Timestamp("2024-03-15"), "team": "England"},
        {"date": pd.Timestamp("2024-01-15"), "team": "England"},
        {"date": pd.Timestamp("2024-06-01"), "team": "France"},
    ])
    res = asof_rating_join(m, asof_ratings, "team", ["elo_rating"], "home")
    assert list(res["home_elo_rating"]) == [1820.0, 1800.0, 1750.0]


@pytest.fixture
def sample_elo_data():
    """Create sample Elo rating data."""
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=12, freq="MS"),
        "team": ["England"] * 12,
        "rating": [1800, 1805, 1810, 1808, 1815, 1820, 1818, 1825, 1830, 1835, 1840, 1845],
    })


def test_get_rating_asof_exact_date(sample_elo_data):
    """Test getting rating on a match date (latest strictly before)."""
    # Match on 2024-02-01, get latest rating strictly before
    rating = get_rating_asof_date(sample_elo_data, "England", "2024-02-01", rating_col="rating")

    # Should get 2024-01-01 rating (1800), not 2024-02-01
    assert rating == 1800


def test_get_rating_no_future_leakage(sample_elo_data):
    """Test that future ratings are not used (strict no-leakage rule)."""
    # Match on 2024-03-01
    rating = get_rating_asof_date(sample_elo_data, "England", "2024-03-01", rating_col="rating")

    # Should get 2024-02-01 rating (1805), not any later rating
    assert rating == 1805


def test_get_rating_no_rating_available(sample_elo_data):
    """Test error when team has no rating before date."""
    with pytest.raises(ValueError, match="No ratings found"):
        get_rating_asof_date(sample_elo_data, "England", "2023-01-01", rating_col="rating")


def test_get_rating_unknown_team(sample_elo_data):
    """Test error for unknown team."""
    with pytest.raises(ValueError, match="not found"):
        get_rating_asof_date(sample_elo_data, "UnknownTeam", "2024-02-01", rating_col="rating")


def test_get_rating_with_fifa_ranking_columns():
    """FIFA ranking uses 'country_full' + 'rank_date'; resolve them correctly."""
    df = pd.DataFrame({
        "rank_date": ["1992-12-31", "2023-07-20", "2024-06-20"],
        "country_full": ["Germany", "Germany", "Germany"],
        "total_points": [57.0, 1600.0, 1700.0],
    })

    rating = get_rating_asof_date(
        df, "Germany", "2024-01-01", rating_col="total_points"
    )
    # Latest strictly before 2024-01-01 is the 2023-07-20 snapshot.
    assert rating == 1600.0


def test_get_rating_no_team_column_raises():
    """A dataframe without any team identifier column raises a clear error."""
    df = pd.DataFrame({"date": ["2024-01-01"], "rating": [1800]})
    with pytest.raises(ValueError, match="'team', 'country', or 'country_full'"):
        get_rating_asof_date(df, "England", "2024-02-01", rating_col="rating")


def test_get_rating_multiple_teams():
    """Test with data for multiple teams."""
    df = pd.DataFrame({
        "date": [
            "2024-01-01", "2024-01-15",
            "2024-02-01", "2024-02-15",
            "2024-03-01", "2024-03-15"
        ],
        "team": ["England", "France", "England", "France", "England", "France"],
        "rating": [1800, 1750, 1805, 1760, 1810, 1770],
    })
    df["date"] = pd.to_datetime(df["date"])

    # Get England's rating before 2024-02-01
    eng_rating = get_rating_asof_date(df, "England", "2024-02-01", rating_col="rating")
    assert eng_rating == 1800

    # Get France's rating before 2024-02-01
    fra_rating = get_rating_asof_date(df, "France", "2024-02-01", rating_col="rating")
    assert fra_rating == 1750

    # Different match date
    eng_rating_later = get_rating_asof_date(df, "England", "2024-04-01", rating_col="rating")
    assert eng_rating_later == 1810
