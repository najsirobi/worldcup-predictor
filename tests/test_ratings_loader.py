"""Test rating cleaning transforms with tiny synthetic data (no raw Kaggle)."""
import pandas as pd
import pytest

from src.ingest.ratings import standardize_elo, standardize_fifa, validate_clean_ratings


def _map(rows):
    cols = ["source", "raw_name", "canonical_team_name", "country_code", "notes"]
    return pd.DataFrame(rows, columns=cols)


EMPTY_MAP = _map([])


def test_standardize_elo_basic_and_whitespace():
    raw = pd.DataFrame({
        # NBSP inside "South\xa0Africa" must normalize to a regular space
        "date": ["1872-11-30", "3/23/1901", "2024-01-01"],
        "team": ["England", "South\xa0Africa", "England"],
        "rating": [2003.0, 1500.0, 1850.0],
        "change": [3, 0, 5],
    })
    out = standardize_elo(raw, EMPTY_MAP)
    assert list(out.columns) == [
        "rating_date", "raw_team_name", "canonical_team_name", "country_code",
        "elo_rating", "source", "source_file",
    ]
    # mixed date formats both parse
    assert out["rating_date"].notna().all()
    # NBSP normalized -> canonical (identity) is "South Africa" with a normal space
    assert "South Africa" in set(out["canonical_team_name"])
    assert "South\xa0Africa" not in set(out["canonical_team_name"])


def test_standardize_elo_drops_null_rating():
    raw = pd.DataFrame({
        "date": ["2024-01-01", "2024-02-01"],
        "team": ["England", "France"],
        "rating": [1800.0, None],   # France row dropped
        "change": [0, 0],
    })
    out = standardize_elo(raw, EMPTY_MAP)
    assert set(out["canonical_team_name"]) == {"England"}


def test_standardize_elo_uses_explicit_map():
    raw = pd.DataFrame({
        "date": ["2024-01-01"], "team": ["China"], "rating": [1700.0], "change": [0],
    })
    mp = _map([{"source": "international_elo", "raw_name": "China",
                "canonical_team_name": "China PR", "country_code": "CHN", "notes": "test"}])
    out = standardize_elo(raw, mp)
    assert out["canonical_team_name"].iloc[0] == "China PR"


def test_standardize_elo_dedupe_same_team_date():
    raw = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-01"],
        "team": ["England", "England"],
        "rating": [1800.0, 1805.0],
        "change": [0, 0],
    })
    out = standardize_elo(raw, EMPTY_MAP)
    # one (team,date) row kept deterministically (last after stable sort -> 1805)
    assert len(out) == 1
    assert out["elo_rating"].iloc[0] == 1805.0


def test_standardize_fifa_basic():
    raw = pd.DataFrame({
        "rank": [1.0, 50.0],
        "country_full": ["Germany", "Korea Republic"],
        "country_abrv": ["GER", "KOR"],
        "total_points": [1700.0, 1500.0],
        "rank_date": ["1992-12-31", "2024-06-20"],
    })
    out = standardize_fifa(raw, EMPTY_MAP)
    assert list(out.columns) == [
        "ranking_date", "raw_team_name", "canonical_team_name", "country_code",
        "fifa_rank", "fifa_points", "source", "source_file",
    ]
    assert out["fifa_rank"].notna().all()
    assert set(out["country_code"]) == {"GER", "KOR"}


def test_validate_clean_ratings_rejects_duplicates():
    df = pd.DataFrame({
        "canonical_team_name": ["England", "England"],
        "rating_date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
        "elo_rating": [1800.0, 1810.0],
    })
    with pytest.raises(ValueError, match="duplicate"):
        validate_clean_ratings(df, "rating_date", ["elo_rating"])


def test_validate_clean_ratings_rejects_null_canonical():
    df = pd.DataFrame({
        "canonical_team_name": ["England", None],
        "rating_date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
        "elo_rating": [1800.0, 1810.0],
    })
    with pytest.raises(ValueError, match="canonical_team_name"):
        validate_clean_ratings(df, "rating_date", ["elo_rating"])
