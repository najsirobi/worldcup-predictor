"""Tests for leak-free rating momentum features."""

import pandas as pd
import pytest

from src.features.rating_momentum import (
    MOMENTUM_FEATURES,
    add_rating_momentum_features,
    validate_rating_momentum_no_leakage,
)


def _ratings():
    elo = pd.DataFrame(
        {
            "canonical_team_name": ["Team A", "Team A", "Team A", "Team A", "Team A", "Team B", "Team B", "Team B"],
            "rating_date": pd.to_datetime(
                [
                    "2022-06-01",
                    "2022-07-01",
                    "2023-01-01",
                    "2023-07-01",
                    "2024-06-01",
                    "2023-01-01",
                    "2023-07-01",
                    "2024-06-01",
                ]
            ),
            "elo_rating": [880.0, 900.0, 1000.0, 1060.0, 1180.0, 990.0, 1010.0, 1050.0],
        }
    )
    fifa = pd.DataFrame(
        {
            "canonical_team_name": ["Team A", "Team A", "Team B", "Team B"],
            "ranking_date": pd.to_datetime(["2023-01-01", "2024-06-01", "2023-01-01", "2024-06-01"]),
            "fifa_rank": [20.0, 10.0, 30.0, 35.0],
            "fifa_points": [1500.0, 1600.0, 1400.0, 1380.0],
        }
    )
    return elo, fifa


def test_rating_momentum_uses_latest_rating_strictly_before_shifted_cutoff():
    elo, fifa = _ratings()
    matches = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-07-01"]),
            "home_canon": ["Team A"],
            "away_canon": ["Team B"],
            "home_elo": [1180.0],
            "away_elo": [1050.0],
            "home_fifa_rank": [10.0],
            "away_fifa_rank": [35.0],
            "home_fifa_points": [1600.0],
            "away_fifa_points": [1380.0],
        }
    )

    out = add_rating_momentum_features(matches, elo, fifa)

    assert out["home_elo_change_6m"].iloc[0] == 120.0
    assert out["home_elo_change_12m"].iloc[0] == 180.0
    assert out["home_elo_change_24m"].iloc[0] == 300.0
    assert out["away_elo_change_12m"].iloc[0] == 60.0
    assert out["elo_change_12m_diff"].iloc[0] == 120.0
    assert out["home_fifa_points_change_12m"].iloc[0] == 100.0
    assert out["away_fifa_points_change_12m"].iloc[0] == -20.0
    assert out["fifa_rank_change_12m_diff"].iloc[0] == -15.0
    assert out["rating_momentum_slope_12m"].iloc[0] == 10.0
    validate_rating_momentum_no_leakage(out)


def test_missing_prior_rating_stays_null_not_zero():
    elo, fifa = _ratings()
    matches = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-02-01"]),
            "home_canon": ["Team C"],
            "away_canon": ["Team D"],
            "home_elo": [1000.0],
            "away_elo": [990.0],
            "home_fifa_rank": [20.0],
            "away_fifa_rank": [30.0],
            "home_fifa_points": [1500.0],
            "away_fifa_points": [1400.0],
        }
    )

    out = add_rating_momentum_features(matches, elo, fifa)

    assert pd.isna(out["home_elo_change_6m"].iloc[0])
    assert pd.isna(out["fifa_points_change_12m_diff"].iloc[0])


def test_momentum_leakage_validator_rejects_prior_date_on_cutoff():
    row = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-07-01"]),
            "home_elo_prior_6m_rating_date": pd.to_datetime(["2024-01-01"]),
            "away_elo_prior_6m_rating_date": pd.to_datetime(["2023-12-31"]),
            "home_elo_prior_12m_rating_date": pd.to_datetime(["2023-06-30"]),
            "away_elo_prior_12m_rating_date": pd.to_datetime(["2023-06-30"]),
            "home_elo_prior_24m_rating_date": pd.to_datetime(["2022-06-30"]),
            "away_elo_prior_24m_rating_date": pd.to_datetime(["2022-06-30"]),
            "home_fifa_prior_12m_ranking_date": pd.to_datetime(["2023-06-30"]),
            "away_fifa_prior_12m_ranking_date": pd.to_datetime(["2023-06-30"]),
        }
    )

    with pytest.raises(ValueError, match="LEAKAGE"):
        validate_rating_momentum_no_leakage(row)


def test_model_matrix_contains_rating_momentum_features():
    matrix = pd.read_parquet("data/processed/model_matrix_baseline.parquet")

    for feature in MOMENTUM_FEATURES:
        assert feature in matrix.columns
