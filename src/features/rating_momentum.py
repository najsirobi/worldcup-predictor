"""Leak-free rating momentum features.

For each match, current ratings are already joined as latest rating strictly
before match date. Momentum compares that current pre-match rating with the
latest rating strictly before a shifted date such as match date minus 12 months.
"""

from __future__ import annotations

import pandas as pd

from src.ingest.ratings import asof_rating_join


MOMENTUM_FEATURES = [
    "home_fifa_points_change_12m",
    "away_fifa_points_change_12m",
    "fifa_points_change_12m_diff",
    "home_fifa_rank_change_12m",
    "away_fifa_rank_change_12m",
    "fifa_rank_change_12m_diff",
    "home_elo_change_6m",
    "away_elo_change_6m",
    "elo_change_6m_diff",
    "home_elo_change_12m",
    "away_elo_change_12m",
    "elo_change_12m_diff",
    "home_elo_change_24m",
    "away_elo_change_24m",
    "elo_change_24m_diff",
    "rating_momentum_slope_12m",
]


def _shifted_matches(matches: pd.DataFrame, months: int) -> pd.DataFrame:
    shifted = matches.copy()
    shifted["date"] = pd.to_datetime(shifted["date"]) - pd.DateOffset(months=months)
    return shifted


def _asof_shifted_rating(
    matches: pd.DataFrame,
    ratings: pd.DataFrame,
    *,
    side: str,
    months: int,
    value_cols: list[str],
    rating_date_col: str,
) -> pd.DataFrame:
    return asof_rating_join(
        _shifted_matches(matches, months),
        ratings,
        f"{side}_canon",
        value_cols,
        f"{side}_{months}m_prior",
        rating_date_col=rating_date_col,
    )


def add_rating_momentum_features(
    matches: pd.DataFrame,
    elo: pd.DataFrame,
    fifa: pd.DataFrame,
) -> pd.DataFrame:
    """Add pre-match rating momentum features without using future ratings.

    Required match columns:
    - `date`, `home_canon`, `away_canon`
    - current as-of columns already joined strictly before match date:
      `home_elo`, `away_elo`, `home_fifa_points`, `away_fifa_points`,
      `home_fifa_rank`, `away_fifa_rank`
    """
    out = matches.copy()
    out["date"] = pd.to_datetime(out["date"])

    for months in [6, 12, 24]:
        home_prior = _asof_shifted_rating(
            out,
            elo,
            side="home",
            months=months,
            value_cols=["elo_rating"],
            rating_date_col="rating_date",
        )
        away_prior = _asof_shifted_rating(
            out,
            elo,
            side="away",
            months=months,
            value_cols=["elo_rating"],
            rating_date_col="rating_date",
        )
        out[f"home_elo_change_{months}m"] = out["home_elo"] - home_prior[f"home_{months}m_prior_elo_rating"].values
        out[f"away_elo_change_{months}m"] = out["away_elo"] - away_prior[f"away_{months}m_prior_elo_rating"].values
        out[f"elo_change_{months}m_diff"] = out[f"home_elo_change_{months}m"] - out[f"away_elo_change_{months}m"]
        out[f"home_elo_prior_{months}m_rating_date"] = home_prior[f"home_{months}m_prior_rating_date"].values
        out[f"away_elo_prior_{months}m_rating_date"] = away_prior[f"away_{months}m_prior_rating_date"].values

    home_fifa_prior = _asof_shifted_rating(
        out,
        fifa,
        side="home",
        months=12,
        value_cols=["fifa_rank", "fifa_points"],
        rating_date_col="ranking_date",
    )
    away_fifa_prior = _asof_shifted_rating(
        out,
        fifa,
        side="away",
        months=12,
        value_cols=["fifa_rank", "fifa_points"],
        rating_date_col="ranking_date",
    )
    out["home_fifa_points_change_12m"] = out["home_fifa_points"] - home_fifa_prior["home_12m_prior_fifa_points"].values
    out["away_fifa_points_change_12m"] = out["away_fifa_points"] - away_fifa_prior["away_12m_prior_fifa_points"].values
    out["fifa_points_change_12m_diff"] = out["home_fifa_points_change_12m"] - out["away_fifa_points_change_12m"]

    out["home_fifa_rank_change_12m"] = out["home_fifa_rank"] - home_fifa_prior["home_12m_prior_fifa_rank"].values
    out["away_fifa_rank_change_12m"] = out["away_fifa_rank"] - away_fifa_prior["away_12m_prior_fifa_rank"].values
    out["fifa_rank_change_12m_diff"] = out["home_fifa_rank_change_12m"] - out["away_fifa_rank_change_12m"]
    out["home_fifa_prior_12m_ranking_date"] = home_fifa_prior["home_12m_prior_rating_date"].values
    out["away_fifa_prior_12m_ranking_date"] = away_fifa_prior["away_12m_prior_rating_date"].values

    out["rating_momentum_slope_12m"] = out["elo_change_12m_diff"] / 12.0
    return out


def validate_rating_momentum_no_leakage(df: pd.DataFrame) -> None:
    """Validate that every prior-rating date is strictly before its shifted cutoff."""
    date = pd.to_datetime(df["date"])
    checks = {
        "home_elo_prior_6m_rating_date": date - pd.DateOffset(months=6),
        "away_elo_prior_6m_rating_date": date - pd.DateOffset(months=6),
        "home_elo_prior_12m_rating_date": date - pd.DateOffset(months=12),
        "away_elo_prior_12m_rating_date": date - pd.DateOffset(months=12),
        "home_elo_prior_24m_rating_date": date - pd.DateOffset(months=24),
        "away_elo_prior_24m_rating_date": date - pd.DateOffset(months=24),
        "home_fifa_prior_12m_ranking_date": date - pd.DateOffset(months=12),
        "away_fifa_prior_12m_ranking_date": date - pd.DateOffset(months=12),
    }
    for column, cutoff in checks.items():
        if column not in df.columns:
            raise ValueError(f"Missing momentum prior date column: {column}")
        used = df[column].notna()
        prior_dates = pd.to_datetime(df.loc[used, column])
        if (prior_dates >= cutoff.loc[used]).any():
            raise ValueError(f"LEAKAGE: {column} is not strictly before its shifted cutoff")
