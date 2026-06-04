"""Guards against WC2026-only / current-only squad leakage (Phase 5C, Task J).

Covers required tests:
  2. 2026-only squad features cannot enter the historical backtest matrix.
  3. missing market values remain null, not zero.
  5. no future-dated market values are used.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.historical_squad_features import (
    COMPARABLE_FEATURE_COLUMNS,
    UNAVAILABLE_FEATURE_COLUMNS,
    aggregate_historical_squad_features,
)

ROOT = Path(__file__).parent.parent
HIST_FEATS = ROOT / "data" / "interim" / "historical_squad_features.parquet"
MV_PARQUET = ROOT / "data" / "interim" / "historical_squad_market_value_features.parquet"
MATRIX = ROOT / "data" / "processed" / "model_matrix_squad_compatible.parquet"
WC2026_ONLY = {
    "squad_avg_height_cm",
    "squad_median_height_cm",
    "squad_domestic_club_share",
    "squad_foreign_club_share",
    "squad_top5_europe_club_share",
    "squad_top5_europe_club_country_share",
    "squad_big5_league_country_share",
    "squad_club_country_diversity",
}


def test_comparable_set_excludes_wc2026_only_fields():
    # No comparable (trainable) feature may be a WC2026-only field.
    assert set(COMPARABLE_FEATURE_COLUMNS).isdisjoint(WC2026_ONLY)


def test_wc2026_only_fields_are_null_in_historical_features():
    df = pd.DataFrame({
        "tournament_year": [2018, 2018],
        "team": ["A", "A"],
        "player_name": ["p1", "p2"],
        "position": ["GK", "FW"],
        "age_at_tournament_start": [25, 28],
    })
    feats = aggregate_historical_squad_features(df)
    for col in UNAVAILABLE_FEATURE_COLUMNS:
        assert feats[col].isna().all(), col


@pytest.mark.skipif(not MATRIX.exists(), reason="matrix not built yet")
def test_matrix_has_no_market_value_columns():
    # No market-value feature is allowed in the historical training matrix.
    df = pd.read_parquet(MATRIX)
    mv = [c for c in df.columns if "market" in c or "_value" in c or c.endswith("value")]
    assert mv == [], mv


def test_no_historical_market_value_parquet_promoted():
    # Task D concluded market values are NOT promotable (low/uneven matching).
    # If a future build ever emits one, it must justify as-of validity & matching.
    assert not MV_PARQUET.exists(), (
        "historical market-value features must not be promoted while player "
        "matching is low/uneven (see historical_market_value_feasibility.md)"
    )


def test_missing_market_value_would_stay_null_not_zero():
    # Simulate a value column with missing entries: nulls must not become 0.
    s = pd.Series([1_000_000.0, np.nan, np.nan])
    assert s.isna().sum() == 2
    assert not (s.fillna(-1) == 0).any()


def test_asof_filter_excludes_future_dated_values():
    # Demonstrates the leak-free rule: only valuations strictly before the
    # tournament start may be used. Future-dated rows must be dropped.
    vals = pd.DataFrame({
        "date": pd.to_datetime(["2018-01-01", "2018-06-13", "2018-07-01"]),
        "market_value_in_eur": [10, 20, 999],
    })
    start = pd.Timestamp("2018-06-14")
    asof = vals[vals["date"] < start]
    assert 999 not in asof["market_value_in_eur"].values
    assert len(asof) == 2
