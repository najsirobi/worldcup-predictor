"""Tests for historical squad feature aggregation (Phase 5C, Task C/J)."""

import numpy as np
import pandas as pd

from src.features.historical_squad_features import (
    COMPARABLE_FEATURE_COLUMNS,
    UNAVAILABLE_FEATURE_COLUMNS,
    aggregate_historical_squad_features,
)


def _synthetic_squads():
    # One team, known position mix and ages; no height / club_country.
    positions = ["GK", "GK", "DF", "DF", "DF", "MF", "MF", "MF", "FW", "FW"]
    ages = [30, 28, 27, 26, 25, 24, 23, 29, 31, 22]
    return pd.DataFrame({
        "tournament_year": [2018] * 10,
        "team": ["Wonderland"] * 10,
        "player_name": [f"P{i}" for i in range(10)],
        "position": positions,
        "age_at_tournament_start": ages,
        "height_cm": [pd.NA] * 10,
        "club_country": [pd.NA] * 10,
        "source": ["world_cup_database"] * 10,
    })


def test_aggregation_basic_counts_and_shares():
    feats = aggregate_historical_squad_features(_synthetic_squads())
    assert len(feats) == 1
    row = feats.iloc[0]
    assert row["squad_player_count"] == 10
    assert row["squad_gk_count"] == 2
    assert row["squad_df_count"] == 3
    assert row["squad_mf_count"] == 3
    assert row["squad_fw_count"] == 2
    # attacker share uses ONLY forward positions
    assert row["squad_fw_share"] == 2 / 10
    assert row["squad_defensive_share"] == (2 + 3) / 10
    assert row["squad_midfield_share"] == 3 / 10
    assert abs(row["squad_avg_age"] - np.mean([30, 28, 27, 26, 25, 24, 23, 29, 31, 22])) < 1e-9
    assert row["squad_oldest_player_age"] == 31
    assert row["squad_youngest_player_age"] == 22


def test_attacker_share_excludes_non_attackers():
    # Flip every non-FW to a clearly non-attacking label; share must stay fw/total.
    df = _synthetic_squads()
    feats = aggregate_historical_squad_features(df)
    fw = (df["position"] == "FW").sum()
    assert feats.iloc[0]["squad_fw_share"] == fw / len(df)


def test_unavailable_features_are_null_not_zero():
    feats = aggregate_historical_squad_features(_synthetic_squads())
    for col in UNAVAILABLE_FEATURE_COLUMNS:
        # height/club shares unavailable -> null, never zero
        assert feats[col].isna().all(), col
        assert not (feats[col].fillna(-1) == 0).any(), col


def test_comparable_columns_have_no_current_only_fields():
    banned = ("height", "club", "domestic", "foreign", "market", "value")
    for col in COMPARABLE_FEATURE_COLUMNS:
        assert not any(b in col for b in banned), col


def test_missing_age_stays_null():
    df = _synthetic_squads()
    df["age_at_tournament_start"] = pd.NA
    feats = aggregate_historical_squad_features(df)
    assert pd.isna(feats.iloc[0]["squad_avg_age"])
    # counts still computed
    assert feats.iloc[0]["squad_player_count"] == 10
