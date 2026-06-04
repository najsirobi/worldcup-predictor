"""Tests for squad feature aggregation."""

import pandas as pd

from src.features.squad_features import aggregate_squad_features, is_attacking_position


def test_squad_features_aggregate_from_tiny_player_data():
    players = pd.DataFrame(
        [
            {"tournament_name": "T", "team": "A", "player": "p1", "position": "ST", "age": 25, "market_value_eur": 100},
            {"tournament_name": "T", "team": "A", "player": "p2", "position": "GK", "age": 30, "market_value_eur": 50},
            {"tournament_name": "T", "team": "A", "player": "p3", "position": "CB", "age": 20, "market_value_eur": 25},
        ]
    )

    out = aggregate_squad_features(players)
    row = out.iloc[0]

    assert row["squad_player_count"] == 3
    assert row["squad_total_value"] == 175
    assert row["squad_top_11_value"] == 175
    assert row["squad_avg_age"] == 25
    assert row["goalkeeper_value"] == 50
    assert row["top_1_attacker_value"] == 100


def test_top_k_attacker_features_only_use_attacking_positions():
    players = pd.DataFrame(
        [
            {"tournament_name": "T", "team": "A", "player": "st", "position": "ST", "market_value_eur": 100},
            {"tournament_name": "T", "team": "A", "player": "am", "position": "AM", "market_value_eur": 80},
            {"tournament_name": "T", "team": "A", "player": "cb", "position": "CB", "market_value_eur": 200},
        ]
    )

    out = aggregate_squad_features(players)

    assert is_attacking_position("attacking midfielder")
    assert not is_attacking_position("CB")
    assert out.iloc[0]["top_3_attacker_value"] == 180


def test_missing_market_values_stay_missing_not_fake_zero():
    players = pd.DataFrame(
        [
            {"tournament_name": "T", "team": "A", "player": "p1", "position": "ST", "age": 25},
            {"tournament_name": "T", "team": "A", "player": "p2", "position": "GK", "age": 30},
        ]
    )

    out = aggregate_squad_features(players)

    assert pd.isna(out.iloc[0]["squad_total_value"])
    assert pd.isna(out.iloc[0]["top_1_attacker_value"])
    assert not out.iloc[0]["has_attacker_features"]
