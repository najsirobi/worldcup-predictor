"""Tests for current-only WC2026 official squad features."""

import pandas as pd

from src.features.wc2026_squad_features import (
    aggregate_wc2026_squad_features,
    build_template_join_table,
)


def test_wc2026_positions_aggregate_correctly():
    squads = pd.DataFrame(
        [
            {"team": "A", "fifa_code": "AAA", "player_name": "gk", "position": "GK", "age_on_2026_06_11": 30, "height_cm": 190, "club_country_code": "AAA"},
            {"team": "A", "fifa_code": "AAA", "player_name": "df", "position": "DF", "age_on_2026_06_11": 25, "height_cm": 185, "club_country_code": "ENG"},
            {"team": "A", "fifa_code": "AAA", "player_name": "mf", "position": "MF", "age_on_2026_06_11": 24, "height_cm": 180, "club_country_code": "ESP"},
            {"team": "A", "fifa_code": "AAA", "player_name": "fw", "position": "FW", "age_on_2026_06_11": 20, "height_cm": 175, "club_country_code": "AAA"},
        ]
    )

    out = aggregate_wc2026_squad_features(squads).iloc[0]

    assert out["squad_player_count"] == 4
    assert out["squad_gk_count"] == 1
    assert out["squad_df_count"] == 1
    assert out["squad_mf_count"] == 1
    assert out["squad_fw_count"] == 1
    assert out["squad_fw_share"] == 0.25
    assert out["squad_domestic_club_share"] == 0.5
    assert out["squad_top5_europe_club_share"] == 0.5


def test_missing_heights_remain_null_not_fake_zero():
    squads = pd.DataFrame(
        [
            {"team": "A", "fifa_code": "AAA", "player_name": "p1", "position": "GK", "age_on_2026_06_11": 30, "height_cm": pd.NA, "club_country_code": "AAA"},
            {"team": "A", "fifa_code": "AAA", "player_name": "p2", "position": "FW", "age_on_2026_06_11": 20, "height_cm": pd.NA, "club_country_code": "ENG"},
        ]
    )

    out = aggregate_wc2026_squad_features(squads).iloc[0]

    assert pd.isna(out["squad_avg_height_cm"])
    assert pd.isna(out["squad_median_height_cm"])


def test_team_join_preserves_all_48_teams_when_data_complete():
    teams = [f"Team {idx:02d}" for idx in range(48)]
    template = pd.DataFrame(
        [
            {"group": f"G{idx // 4}", "team_a": teams[idx], "team_b": teams[(idx + 1) % 48]}
            for idx in range(48)
        ]
    )
    squads = pd.DataFrame(
        [
            {"team": team, "fifa_code": f"C{idx:02d}", "player_name": f"{team} player"}
            for idx, team in enumerate(teams)
        ]
    )

    join, mapping = build_template_join_table(template, squads)

    assert mapping.empty
    assert len(join) == 48
    assert join["has_official_squad_coverage"].all()
