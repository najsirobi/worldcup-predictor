import pandas as pd

from src.features.group_incentives import build_incentive_features_for_matches


def _fixture(rows):
    return pd.DataFrame(
        [
            {
                "year": 2026,
                "tournament_id": "TEST",
                "group": "A",
                "match_number": i + 1,
                "date": date,
                "team_a": a,
                "team_b": b,
                "team_a_goals": ga,
                "team_b_goals": gb,
            }
            for i, (date, a, b, ga, gb) in enumerate(rows)
        ]
    )


def test_incentive_features_use_only_prior_group_matches():
    matches = _fixture(
        [
            ("2026-06-01", "A", "B", 3, 0),
            ("2026-06-02", "A", "C", 2, 0),
            ("2026-06-02", "B", "D", 1, 0),
            ("2026-06-03", "C", "D", 1, 0),
            ("2026-06-04", "A", "D", 0, 0),
            ("2026-06-04", "B", "C", 0, 0),
        ]
    )
    features = build_incentive_features_for_matches(matches)

    first = features[features["match_number"].eq(1)].iloc[0]
    assert first["team_a_group_points_before"] == 0
    assert first["team_a_group_goals_for_before"] == 0
    assert first["team_b_group_points_before"] == 0

    second = features[features["match_number"].eq(2)].iloc[0]
    assert second["team_a_group_points_before"] == 3
    assert second["team_a_group_goals_for_before"] == 3
    # The current match result itself is not included.
    assert second["team_b_group_points_before"] == 0


def test_six_points_clinches_top2_before_final_match():
    matches = _fixture(
        [
            ("2026-06-01", "A", "B", 1, 0),
            ("2026-06-02", "A", "C", 1, 0),
            ("2026-06-02", "B", "D", 0, 0),
            ("2026-06-03", "C", "D", 1, 0),
            ("2026-06-04", "A", "D", 0, 0),
            ("2026-06-04", "B", "C", 0, 0),
        ]
    )
    features = build_incentive_features_for_matches(matches)
    final_a = features[features["match_number"].eq(5)].iloc[0]

    assert final_a["team_a_group_points_before"] == 6
    assert bool(final_a["team_a_has_clinched_top2"]) is True
    assert bool(final_a["team_a_low_incentive_flag"]) is True


def test_eliminated_team_is_detected():
    matches = _fixture(
        [
            ("2026-06-01", "A", "B", 1, 0),
            ("2026-06-02", "A", "C", 1, 0),
            ("2026-06-02", "B", "D", 2, 0),
            ("2026-06-03", "C", "D", 2, 0),
            ("2026-06-04", "A", "D", 0, 0),
            ("2026-06-04", "B", "C", 0, 0),
        ]
    )
    features = build_incentive_features_for_matches(matches)
    d_final = features[features["match_number"].eq(5)].iloc[0]

    assert d_final["team_b_group_points_before"] == 0
    assert bool(d_final["team_b_is_eliminated"]) is True
    assert bool(d_final["team_b_low_incentive_flag"]) is True


def test_must_win_team_is_detected():
    matches = _fixture(
        [
            ("2026-06-01", "A", "B", 1, 0),
            ("2026-06-02", "A", "C", 1, 0),
            ("2026-06-02", "B", "D", 1, 1),
            ("2026-06-03", "C", "D", 1, 0),
            ("2026-06-04", "A", "D", 0, 0),
            ("2026-06-04", "B", "C", 0, 0),
        ]
    )
    features = build_incentive_features_for_matches(matches)
    b_final = features[features["match_number"].eq(6)].iloc[0]

    assert b_final["team_a_group_points_before"] == 1
    assert bool(b_final["team_a_must_win_for_top2"]) is True
    assert bool(b_final["team_a_high_incentive_flag"]) is True
