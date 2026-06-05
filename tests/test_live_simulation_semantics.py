"""Tests for live simulation submitted/actual separation."""

import pandas as pd

from src.live.live_simulation import build_match_lambdas, simulate_live
from src.live.scores_override import build_initial_override, update_match
from src.live.tournament_state import actual_bracket_state_payload, compute_group_tables


def _one_group_scores(played_first: bool = False) -> pd.DataFrame:
    rows = [
        (1, "A", "A Team", "B Team"),
        (2, "A", "C Team", "D Team"),
        (3, "A", "A Team", "C Team"),
        (4, "A", "B Team", "D Team"),
        (5, "A", "A Team", "D Team"),
        (6, "A", "B Team", "C Team"),
    ]
    frame = pd.DataFrame(
        [
            {
                "match_number": match_number,
                "group": group,
                "date": "2026-06-11",
                "team_a": team_a,
                "team_b": team_b,
                "team_a_goals": pd.NA,
                "team_b_goals": pd.NA,
                "status": "scheduled",
                "source": "test",
                "updated_at": "",
                "notes": "",
            }
            for match_number, group, team_a, team_b in rows
        ]
    )
    if played_first:
        frame.loc[0, "team_a_goals"] = 0
        frame.loc[0, "team_b_goals"] = 5
        frame.loc[0, "status"] = "played"
    return frame


def _predictions(a_team_favored: bool) -> pd.DataFrame:
    score_by_match = {
        1: "4-0" if a_team_favored else "0-4",
        2: "0-0",
        3: "4-0" if a_team_favored else "0-4",
        4: "0-0",
        5: "4-0" if a_team_favored else "0-4",
        6: "0-0",
    }
    return pd.DataFrame(
        {
            "match_number": list(score_by_match),
            "final_recommended_score": list(score_by_match.values()),
        }
    )


def test_played_group_match_is_pinned_to_actual_result():
    summary = simulate_live(
        _one_group_scores(played_first=True),
        _predictions(a_team_favored=True),
        n_sims=500,
        seed=11,
    )
    by_team = summary.set_index("team")

    assert by_team.loc["A Team", "matches_played"] == 1
    assert by_team.loc["A Team", "current_points"] == 0
    assert by_team.loc["B Team", "matches_played"] == 1
    assert by_team.loc["B Team", "current_points"] == 3


def test_unplayed_matches_use_frozen_candidate_scorelines():
    scores = _one_group_scores(played_first=False)
    a_favored = simulate_live(scores, _predictions(a_team_favored=True), n_sims=2000, seed=7)
    a_faded = simulate_live(scores, _predictions(a_team_favored=False), n_sims=2000, seed=7)

    a_points_favored = a_favored[a_favored["team"].eq("A Team")]["expected_points"].iloc[0]
    a_points_faded = a_faded[a_faded["team"].eq("A Team")]["expected_points"].iloc[0]

    assert a_points_favored > a_points_faded + 3.0


def test_match_lambdas_are_from_frozen_score_predictions():
    predictions = pd.DataFrame(
        {"match_number": [1], "final_recommended_score": ["0-3"]}
    )

    assert build_match_lambdas(predictions)[1] == (0.25, 3.0)


def test_actual_bracket_state_waits_for_complete_group_stage():
    tables = compute_group_tables(build_initial_override())
    empty_mapping = pd.DataFrame()
    empty_annex = pd.DataFrame()

    state = actual_bracket_state_payload(tables, empty_mapping, empty_annex)

    assert state["status"] == "pending_group_stage"
    assert state["round_of_32_matches"] == []
