"""Tests for round-by-round knockout exact-score predictions."""

import re
from pathlib import Path

import pandas as pd
import pytest

from src.live.knockout_predictions import (
    ROUND_BY_MATCH,
    build_knockout_predictions,
    most_probable_scoreline,
    predict_match,
)
from src.live.tournament_state import compute_group_tables

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "data" / "reference"
GROUP_VIEW = ROOT / "outputs" / "predictions" / "group_submission_view.csv"
SCORE_RE = re.compile(r"^\d+-\d+$")


@pytest.fixture(scope="module")
def inputs():
    return {
        "group_view": pd.read_csv(GROUP_VIEW),
        "r32_mapping": pd.read_csv(REFERENCE / "round_of_32_mapping.csv"),
        "progression": pd.read_csv(REFERENCE / "knockout_round_progression.csv"),
        "annex": pd.read_csv(REFERENCE / "third_place_assignment_annex_c.csv"),
    }


def _complete_group_scores() -> pd.DataFrame:
    """A deterministic fully-played group stage (home team wins every match)."""
    scores = pd.read_csv(ROOT / "data" / "live" / "scores_override.csv")
    scores = scores[scores["match_number"].between(1, 72)].copy()
    scores["team_a_goals"] = 2
    scores["team_b_goals"] = 0
    scores["status"] = "played"
    return scores


def test_most_probable_scoreline_favours_stronger_side():
    a, b = most_probable_scoreline(2.4, 0.4)
    assert a > b
    a, b = most_probable_scoreline(0.4, 2.4)
    assert b > a


def test_predict_match_picks_an_actual_participant(inputs):
    teams = inputs["group_view"]["team"].tolist()
    pred = predict_match(teams[0], teams[1], inputs["group_view"])
    assert SCORE_RE.match(pred["score"])
    assert pred["advancing_team"] in (teams[0], teams[1])
    # A drawn predicted scoreline must call the shoot-out; a decisive one must not.
    ga, gb = pred["score"].split("-")
    assert pred["shootout"] == (ga == gb)


def test_projected_bracket_is_complete_and_unique(inputs):
    payload = build_knockout_predictions(
        inputs["group_view"], inputs["r32_mapping"], inputs["progression"], inputs["annex"]
    )
    matches = payload["matches"]
    assert len(matches) == 32
    assert {m["match_number"] for m in matches} == set(ROUND_BY_MATCH)
    # Every match has a parseable projected score and a named advancing team.
    for m in matches:
        assert SCORE_RE.match(m["projected_score"])
        assert m["projected_advancing_team"] in (m["projected_team_a"], m["projected_team_b"])
    # The projected Round of 32 contains 32 unique teams.
    r32 = [m for m in matches if m["round"] == "R32"]
    teams = [m["projected_team_a"] for m in r32] + [m["projected_team_b"] for m in r32]
    assert len(teams) == 32 and len(set(teams)) == 32
    # There is a single projected champion (Final winner).
    final = next(m for m in matches if m["match_number"] == 104)
    assert final["projected_advancing_team"] in (final["projected_team_a"], final["projected_team_b"])


def test_next_round_is_r32_with_16_matches_before_tournament(inputs):
    payload = build_knockout_predictions(
        inputs["group_view"], inputs["r32_mapping"], inputs["progression"], inputs["annex"]
    )
    assert payload["next_round"] == "R32"
    assert payload["next_round_label"] == "Round of 32"
    nxt = payload["next_round_matches"]
    assert len(nxt) == 16
    # Pre-tournament every next-round match is projected and has copy text.
    assert all(m["status"] == "projected" for m in nxt)
    assert all(m["copy_text"] and "— adv" in m["copy_text"] for m in nxt)


def test_next_round_r32_uses_actual_teams_when_group_complete(inputs):
    scores = _complete_group_scores()
    tables = compute_group_tables(scores)
    payload = build_knockout_predictions(
        inputs["group_view"], inputs["r32_mapping"], inputs["progression"],
        inputs["annex"], scores=scores, actual_group_tables=tables,
    )
    assert payload["next_round"] == "R32"
    nxt = payload["next_round_matches"]
    assert len(nxt) == 16
    assert all(m["teams_source"] == "actual" for m in nxt)
    assert all(m["status"] == "teams_set" for m in nxt)


def test_next_round_advances_to_r16_when_all_r32_played(inputs):
    scores = _complete_group_scores()
    tables = compute_group_tables(scores)
    base = build_knockout_predictions(
        inputs["group_view"], inputs["r32_mapping"], inputs["progression"],
        inputs["annex"], scores=scores, actual_group_tables=tables,
    )
    # Play all 16 R32 matches decisively (team_a wins 2-0).
    r32_rows = []
    for m in base["matches"]:
        if m["round"] != "R32":
            continue
        r32_rows.append({
            "match_number": m["match_number"], "group": "", "date": "2026-06-30",
            "team_a": m["current_team_a"], "team_b": m["current_team_b"],
            "team_a_goals": 2, "team_b_goals": 0, "status": "played",
            "source": "test", "updated_at": "", "notes": "",
        })
    scores2 = pd.concat([scores, pd.DataFrame(r32_rows)], ignore_index=True)
    payload = build_knockout_predictions(
        inputs["group_view"], inputs["r32_mapping"], inputs["progression"],
        inputs["annex"], scores=scores2, actual_group_tables=tables,
    )
    assert payload["next_round"] == "R16"
    nxt = payload["next_round_matches"]
    assert len(nxt) == 8
    assert all(m["teams_source"] == "actual" for m in nxt)
    # Played R32 matches are no longer the round needing prediction.
    assert all(m["match_number"] >= 89 for m in nxt)


def test_next_round_keeps_played_matches_labelled_not_pending(inputs):
    scores = _complete_group_scores()
    tables = compute_group_tables(scores)
    base = build_knockout_predictions(
        inputs["group_view"], inputs["r32_mapping"], inputs["progression"],
        inputs["annex"], scores=scores, actual_group_tables=tables,
    )
    m73 = next(m for m in base["matches"] if m["match_number"] == 73)
    row = {
        "match_number": 73, "group": "", "date": "2026-06-30",
        "team_a": m73["current_team_a"], "team_b": m73["current_team_b"],
        "team_a_goals": 3, "team_b_goals": 1, "status": "played",
        "source": "test", "updated_at": "", "notes": "",
    }
    scores2 = pd.concat([scores, pd.DataFrame([row])], ignore_index=True)
    payload = build_knockout_predictions(
        inputs["group_view"], inputs["r32_mapping"], inputs["progression"],
        inputs["annex"], scores=scores2, actual_group_tables=tables,
    )
    # R32 still has unplayed matches, so it is still the next round...
    assert payload["next_round"] == "R32"
    nxt = {m["match_number"]: m for m in payload["next_round_matches"]}
    # ...and the played match is shown as played, not as pending prediction.
    assert nxt[73]["status"] == "played"
    assert nxt[73]["actual_score"] == "3-1"


def test_projection_is_deterministic(inputs):
    a = build_knockout_predictions(
        inputs["group_view"], inputs["r32_mapping"], inputs["progression"], inputs["annex"]
    )
    b = build_knockout_predictions(
        inputs["group_view"], inputs["r32_mapping"], inputs["progression"], inputs["annex"]
    )
    assert a["matches"] == b["matches"]


def test_actual_group_results_pin_round_of_32_without_touching_projection(inputs):
    scores = _complete_group_scores()
    tables = compute_group_tables(scores)
    payload = build_knockout_predictions(
        inputs["group_view"],
        inputs["r32_mapping"],
        inputs["progression"],
        inputs["annex"],
        scores=scores,
        actual_group_tables=tables,
    )
    assert payload["group_stage_complete"] is True
    assert len(payload["qualified_third_groups"]) == 8
    r32 = [m for m in payload["matches"] if m["round"] == "R32"]
    # R32 participants now come from the actual standings.
    assert all(m["teams_source"] == "actual" for m in r32)
    actual_teams = [m["current_team_a"] for m in r32] + [m["current_team_b"] for m in r32]
    assert len(set(actual_teams)) == 32


def test_played_knockout_match_pins_result_and_resolves_next_round(inputs):
    scores = _complete_group_scores()
    tables = compute_group_tables(scores)
    base = build_knockout_predictions(
        inputs["group_view"], inputs["r32_mapping"], inputs["progression"],
        inputs["annex"], scores=scores, actual_group_tables=tables,
    )
    # R16 match 89 feeds from the winners of R32 matches 74 and 77; play both
    # decisively for their actual team_a so 89's participants fully resolve.
    def _played_row(match_number):
        m = next(x for x in base["matches"] if x["match_number"] == match_number)
        return {
            "match_number": match_number, "group": "", "date": "2026-06-30",
            "team_a": m["current_team_a"], "team_b": m["current_team_b"],
            "team_a_goals": 3, "team_b_goals": 1, "status": "played",
            "source": "test", "updated_at": "", "notes": "",
        }, m["current_team_a"]

    row74, winner74 = _played_row(74)
    row77, winner77 = _played_row(77)
    scores2 = pd.concat([scores, pd.DataFrame([row74, row77])], ignore_index=True)
    payload = build_knockout_predictions(
        inputs["group_view"], inputs["r32_mapping"], inputs["progression"],
        inputs["annex"], scores=scores2, actual_group_tables=tables,
    )
    played = next(m for m in payload["matches"] if m["match_number"] == 74)
    assert played["status"] == "played"
    assert played["actual_score"] == "3-1"
    assert played["actual_advancing_team"] == winner74
    assert played["points_earned_estimate"] is not None
    # Match 89 (R16) = W74 vs W77, so it now uses the actual winners of 74 and 77.
    m89 = next(m for m in payload["matches"] if m["match_number"] == 89)
    assert {winner74, winner77} == {m89["current_team_a"], m89["current_team_b"]}
    assert m89["teams_source"] == "actual"
