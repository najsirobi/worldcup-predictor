"""Tests for the manual live score override (Travel Mode, Task A)."""

import pandas as pd
import pytest

from src.live.scores_override import (
    EXPECTED_MATCH_COUNT,
    KNOCKOUT_MATCH_NUMBERS,
    TOTAL_MATCH_COUNT,
    build_initial_override,
    update_match,
    validate_override,
)


def test_scores_override_initialises_to_group_plus_knockout_rows():
    frame = build_initial_override()
    # 72 group matches + 32 knockout placeholders = 104 total.
    assert len(frame) == TOTAL_MATCH_COUNT == 104
    assert (frame["match_number"] <= EXPECTED_MATCH_COUNT).sum() == EXPECTED_MATCH_COUNT == 72
    assert frame["match_number"].isin(KNOCKOUT_MATCH_NUMBERS).sum() == 32
    assert frame["match_number"].is_unique
    assert (frame["status"] == "scheduled").all()
    assert frame["team_a_goals"].isna().all()
    assert frame["team_b_goals"].isna().all()
    # Knockout placeholders carry no team names until their feeders resolve.
    knockout = frame[frame["match_number"].isin(KNOCKOUT_MATCH_NUMBERS)]
    assert (knockout["team_a"] == "").all()
    assert (knockout["team_b"] == "").all()
    assert "advanced_team" in frame.columns


def test_valid_score_update_changes_one_row():
    frame = build_initial_override()
    updated = update_match(frame, match_number=1, team_a_goals=2, team_b_goals=1)

    changed = updated[updated["status"] == "played"]
    assert len(changed) == 1
    row = changed.iloc[0]
    assert row["match_number"] == 1
    assert int(row["team_a_goals"]) == 2
    assert int(row["team_b_goals"]) == 1
    assert row["updated_at"]  # timestamp populated
    # All other rows untouched.
    assert (updated.loc[updated["match_number"] != 1, "status"] == "scheduled").all()


def test_invalid_match_number_fails_clearly():
    frame = build_initial_override()
    with pytest.raises(ValueError, match="does not exist"):
        update_match(frame, match_number=999, team_a_goals=1, team_b_goals=0)


def test_played_without_goals_fails():
    frame = build_initial_override()
    with pytest.raises(ValueError, match="requires both"):
        update_match(frame, match_number=5, status="played")


def test_negative_goals_fail():
    frame = build_initial_override()
    with pytest.raises(ValueError):
        update_match(frame, match_number=5, team_a_goals=-1, team_b_goals=0)


def test_no_duplicate_match_rows_detected():
    frame = build_initial_override()
    dup = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate match rows"):
        validate_override(dup)


def test_clearing_result_wipes_goals():
    frame = build_initial_override()
    played = update_match(frame, match_number=3, team_a_goals=1, team_b_goals=1)
    cleared = update_match(played, match_number=3, status="scheduled")
    row = cleared[cleared["match_number"] == 3].iloc[0]
    assert row["status"] == "scheduled"
    assert pd.isna(row["team_a_goals"])
    assert pd.isna(row["team_b_goals"])
