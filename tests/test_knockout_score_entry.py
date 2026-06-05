"""Tests for entering knockout (73-104) actual results into the score override."""

import pandas as pd
import pytest

from src.live.batch_update import apply_batch
from src.live.scores_override import (
    BASE_COLUMNS,
    KNOCKOUT_MATCH_NUMBERS,
    build_initial_override,
    ensure_knockout_rows,
    load_override,
    update_match,
    validate_override,
    write_override,
)


def test_group_match_1_to_72_still_updates():
    frame = build_initial_override()
    updated = update_match(frame, match_number=1, team_a_goals=2, team_b_goals=1)
    row = updated.loc[updated["match_number"] == 1].iloc[0]
    assert row["status"] == "played"
    assert int(row["team_a_goals"]) == 2 and int(row["team_b_goals"]) == 1


def test_decisive_knockout_match_updates_without_advanced_team():
    frame = build_initial_override()
    updated = update_match(frame, match_number=73, team_a_goals=2, team_b_goals=0)
    row = updated.loc[updated["match_number"] == 73].iloc[0]
    assert row["status"] == "played"
    assert int(row["team_a_goals"]) == 2 and int(row["team_b_goals"]) == 0
    assert row["advanced_team"] == ""


def test_level_knockout_requires_advanced_team():
    frame = build_initial_override()
    # Give the row teams so advanced_team can be validated against them.
    frame.loc[frame["match_number"] == 73, ["team_a", "team_b"]] = ["Canada", "Mexico"]
    with pytest.raises(ValueError, match="level"):
        update_match(frame, match_number=73, team_a_goals=1, team_b_goals=1)
    ok = update_match(
        frame, match_number=73, team_a_goals=1, team_b_goals=1, advanced_team="Canada"
    )
    row = ok.loc[ok["match_number"] == 73].iloc[0]
    assert row["advanced_team"] == "Canada"


def test_advanced_team_must_be_a_participant_when_teams_known():
    frame = build_initial_override()
    frame.loc[frame["match_number"] == 73, ["team_a", "team_b"]] = ["Canada", "Mexico"]
    with pytest.raises(ValueError, match="must be one of"):
        update_match(
            frame, match_number=73, team_a_goals=1, team_b_goals=1, advanced_team="Brazil"
        )


def test_advanced_team_rejected_on_group_match():
    frame = build_initial_override()
    with pytest.raises(ValueError, match="only valid for knockout"):
        update_match(
            frame, match_number=1, team_a_goals=1, team_b_goals=0, advanced_team="Mexico"
        )


def test_load_override_migrates_old_group_only_file(tmp_path):
    # An old 72-row file without advanced_team and without knockout rows.
    legacy = build_initial_override()
    legacy = legacy[legacy["match_number"] <= 72][BASE_COLUMNS]
    path = tmp_path / "scores_override.csv"
    legacy.to_csv(path, index=False)

    loaded = load_override(path)
    assert "advanced_team" in loaded.columns
    assert len(loaded) == 104
    assert loaded["match_number"].isin(KNOCKOUT_MATCH_NUMBERS).sum() == 32
    validate_override(loaded)


def test_ensure_knockout_rows_is_idempotent():
    frame = build_initial_override()
    again = ensure_knockout_rows(frame)
    assert len(again) == len(frame) == 104


def test_batch_apply_accepts_knockout_with_advanced_team():
    frame = build_initial_override()
    frame.loc[frame["match_number"] == 73, ["team_a", "team_b"]] = ["Canada", "Mexico"]
    rows = [
        {
            "match_number": "73",
            "team_a_goals": "1",
            "team_b_goals": "1",
            "status": "played",
            "notes": "R32 pens",
            "advanced_team": "Canada",
        }
    ]
    updated, applied = apply_batch(frame, rows, source="test")
    assert applied[0]["match_number"] == 73
    row = updated.loc[updated["match_number"] == 73].iloc[0]
    assert row["advanced_team"] == "Canada"
