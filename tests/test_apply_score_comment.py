"""End-to-end tests for applying a comment to the override (Travel Mode, Task D)."""

import pytest

from src.live.batch_update import BatchValidationError, apply_batch
from src.live.score_comment_parser import CommentParseError, parse_comment
from src.live.scores_override import build_initial_override, load_override, write_override

VALID = """/WK-SCORES
match_number,team_a_goals,team_b_goals,status,notes
1,2,1,played,Mexico v South Africa
2,1,1,played,Korea Republic v Czechia
/END-WK-SCORES"""


def _seed_override(tmp_path):
    path = tmp_path / "scores_override.csv"
    write_override(build_initial_override(), path)
    return path


def test_comment_round_trip_applies_and_persists(tmp_path):
    path = _seed_override(tmp_path)
    frame = load_override(path)
    rows = parse_comment(VALID)
    updated, applied = apply_batch(frame, rows, source="github_issue_comment")
    write_override(updated, path)

    reloaded = load_override(path)
    played = reloaded[reloaded["status"] == "played"]
    assert sorted(played["match_number"]) == [1, 2]
    assert (played["source"] == "github_issue_comment").all()
    assert len(applied) == 2


def test_comment_with_bad_row_applies_nothing(tmp_path):
    path = _seed_override(tmp_path)
    frame = load_override(path)
    bad = VALID.replace("2,1,1,played", "2,,,played")  # missing goals
    rows = parse_comment(bad)
    with pytest.raises(BatchValidationError):
        apply_batch(frame, rows, source="github_issue_comment")
    # File on disk is unchanged: still no played matches.
    assert (load_override(path)["status"] == "scheduled").all()


def test_comment_missing_markers_never_reaches_apply():
    with pytest.raises(CommentParseError):
        parse_comment("no score block here")
