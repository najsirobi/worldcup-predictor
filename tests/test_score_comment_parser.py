"""Tests for the GitHub Issue comment score parser (Travel Mode, Task D)."""

import pytest

from src.live.score_comment_parser import (
    CommentParseError,
    has_score_block,
    parse_comment,
)

VALID = """Hi team, here are today's results:

/WK-SCORES
match_number,team_a_goals,team_b_goals,status,notes
1,2,1,played,Mexico v South Africa
2,1,1,played,Korea Republic v Czechia
/END-WK-SCORES

cheers
"""


def test_valid_comment_parses_multiple_scores():
    rows = parse_comment(VALID)
    assert len(rows) == 2
    assert rows[0] == {
        "match_number": "1",
        "team_a_goals": "2",
        "team_b_goals": "1",
        "status": "played",
        "notes": "Mexico v South Africa",
    }
    assert rows[1]["match_number"] == "2"
    assert has_score_block(VALID)


def test_missing_start_marker_fails_clearly():
    body = "match_number,team_a_goals,team_b_goals,status,notes\n1,2,1,played,x\n/END-WK-SCORES"
    with pytest.raises(CommentParseError, match="start marker"):
        parse_comment(body)
    assert not has_score_block(body)


def test_missing_end_marker_fails_clearly():
    body = "/WK-SCORES\nmatch_number,team_a_goals,team_b_goals,status,notes\n1,2,1,played,x"
    with pytest.raises(CommentParseError, match="end marker"):
        parse_comment(body)


def test_invalid_header_fails_clearly():
    body = "/WK-SCORES\nmatch,a,b\n1,2,1\n/END-WK-SCORES"
    with pytest.raises(CommentParseError, match="header"):
        parse_comment(body)


def test_no_data_rows_fails_clearly():
    body = "/WK-SCORES\nmatch_number,team_a_goals,team_b_goals,status,notes\n/END-WK-SCORES"
    with pytest.raises(CommentParseError, match="No data rows"):
        parse_comment(body)


def test_notes_with_comma_is_preserved():
    body = (
        "/WK-SCORES\nmatch_number,team_a_goals,team_b_goals,status,notes\n"
        "1,2,1,played,big upset, late winner\n/END-WK-SCORES"
    )
    rows = parse_comment(body)
    assert rows[0]["notes"] == "big upset, late winner"


def test_six_column_knockout_header_parses_advanced_team():
    body = (
        "/WK-SCORES\n"
        "match_number,team_a_goals,team_b_goals,status,notes,advanced_team\n"
        "73,1,1,played,R32 penalties,Canada\n"
        "74,2,0,played,R32,\n"
        "/END-WK-SCORES"
    )
    rows = parse_comment(body)
    assert rows[0] == {
        "match_number": "73",
        "team_a_goals": "1",
        "team_b_goals": "1",
        "status": "played",
        "notes": "R32 penalties",
        "advanced_team": "Canada",
    }
    # Decisive knockout row: advanced_team may be blank.
    assert rows[1]["advanced_team"] == ""
    assert rows[1]["notes"] == "R32"


def test_six_column_notes_with_comma_keeps_advanced_team_last():
    body = (
        "/WK-SCORES\n"
        "match_number,team_a_goals,team_b_goals,status,notes,advanced_team\n"
        "73,1,1,played,tight game, on pens,Mexico\n"
        "/END-WK-SCORES"
    )
    rows = parse_comment(body)
    assert rows[0]["notes"] == "tight game, on pens"
    assert rows[0]["advanced_team"] == "Mexico"


def test_five_column_group_comment_still_backward_compatible():
    # The original 5-column format must keep working unchanged.
    rows = parse_comment(VALID)
    assert "advanced_team" not in rows[0]
    assert len(rows) == 2
