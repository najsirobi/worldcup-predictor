"""Tests for official FIFA squad PDF text parsing helpers."""

import pandas as pd

from src.ingest.fifa_squad_pdf import age_on, parse_page_text


def test_parser_handles_tiny_synthetic_squad_table_text():
    text = """
    Testland (TST)
    # POS PLAYER NAME FIRST NAME(S) LAST NAME(S) NAME ON SHIRT DOB CLUB HEIGHT (CM)
    1   GK  KEEPER Alex              Alex              KEEPER          KEEPER         10/06/2000Test FC (TST)        190
    9   FW  STRIKER Bea              Bea               STRIKER         STRIKER        12/06/2001Away FC (ENG)        175
    ROLE COACH NAME FIRST NAME(S) LAST NAME(S) NATIONALITY
    Head coach              COACH Casey                       Casey                           COACH                         Testland
    Wednesday, 3 June 2026 | 11:30 UTC | Version 1 | Page 1 / 48
    """

    parsed = parse_page_text(text, source_file="fixture.pdf")

    assert parsed.exceptions.empty
    assert len(parsed.players) == 2
    assert parsed.players.loc[0, "team"] == "Testland"
    assert parsed.players.loc[0, "fifa_code"] == "TST"
    assert parsed.players.loc[0, "position"] == "GK"
    assert parsed.players.loc[0, "player_name"] == "KEEPER Alex"
    assert parsed.players.loc[0, "date_of_birth"] == "2000-06-10"
    assert parsed.players.loc[0, "height_cm"] == 190
    assert parsed.players.loc[0, "head_coach"] == "COACH Casey"


def test_parser_handles_shifted_date_column_without_inventing_names():
    text = """
    Belgium (BEL)
    1   GK  COURTOIS Thibaut           Thibaut Nicolas MCOURTOIS                 COURTOIS               11/05/1992Real Madrid C. F. (ESP)               199
    Wednesday, 3 June 2026 | 11:30 UTC | Version 1 | Page 1 / 48
    """

    parsed = parse_page_text(text, source_file="fixture.pdf")

    assert parsed.exceptions.empty
    assert parsed.players.loc[0, "player_name"] == "COURTOIS Thibaut"
    assert parsed.players.loc[0, "height_cm"] == 199
    assert "shifted_date_column_fallback" in parsed.players.loc[0, "parse_notes"]


def test_age_computes_as_of_world_cup_opening_day():
    assert age_on("11/06/2000") == 26
    assert age_on("12/06/2000") == 25
    assert pd.isna(age_on(""))
