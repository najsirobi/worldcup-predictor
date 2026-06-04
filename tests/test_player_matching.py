"""Tests for strict official-squad to Transfermarkt player matching."""

import pandas as pd

from src.ingest.player_matching import match_official_to_transfermarkt


def test_exact_name_and_dob_match_is_accepted():
    official = pd.DataFrame(
        [
            {
                "team": "A",
                "player_name": "PLAYER Alex",
                "first_names": "Alex",
                "last_names": "PLAYER",
                "date_of_birth": "2000-01-01",
            }
        ]
    )
    transfermarkt = pd.DataFrame(
        [
            {
                "player_id": 1,
                "name": "Alex Player",
                "date_of_birth": "2000-01-01",
                "market_value_in_eur": 123,
            }
        ]
    )

    enriched, candidates = match_official_to_transfermarkt(official, transfermarkt)

    assert candidates.empty
    assert enriched.loc[0, "transfermarkt_match_status"] == "accepted"
    assert enriched.loc[0, "transfermarkt_market_value_in_eur"] == 123


def test_ambiguous_player_matches_are_not_auto_accepted():
    official = pd.DataFrame(
        [
            {
                "team": "A",
                "player_name": "Alex Player",
                "first_names": "",
                "last_names": "",
                "date_of_birth": "2000-01-01",
            }
        ]
    )
    transfermarkt = pd.DataFrame(
        [
            {"player_id": 1, "name": "Alex Player", "date_of_birth": "2000-01-01"},
            {"player_id": 2, "name": "Alex Player", "date_of_birth": "2000-01-01"},
        ]
    )

    enriched, candidates = match_official_to_transfermarkt(official, transfermarkt)

    assert enriched.loc[0, "transfermarkt_match_status"] == "ambiguous"
    assert pd.isna(enriched.loc[0, "transfermarkt_player_id"])
    assert len(candidates) == 2
    assert candidates["needs_review"].all()


def test_missing_market_values_remain_null_not_fake_zero():
    official = pd.DataFrame(
        [
            {
                "team": "A",
                "player_name": "Alex Player",
                "first_names": "",
                "last_names": "",
                "date_of_birth": "2000-01-01",
            }
        ]
    )
    transfermarkt = pd.DataFrame(
        [{"player_id": 1, "name": "Alex Player", "date_of_birth": "2000-01-01"}]
    )

    enriched, _ = match_official_to_transfermarkt(official, transfermarkt)

    assert enriched.loc[0, "transfermarkt_match_status"] == "accepted"
    assert pd.isna(enriched.loc[0, "transfermarkt_market_value_in_eur"])
