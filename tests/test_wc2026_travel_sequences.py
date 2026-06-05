from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from scripts.build_wc2026_travel_sequences import build_travel_sequences


ROOT = Path(__file__).resolve().parents[1]


def _match_rows() -> pd.DataFrame:
    rows = [
        {
            "match_number": 1,
            "date": "2026-06-11",
            "round": "Group Stage",
            "group": "A",
            "team_a": "Alpha",
            "team_b": "Beta",
            "team_a_source": "Alpha",
            "team_b_source": "Beta",
            "venue_id": "mexico_city_stadium",
            "stadium": "Mexico City Stadium",
            "host_city": "Mexico City",
            "host_country": "Mexico",
            "actual_country": "Mexico",
            "latitude": 19.3029,
            "longitude": -99.1505,
            "elevation_m": 2240.0,
            "timezone": "America/Mexico_City",
            "needs_review": False,
            "notes": "",
        },
        {
            "match_number": 2,
            "date": "2026-06-12",
            "round": "Group Stage",
            "group": "A",
            "team_a": "Gamma",
            "team_b": "Delta",
            "team_a_source": "Gamma",
            "team_b_source": "Delta",
            "venue_id": "guadalajara_stadium",
            "stadium": "Estadio Guadalajara",
            "host_city": "Guadalajara",
            "host_country": "Mexico",
            "actual_country": "Mexico",
            "latitude": 20.6739,
            "longitude": -103.4586,
            "elevation_m": 1560.0,
            "timezone": "America/Mexico_City",
            "needs_review": False,
            "notes": "",
        },
        {
            "match_number": 25,
            "date": "2026-06-18",
            "round": "Group Stage",
            "group": "A",
            "team_a": "Alpha",
            "team_b": "Gamma",
            "team_a_source": "Alpha",
            "team_b_source": "Gamma",
            "venue_id": "new_york_new_jersey_stadium",
            "stadium": "New York New Jersey Stadium",
            "host_city": "New York New Jersey",
            "host_country": "USA",
            "actual_country": "USA",
            "latitude": 40.8135,
            "longitude": -74.0745,
            "elevation_m": 10.0,
            "timezone": "America/New_York",
            "needs_review": False,
            "notes": "",
        },
        {
            "match_number": 53,
            "date": "2026-06-25",
            "round": "Group Stage",
            "group": "A",
            "team_a": "Alpha",
            "team_b": "Delta",
            "team_a_source": "Alpha",
            "team_b_source": "Delta",
            "venue_id": "new_york_new_jersey_stadium",
            "stadium": "New York New Jersey Stadium",
            "host_city": "New York New Jersey",
            "host_country": "USA",
            "actual_country": "USA",
            "latitude": 40.8135,
            "longitude": -74.0745,
            "elevation_m": 10.0,
            "timezone": "America/New_York",
            "needs_review": False,
            "notes": "",
        },
        {
            "match_number": 73,
            "date": "2026-06-28",
            "round": "Round of 32",
            "group": None,
            "team_a": "2A",
            "team_b": "2B",
            "team_a_source": "2A",
            "team_b_source": "2B",
            "venue_id": "dallas_stadium",
            "stadium": "Dallas Stadium",
            "host_city": "Dallas",
            "host_country": "USA",
            "actual_country": "USA",
            "latitude": 32.7473,
            "longitude": -97.0945,
            "elevation_m": 170.0,
            "timezone": "America/Chicago",
            "needs_review": False,
            "notes": "",
        },
    ]
    return pd.DataFrame(rows)


def test_first_match_has_no_previous_match_burden():
    sequence = build_travel_sequences(_match_rows(), projected_knockout=None)
    alpha_first = sequence[(sequence["team"] == "Alpha") & (sequence["match_number"] == 1)].iloc[0]

    assert pd.isna(alpha_first["previous_match_number"])
    assert pd.isna(alpha_first["rest_days"])
    assert alpha_first["travel_burden_score"] == 0


def test_distance_calculation_is_positive_and_plausible():
    sequence = build_travel_sequences(_match_rows(), projected_knockout=None)
    alpha_second = sequence[(sequence["team"] == "Alpha") & (sequence["match_number"] == 25)].iloc[0]

    assert 3300 <= alpha_second["distance_from_previous_km"] <= 3500


def test_timezone_and_elevation_delta_are_calculated_correctly():
    sequence = build_travel_sequences(_match_rows(), projected_knockout=None)
    alpha_second = sequence[(sequence["team"] == "Alpha") & (sequence["match_number"] == 25)].iloc[0]

    assert alpha_second["timezone_delta_hours"] == 2
    assert alpha_second["elevation_delta_m"] == -2230


def test_unresolved_knockout_participants_stay_pending():
    sequence = build_travel_sequences(_match_rows(), projected_knockout=None)
    pending = sequence[sequence["match_number"] == 73]

    assert set(pending["team"]) == {"2A", "2B"}
    assert pending["needs_review"].all()
    assert pending["notes"].str.contains("pending").all()


def test_final_candidate_v2_auto_science_is_not_modified_unless_v3_is_created():
    manifest = json.loads(
        (ROOT / "outputs" / "final_candidate_v2_auto_science" / "FROZEN_MANIFEST.json").read_text()
    )
    for item in manifest["files"]:
        path = ROOT / item["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]

    assert not (ROOT / "outputs" / "final_candidate_v3_travel_burden").exists()
