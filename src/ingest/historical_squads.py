"""Historical World Cup squad ingestion into a standard, comparable schema.

Produces one row per (tournament, team, player) with explicit team mappings and
no invented values. Missing fields (club, club_country, height) stay null rather
than zero. Two sources are supported:

- ``world_cup_database`` (Joshua Fjelstul) — official-derived squad lists with
  position, date of birth and coach (manager) appointments, covering 1930-2018.
  Contains no club / club-country / height information.
- ``world_cup_2022_player_data`` (swaptr/FBref) — 2022 appearance-based player
  table with position and age, covering 2022. No DOB / club / club-country /
  height either.

The schema mirrors the columns available for the WC2026 official squad parse so
that only genuinely comparable features are built downstream.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

STANDARD_COLUMNS = [
    "tournament_year",
    "team",
    "raw_team_name",
    "canonical_team_name",
    "country_code",
    "player_name",
    "position",
    "date_of_birth",
    "age_at_tournament_start",
    "club",
    "club_country",
    "height_cm",
    "coach_name",
    "source",
    "source_file",
    "source_quality",
    "parse_notes",
]

# world_cup_database position_code -> canonical position bucket (matches WC2026)
WCD_POSITION_MAP = {"GK": "GK", "DF": "DF", "MF": "MF", "FW": "FW"}


def _completed_age(birth: pd.Series, ref: pd.Series) -> pd.Series:
    """Completed years between birth date and reference date (null-safe)."""
    birth = pd.to_datetime(birth, errors="coerce")
    ref = pd.to_datetime(ref, errors="coerce")
    years = (ref - birth).dt.days / 365.25
    return years.apply(lambda v: float(np.floor(v)) if pd.notna(v) else np.nan)


def world_cup_database_squads(raw_dir: str) -> pd.DataFrame:
    """Load 1930-2018 squad lists from the world_cup_database dataset."""
    raw = Path(raw_dir)
    squads = pd.read_csv(raw / "squads.csv")
    players = pd.read_csv(raw / "players.csv")[["player_id", "birth_date"]]
    tournaments = pd.read_csv(raw / "tournaments.csv")[["tournament_name", "year", "start_date"]]
    appointments = pd.read_csv(raw / "manager_appointments.csv")

    merged = squads.merge(players, on="player_id", how="left")
    merged = merged.merge(tournaments, on="tournament_name", how="left")

    # One coach per tournament/team (manager_appointments is the pre-tournament
    # appointment table). Concatenate co-managers deterministically.
    appointments["coach_name"] = (
        appointments["given_name"].fillna("") + " " + appointments["family_name"].fillna("")
    ).str.strip()
    coach = (
        appointments.groupby(["tournament_name", "team_name"])["coach_name"]
        .apply(lambda s: " / ".join(sorted({v for v in s if v})))
        .reset_index()
    )
    merged = merged.merge(coach, on=["tournament_name", "team_name"], how="left")

    age = _completed_age(merged["birth_date"], merged["start_date"])
    out = pd.DataFrame(
        {
            "tournament_year": merged["year"].astype("Int64"),
            "team": merged["team_name"],
            "raw_team_name": merged["team_name"],
            "canonical_team_name": merged["team_name"],
            "country_code": merged["team_code"],
            "player_name": (
                merged["given_name"].fillna("") + " " + merged["family_name"].fillna("")
            ).str.strip(),
            "position": merged["position_code"].map(WCD_POSITION_MAP),
            "date_of_birth": pd.to_datetime(merged["birth_date"], errors="coerce"),
            "age_at_tournament_start": age,
            "club": pd.NA,
            "club_country": pd.NA,
            "height_cm": pd.NA,
            "coach_name": merged["coach_name"],
            "source": "world_cup_database",
            "source_file": "data/raw/kaggle/world_cup_database/squads.csv",
            "source_quality": "official-derived; full named squads, position+DOB+coach; no club/height",
            "parse_notes": "age=completed years at tournament start_date; club/club_country/height unavailable in source",
        }
    )
    return out


def world_cup_2022_squads(raw_dir: str) -> pd.DataFrame:
    """Load 2022 player table from the world_cup_2022_player_data dataset."""
    raw = Path(raw_dir)
    pt = pd.read_csv(raw / "player_playingtime.csv")
    # FBref encodes age as "years-days" (e.g. "30-067") as of the tournament.
    # Take completed years to match the WC2026 completed-age convention.
    age_years = pd.to_numeric(
        pt["age"].astype("string").str.split("-").str[0], errors="coerce"
    )
    out = pd.DataFrame(
        {
            "tournament_year": pd.array([2022] * len(pt), dtype="Int64"),
            "team": pt["team"],
            "raw_team_name": pt["team"],
            "canonical_team_name": pt["team"],
            "country_code": pd.NA,
            "player_name": pt["player"],
            "position": pt["position"].map(WCD_POSITION_MAP),
            "date_of_birth": pd.NaT,
            "age_at_tournament_start": age_years,
            "club": pd.NA,
            "club_country": pd.NA,
            "height_cm": pd.NA,
            "coach_name": pd.NA,
            "source": "world_cup_2022_player_data",
            "source_file": "data/raw/kaggle/world_cup_2022_player_data/player_playingtime.csv",
            "source_quality": "appearance-based (players who featured); position+age; no DOB/club/height/coach",
            "parse_notes": "age provided by source; DOB unavailable (birth_year only); roster is appearance-based not the full 26-man list",
        }
    )
    return out


def build_historical_squads(wcd_dir: str, wc2022_dir: str) -> pd.DataFrame:
    """Concatenate all available historical squad sources into the standard schema."""
    frames = []
    if (Path(wcd_dir) / "squads.csv").exists():
        frames.append(world_cup_database_squads(wcd_dir))
    if (Path(wc2022_dir) / "player_playingtime.csv").exists():
        frames.append(world_cup_2022_squads(wc2022_dir))
    if not frames:
        raise FileNotFoundError("No historical squad sources found.")
    out = pd.concat(frames, ignore_index=True)
    return out[STANDARD_COLUMNS]
