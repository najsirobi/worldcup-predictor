"""Strict player matching helpers for official squads and Transfermarkt data."""

from __future__ import annotations

import re
import unicodedata

import pandas as pd


def normalized_player_key(value: object) -> str:
    text = "" if value is None or pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def official_name_keys(row: pd.Series) -> set[str]:
    keys = {normalized_player_key(row.get("player_name", ""))}
    first = str(row.get("first_names", "") or "").strip()
    last = str(row.get("last_names", "") or "").strip()
    if first and last:
        keys.add(normalized_player_key(f"{first} {last}"))
    return {key for key in keys if key}


def match_official_to_transfermarkt(official: pd.DataFrame, transfermarkt_players: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Auto-match only unique exact-name-key + exact-DOB Transfermarkt candidates."""
    required_official = {"team", "player_name", "date_of_birth"}
    required_tm = {"player_id", "name", "date_of_birth"}
    if missing := required_official - set(official.columns):
        raise ValueError(f"Missing official player columns: {sorted(missing)}")
    if missing := required_tm - set(transfermarkt_players.columns):
        raise ValueError(f"Missing Transfermarkt player columns: {sorted(missing)}")

    tm = transfermarkt_players.copy()
    tm["_dob"] = pd.to_datetime(tm["date_of_birth"], errors="coerce").dt.date.astype("string")
    tm["_name_key"] = tm["name"].map(normalized_player_key)
    grouped = tm.groupby(["_dob", "_name_key"], dropna=False)

    enriched_rows = []
    candidate_rows = []
    for official_idx, row in official.reset_index(drop=True).iterrows():
        dob = pd.to_datetime(row["date_of_birth"], errors="coerce")
        dob_key = dob.date().isoformat() if not pd.isna(dob) else ""
        keys = official_name_keys(row)
        candidate_indexes: set[int] = set()
        for key in keys:
            if (dob_key, key) in grouped.groups:
                candidate_indexes.update(grouped.groups[(dob_key, key)].tolist())
        candidates = tm.loc[sorted(candidate_indexes)] if candidate_indexes else pd.DataFrame()

        enriched = row.to_dict()
        enriched["transfermarkt_match_status"] = "unmatched"
        enriched["transfermarkt_match_method"] = ""
        enriched["transfermarkt_match_evidence"] = ""
        if len(candidates) == 1:
            candidate = candidates.iloc[0]
            for column in [
                "player_id",
                "name",
                "current_club_name",
                "market_value_in_eur",
                "highest_market_value_in_eur",
                "position",
                "sub_position",
                "country_of_citizenship",
                "last_season",
            ]:
                enriched[f"transfermarkt_{column}"] = candidate.get(column, pd.NA)
            enriched["transfermarkt_match_status"] = "accepted"
            enriched["transfermarkt_match_method"] = "exact_normalized_name_and_dob"
            enriched["transfermarkt_match_evidence"] = f"Official DOB {dob_key}; official keys {sorted(keys)}; Transfermarkt name {candidate.get('name', '')}."
        else:
            for column in [
                "player_id",
                "name",
                "current_club_name",
                "market_value_in_eur",
                "highest_market_value_in_eur",
                "position",
                "sub_position",
                "country_of_citizenship",
                "last_season",
            ]:
                enriched[f"transfermarkt_{column}"] = pd.NA
            if len(candidates) > 1:
                enriched["transfermarkt_match_status"] = "ambiguous"
                for _, candidate in candidates.iterrows():
                    candidate_rows.append(
                        {
                            "official_team": row.get("team", ""),
                            "official_player_name": row.get("player_name", ""),
                            "official_date_of_birth": dob_key,
                            "transfermarkt_player_id": candidate.get("player_id", pd.NA),
                            "transfermarkt_name": candidate.get("name", ""),
                            "transfermarkt_date_of_birth": candidate.get("date_of_birth", ""),
                            "match_method": "exact_normalized_name_and_dob",
                            "confidence": 0.0,
                            "needs_review": True,
                            "evidence": "Multiple Transfermarkt players matched the same exact normalized name and DOB.",
                        }
                    )
        enriched_rows.append(enriched)

    return pd.DataFrame(enriched_rows), pd.DataFrame(candidate_rows)
