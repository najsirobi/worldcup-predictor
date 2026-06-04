"""Current-only WC2026 squad features from the official FIFA squad PDF."""

from __future__ import annotations

import numpy as np
import pandas as pd

POSITION_VALUES = ("GK", "DF", "MF", "FW")
TOP5_EUROPE_CODES = {"ENG", "ESP", "GER", "ITA", "FRA"}
BIG5_LEAGUE_COUNTRY_CODES = TOP5_EUROPE_CODES


def explicit_pdf_team_mapping(template_teams: set[str], squad_teams: set[str]) -> pd.DataFrame:
    """Return explicit safe mapping rows for PDF names that only differ by case."""
    rows = []
    for squad_team in sorted(squad_teams - template_teams):
        matches = [team for team in template_teams if team.casefold() == squad_team.casefold()]
        if len(matches) == 1:
            rows.append(
                {
                    "template_team_name": matches[0],
                    "fifa_pdf_team_name": squad_team,
                    "template_group": "",
                    "fifa_code": "",
                    "match_method": "casefold_exact_connector_capitalization",
                    "confidence": 1.0,
                    "needs_review": False,
                    "evidence": "Only capitalization differs between official PDF team name and FIF8A template team name.",
                    "notes": "Safe explicit mapping for production join; no fuzzy matching used.",
                }
            )
    return pd.DataFrame(rows)


def apply_explicit_team_mapping(squads: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    frame = squads.copy()
    rename = dict(zip(mapping["fifa_pdf_team_name"], mapping["template_team_name"], strict=False)) if not mapping.empty else {}
    frame["team_original_pdf"] = frame["team"]
    frame["team"] = frame["team"].replace(rename)
    return frame


def build_template_join_table(template: pd.DataFrame, squads: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    template_teams = sorted(set(template["team_a"]) | set(template["team_b"]))
    squad_teams = set(squads["team"].dropna())
    mapping = explicit_pdf_team_mapping(set(template_teams), squad_teams)
    group_lookup = {}
    for team in template_teams:
        group_values = template.loc[(template["team_a"] == team) | (template["team_b"] == team), "group"].dropna().unique()
        group_lookup[team] = group_values[0] if len(group_values) else ""
    if not mapping.empty:
        mapping["template_group"] = mapping["template_team_name"].map(group_lookup)
        code_lookup = squads.drop_duplicates("team").set_index("team")["fifa_code"].to_dict()
        mapping["fifa_code"] = mapping["fifa_pdf_team_name"].map(code_lookup)

    mapped = apply_explicit_team_mapping(squads, mapping)
    counts = mapped.groupby("team", dropna=False).agg(
        squad_rows=("player_name", "count"),
        fifa_codes=("fifa_code", lambda values: ", ".join(sorted(set(values.dropna().astype(str))))),
    )
    rows = []
    for team in template_teams:
        rows.append(
            {
                "template_team_name": team,
                "template_group": group_lookup[team],
                "squad_rows_found": int(counts.loc[team, "squad_rows"]) if team in counts.index else 0,
                "fifa_codes": counts.loc[team, "fifa_codes"] if team in counts.index else "",
                "has_official_squad_coverage": bool(team in counts.index and counts.loc[team, "squad_rows"] > 0),
            }
        )
    return pd.DataFrame(rows), mapping


def aggregate_wc2026_squad_features(squads: pd.DataFrame) -> pd.DataFrame:
    required = {"team", "fifa_code", "player_name", "position", "age_on_2026_06_11", "height_cm", "club_country_code"}
    missing = required - set(squads.columns)
    if missing:
        raise ValueError(f"Missing required WC2026 squad columns: {sorted(missing)}")

    frame = squads.copy()
    frame["_age"] = pd.to_numeric(frame["age_on_2026_06_11"], errors="coerce")
    frame["_height"] = pd.to_numeric(frame["height_cm"], errors="coerce")
    frame["_club_country"] = frame["club_country_code"].fillna("").astype(str)

    rows = []
    for (team, fifa_code), group in frame.groupby(["team", "fifa_code"], dropna=False):
        player_count = int(group["player_name"].nunique(dropna=True))
        position_counts = group["position"].value_counts()
        club_country_known = group["_club_country"].ne("")
        domestic = group["_club_country"].eq(str(fifa_code))
        foreign = club_country_known & ~domestic
        top5 = group["_club_country"].isin(TOP5_EUROPE_CODES)
        rows.append(
            {
                "team": team,
                "fifa_code": fifa_code,
                "squad_player_count": player_count,
                "squad_avg_age": float(group["_age"].mean()) if group["_age"].notna().any() else np.nan,
                "squad_median_age": float(group["_age"].median()) if group["_age"].notna().any() else np.nan,
                "squad_age_std": float(group["_age"].std(ddof=0)) if group["_age"].notna().any() else np.nan,
                "squad_avg_height_cm": float(group["_height"].mean()) if group["_height"].notna().any() else np.nan,
                "squad_median_height_cm": float(group["_height"].median()) if group["_height"].notna().any() else np.nan,
                "squad_gk_count": int(position_counts.get("GK", 0)),
                "squad_df_count": int(position_counts.get("DF", 0)),
                "squad_mf_count": int(position_counts.get("MF", 0)),
                "squad_fw_count": int(position_counts.get("FW", 0)),
                "squad_fw_share": float(position_counts.get("FW", 0) / player_count) if player_count else np.nan,
                "squad_defensive_share": float((position_counts.get("GK", 0) + position_counts.get("DF", 0)) / player_count) if player_count else np.nan,
                "squad_midfield_share": float(position_counts.get("MF", 0) / player_count) if player_count else np.nan,
                "squad_domestic_club_share": float(domestic.mean()) if club_country_known.any() else np.nan,
                "squad_foreign_club_share": float(foreign.mean()) if club_country_known.any() else np.nan,
                "squad_top5_europe_club_share": float(top5.mean()) if club_country_known.any() else np.nan,
                "squad_big5_league_country_share": float(top5.mean()) if club_country_known.any() else np.nan,
                "squad_club_country_diversity": int(group.loc[club_country_known, "_club_country"].nunique()),
                "squad_oldest_player_age": float(group["_age"].max()) if group["_age"].notna().any() else np.nan,
                "squad_youngest_player_age": float(group["_age"].min()) if group["_age"].notna().any() else np.nan,
                "squad_has_official_pdf_data": True,
            }
        )
    return pd.DataFrame(rows).sort_values("team").reset_index(drop=True)
