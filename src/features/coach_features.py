"""Coach feature construction with strict pre-match history."""

from __future__ import annotations

import pandas as pd


COACH_FEATURE_COLUMNS = [
    "coach_name",
    "coach_tenure_days",
    "coach_matches_before_match",
    "coach_winrate_before_match",
    "coach_goal_diff_per_match_before_match",
    "prior_world_cup_experience",
    "prior_international_tournament_experience",
    "recent_coach_change_flag",
    "has_coach_features",
]


def normalize_coach_name(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).replace("\xa0", " ").split())


def build_coach_match_features(appearances: pd.DataFrame) -> pd.DataFrame:
    """Return one row per tournament/team/match using only previous appearances."""
    required = {
        "tournament_name",
        "team_name",
        "match_date",
        "match_id",
        "coach_name",
        "goals_for",
        "goals_against",
        "is_win",
    }
    missing = required - set(appearances.columns)
    if missing:
        raise ValueError(f"Missing required coach columns: {sorted(missing)}")

    frame = appearances.copy()
    frame["match_date"] = pd.to_datetime(frame["match_date"])
    frame["coach_name"] = frame["coach_name"].map(normalize_coach_name)
    frame = frame.sort_values(["team_name", "coach_name", "match_date", "match_id"]).reset_index(drop=True)
    frame["goal_diff"] = frame["goals_for"] - frame["goals_against"]

    rows = []
    for _, row in frame.iterrows():
        previous_team_coach = frame[
            (frame["team_name"] == row["team_name"])
            & (frame["coach_name"] == row["coach_name"])
            & (frame["match_date"] < row["match_date"])
        ]
        previous_team = frame[
            (frame["team_name"] == row["team_name"])
            & (frame["match_date"] < row["match_date"])
        ]
        first_seen = previous_team_coach["match_date"].min()
        tenure = (row["match_date"] - first_seen).days if pd.notna(first_seen) else 0
        last_team_match = previous_team.sort_values("match_date").tail(1)
        recent_change = False
        if not last_team_match.empty:
            recent_change = normalize_coach_name(last_team_match.iloc[0]["coach_name"]) != row["coach_name"]

        rows.append(
            {
                "tournament_name": row["tournament_name"],
                "team": row["team_name"],
                "match_id": row["match_id"],
                "match_date": row["match_date"],
                "coach_name": row["coach_name"],
                "coach_tenure_days": tenure,
                "coach_matches_before_match": int(len(previous_team_coach)),
                "coach_winrate_before_match": float(previous_team_coach["is_win"].mean()) if len(previous_team_coach) else pd.NA,
                "coach_goal_diff_per_match_before_match": float(previous_team_coach["goal_diff"].mean()) if len(previous_team_coach) else pd.NA,
                "prior_world_cup_experience": int(len(previous_team_coach) > 0),
                "prior_international_tournament_experience": int(len(previous_team_coach) > 0),
                "recent_coach_change_flag": bool(recent_change),
                "has_coach_features": bool(row["coach_name"]),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    grouped = []
    for keys, group in out.groupby(["tournament_name", "team", "match_date"], dropna=False):
        row = {
            "tournament_name": keys[0],
            "team": keys[1],
            "match_date": keys[2],
            "match_id": " / ".join(map(str, group["match_id"].dropna().unique())),
            "coach_name": " / ".join(sorted(set(group["coach_name"].dropna()))),
            "coach_tenure_days": group["coach_tenure_days"].max(),
            "coach_matches_before_match": group["coach_matches_before_match"].max(),
            "coach_winrate_before_match": group["coach_winrate_before_match"].dropna().mean() if group["coach_winrate_before_match"].notna().any() else pd.NA,
            "coach_goal_diff_per_match_before_match": group["coach_goal_diff_per_match_before_match"].dropna().mean() if group["coach_goal_diff_per_match_before_match"].notna().any() else pd.NA,
            "prior_world_cup_experience": int(group["prior_world_cup_experience"].max()),
            "prior_international_tournament_experience": int(group["prior_international_tournament_experience"].max()),
            "recent_coach_change_flag": bool(group["recent_coach_change_flag"].max()),
            "has_coach_features": bool(group["has_coach_features"].max()),
        }
        grouped.append(row)
    return pd.DataFrame(grouped)


def world_cup_database_coach_appearances(raw_dir: str) -> pd.DataFrame:
    managers = pd.read_csv(f"{raw_dir}/manager_appearances.csv")
    matches = pd.read_csv(f"{raw_dir}/matches.csv")
    match_cols = matches[
        [
            "match_id",
            "home_team_name",
            "away_team_name",
            "home_team_score",
            "away_team_score",
        ]
    ]
    frame = managers.merge(match_cols, on="match_id", how="left")
    is_home = frame["team_name"].eq(frame["home_team_name"])
    goals_for = frame["home_team_score"].where(is_home, frame["away_team_score"])
    goals_against = frame["away_team_score"].where(is_home, frame["home_team_score"])
    return pd.DataFrame(
        {
            "tournament_name": frame["tournament_name"],
            "team_name": frame["team_name"],
            "match_date": frame["match_date"],
            "match_id": frame["match_id"],
            "coach_name": (frame["given_name"].fillna("") + " " + frame["family_name"].fillna("")).str.strip(),
            "goals_for": goals_for,
            "goals_against": goals_against,
            "is_win": goals_for > goals_against,
            "source": "world_cup_database",
        }
    )


def world_cup_history_2022_coach_appearances(raw_dir: str) -> pd.DataFrame:
    matches = pd.read_csv(f"{raw_dir}/matches_1930_2022.csv")
    matches = matches[matches["Year"].eq(2022)].copy()
    matches["Date"] = pd.to_datetime(matches["Date"])
    rows = []
    for _, row in matches.iterrows():
        rows.append(
            {
                "tournament_name": "2022 FIFA World Cup",
                "team_name": row["home_team"],
                "match_date": row["Date"],
                "match_id": f"2022-{row.name}-home",
                "coach_name": normalize_coach_name(row["home_manager"]),
                "goals_for": row["home_score"],
                "goals_against": row["away_score"],
                "is_win": row["home_score"] > row["away_score"],
                "source": "world_cup_history",
            }
        )
        rows.append(
            {
                "tournament_name": "2022 FIFA World Cup",
                "team_name": row["away_team"],
                "match_date": row["Date"],
                "match_id": f"2022-{row.name}-away",
                "coach_name": normalize_coach_name(row["away_manager"]),
                "goals_for": row["away_score"],
                "goals_against": row["home_score"],
                "is_win": row["away_score"] > row["home_score"],
                "source": "world_cup_history",
            }
        )
    return pd.DataFrame(rows)
