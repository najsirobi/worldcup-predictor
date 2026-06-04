"""Compute live group tables from manually entered scores (Travel Mode, Task B).

Pure functions over the score override frame: no I/O side effects, no model
calls. Group tables apply the standard group-stage points system and the
documented (approximate) tie-break ladder below.
"""

from __future__ import annotations

import pandas as pd

from src.live.scores_override import VALID_STATUSES  # noqa: F401  (re-exported intent)

POINTS_WIN = 3
POINTS_DRAW = 1
POINTS_LOSS = 0

# FIFA's full ladder also uses head-to-head points/GD/goals, then fair-play and
# finally a drawing of lots. We only have manually entered scores, so we sort by
# points -> goal difference -> goals for and break any remaining tie
# deterministically by team name. This is flagged everywhere the table surfaces.
TIE_BREAK_NOTE = (
    "Tie-break limitation: live tables are ordered by points, then goal "
    "difference, then goals for, then team name (alphabetical) as a deterministic "
    "fallback. The full FIFA/FIF8A tie-breakers (head-to-head record, "
    "head-to-head goal difference/goals, disciplinary/fair-play points, drawing "
    "of lots) are NOT applied here. Final standings may differ when teams are "
    "level on points and goal difference."
)

TABLE_COLUMNS = [
    "group",
    "rank",
    "team",
    "played",
    "won",
    "drawn",
    "lost",
    "goals_for",
    "goals_against",
    "goal_difference",
    "points",
]


def _played_rows(scores: pd.DataFrame) -> pd.DataFrame:
    played = scores[scores["status"] == "played"].copy()
    if played.empty:
        return played
    played["team_a_goals"] = played["team_a_goals"].astype(int)
    played["team_b_goals"] = played["team_b_goals"].astype(int)
    return played


def split_played_remaining(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the override frame into played and not-yet-played matches.

    ``played`` matches have a final result. Everything else (scheduled,
    postponed, void) is "remaining" for table/simulation purposes; ``void``
    matches are excluded from both because they will never produce points.
    """
    played = _played_rows(scores)
    remaining = scores[~scores["status"].isin(("played", "void"))].copy()
    played_cols = [
        "match_number",
        "group",
        "date",
        "team_a",
        "team_b",
        "team_a_goals",
        "team_b_goals",
        "status",
        "source",
        "updated_at",
        "notes",
    ]
    remaining_cols = [
        "match_number",
        "group",
        "date",
        "team_a",
        "team_b",
        "status",
        "notes",
    ]
    played = played[played_cols].sort_values("match_number").reset_index(drop=True)
    remaining = remaining[remaining_cols].sort_values("match_number").reset_index(drop=True)
    return played, remaining


def compute_group_tables(scores: pd.DataFrame) -> pd.DataFrame:
    """Build ranked group tables from the played matches in ``scores``.

    Every team that appears in the fixture list gets a row even with zero games
    played, so the tables are complete from the very first update.
    """
    # Seed every team with an empty record from the full fixture list.
    teams: dict[tuple[str, str], dict] = {}
    for _, row in scores.iterrows():
        for team in (row["team_a"], row["team_b"]):
            key = (row["group"], team)
            teams.setdefault(
                key,
                {
                    "group": row["group"],
                    "team": team,
                    "played": 0,
                    "won": 0,
                    "drawn": 0,
                    "lost": 0,
                    "goals_for": 0,
                    "goals_against": 0,
                    "points": 0,
                },
            )

    for _, row in _played_rows(scores).iterrows():
        ga, gb = int(row["team_a_goals"]), int(row["team_b_goals"])
        a = teams[(row["group"], row["team_a"])]
        b = teams[(row["group"], row["team_b"])]
        a["played"] += 1
        b["played"] += 1
        a["goals_for"] += ga
        a["goals_against"] += gb
        b["goals_for"] += gb
        b["goals_against"] += ga
        if ga > gb:
            a["won"] += 1
            b["lost"] += 1
            a["points"] += POINTS_WIN
        elif ga < gb:
            b["won"] += 1
            a["lost"] += 1
            b["points"] += POINTS_WIN
        else:
            a["drawn"] += 1
            b["drawn"] += 1
            a["points"] += POINTS_DRAW
            b["points"] += POINTS_DRAW

    frame = pd.DataFrame(list(teams.values()))
    frame["goal_difference"] = frame["goals_for"] - frame["goals_against"]
    # Sort: points desc, GD desc, GF desc, team name asc (deterministic fallback).
    frame = frame.sort_values(
        ["group", "points", "goal_difference", "goals_for", "team"],
        ascending=[True, False, False, False, True],
    ).reset_index(drop=True)
    frame["rank"] = frame.groupby("group").cumcount() + 1
    return frame[TABLE_COLUMNS]


def group_tables_to_records(tables: pd.DataFrame) -> dict:
    """Nest the flat table frame into a JSON-friendly {group: [rows]} dict."""
    out: dict[str, list[dict]] = {}
    for group, sub in tables.groupby("group"):
        out[str(group)] = sub.drop(columns=["group"]).to_dict(orient="records")
    return out
