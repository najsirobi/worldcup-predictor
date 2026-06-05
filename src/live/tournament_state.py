"""Compute live group tables and actual bracket assignment from live results.

Pure functions over the score override frame: no I/O side effects, no model
calls. Group tables apply the standard group-stage points system and the
documented (approximate) tie-break ladder below.

Frozen submitted score predictions are not inputs here. Actual results update
only the live group table and, once all group matches are complete, the actual
Round-of-32 assignment from official Annexe C plus the FIFA bracket mapping.
"""

from __future__ import annotations

import pandas as pd

from src.live.scores_override import (  # noqa: F401  (re-exported intent)
    GROUP_STAGE_MATCH_MAX,
    VALID_STATUSES,
)
from src.simulation.knockout_bracket import assign_official_r32_matches

POINTS_WIN = 3
POINTS_DRAW = 1
POINTS_LOSS = 0
N_ADVANCING_THIRDS = 8

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

ACTUAL_R32_COLUMNS = [
    "match_number",
    "round",
    "team_a",
    "team_b",
    "team_a_source_position",
    "team_b_source_position",
    "team_a_source",
    "team_b_source",
]

LIVE_STATE_SEMANTICS = {
    "submitted_predictions": (
        "Frozen candidate files are read-only after submission; live results do "
        "not rewrite or regenerate submitted group score predictions."
    ),
    "played_group_matches": (
        "Played group matches use actual scores from data/live/scores_override.csv "
        "for live tables and prediction-vs-actual scoring."
    ),
    "unplayed_group_matches": (
        "Unplayed group matches keep using the frozen active candidate's scoreline "
        "distribution for simulations."
    ),
    "group_advancement": (
        "Advancement probabilities combine actual played results with simulated "
        "unplayed results."
    ),
    "knockout_state": (
        "After the group stage is complete, actual group standings and the best "
        "third-placed teams feed official Annexe C and the Round-of-32 bracket "
        "mapping. Later played knockout matches must pin actual winners and only "
        "future matches may be simulated."
    ),
}


def _group_stage_rows(scores: pd.DataFrame) -> pd.DataFrame:
    """Only the group-stage matches (1-72); knockout rows (73-104) are excluded.

    Group tables, qualification and standings must never be influenced by
    knockout fixtures, which live in the same override file."""
    return scores[scores["match_number"].astype(int) <= GROUP_STAGE_MATCH_MAX].copy()


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
    scores = _group_stage_rows(scores)
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
    scores = _group_stage_rows(scores)
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


def group_stage_complete(tables: pd.DataFrame) -> bool:
    """Return True only when every team has three played group matches."""
    return bool(not tables.empty and tables["played"].eq(3).all())


def source_positions_from_live_tables(tables: pd.DataFrame) -> dict[str, str]:
    """Build source positions such as A1, A2, A3 from actual live standings."""
    required = {"group", "rank", "team"}
    missing = required - set(tables.columns)
    if missing:
        raise ValueError(f"Live tables missing columns for source positions: {sorted(missing)}")
    return {
        f"{row['group']}{int(row['rank'])}": str(row["team"])
        for _, row in tables.iterrows()
    }


def actual_best_third_groups(tables: pd.DataFrame) -> list[str]:
    """Return the eight best third-placed groups from actual live tables.

    This uses the available deterministic table ordering columns:
    points -> goal difference -> goals for -> team name. Full disciplinary and
    drawing-of-lots tie-breakers are not available in the live score override.
    """
    third = tables[tables["rank"].eq(3)].copy()
    if len(third) != 12:
        raise ValueError(f"Expected 12 third-placed teams, found {len(third)}.")
    third = third.sort_values(
        ["points", "goal_difference", "goals_for", "team"],
        ascending=[False, False, False, True],
    )
    return third.head(N_ADVANCING_THIRDS)["group"].astype(str).tolist()


def actual_round_of_32_from_group_tables(
    tables: pd.DataFrame,
    r32_mapping: pd.DataFrame,
    annex: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Assign the actual Round of 32 once all group-stage results are available."""
    if not group_stage_complete(tables):
        raise ValueError("Actual Round-of-32 assignment requires every group match to be played.")
    source_positions = source_positions_from_live_tables(tables)
    qualified_third_groups = actual_best_third_groups(tables)
    assigned = assign_official_r32_matches(
        r32_mapping,
        source_positions,
        annex,
        qualified_third_groups,
    )
    return assigned[ACTUAL_R32_COLUMNS], qualified_third_groups


def actual_bracket_state_payload(
    tables: pd.DataFrame,
    r32_mapping: pd.DataFrame,
    annex: pd.DataFrame,
) -> dict:
    """Return JSON-ready actual bracket state without simulating future matches."""
    if not group_stage_complete(tables):
        return {
            "status": "pending_group_stage",
            "qualified_third_groups": [],
            "round_of_32_matches": [],
            "note": (
                "Actual bracket assignment is pending until all group-stage "
                "matches have final scores."
            ),
        }
    r32, qualified_third_groups = actual_round_of_32_from_group_tables(
        tables,
        r32_mapping,
        annex,
    )
    return {
        "status": "actual_group_stage_complete",
        "qualified_third_groups": qualified_third_groups,
        "round_of_32_matches": r32.to_dict(orient="records"),
        "note": "Actual Round-of-32 assignment from live standings, Annexe C, and bracket mapping.",
    }
