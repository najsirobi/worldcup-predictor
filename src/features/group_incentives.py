"""Pre-match group-stage incentive states and capped scoreline adjustments.

The incentive state functions only use group matches that were already played
before the match being labelled. Future fixtures are used only as unknown
remaining fixtures for possibility checks; their actual results are never read.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from math import exp, factorial
from typing import Iterable

import numpy as np
import pandas as pd


POINTS_WIN = 3
POINTS_DRAW = 1
TOP2_SLOTS = 2

TEAM_STATE_COLUMNS = [
    "group_matches_played_before",
    "group_points_before",
    "group_goal_difference_before",
    "group_goals_for_before",
    "can_still_finish_1st",
    "has_clinched_1st",
    "has_clinched_top2",
    "has_clinched_advancement_if_best_thirds_supported",
    "is_eliminated",
    "must_win_for_top2",
    "draw_likely_enough_for_top2",
    "low_incentive_flag",
    "high_incentive_flag",
    "incentive_score",
]


@dataclass(frozen=True)
class IncentiveAdjustmentConfig:
    """Conservative scoreline adjustment caps.

    `low_xg_factor` is intentionally small and is capped again by
    `max_xg_shift`. High-incentive teams are not boosted by default; the project
    hypothesis requires data support before increasing any team's expected goals.
    """

    low_xg_factor: float = 0.07
    high_xg_factor: float = 0.0
    max_xg_shift: float = 0.15
    max_probability_shift: float = 0.05
    final_group_only: bool = True


def _empty_table(teams: Iterable[str]) -> dict[str, dict[str, int]]:
    return {
        str(team): {
            "played": 0,
            "points": 0,
            "goals_for": 0,
            "goals_against": 0,
            "goal_difference": 0,
        }
        for team in sorted(set(map(str, teams)))
    }


def _has_result(row: pd.Series) -> bool:
    if "status" in row.index and str(row["status"]) != "played":
        return False
    return not pd.isna(row.get("team_a_goals")) and not pd.isna(row.get("team_b_goals"))


def _apply_score(table: dict[str, dict[str, int]], team_a: str, team_b: str, goals_a: int, goals_b: int) -> None:
    a = table[str(team_a)]
    b = table[str(team_b)]
    a["played"] += 1
    b["played"] += 1
    a["goals_for"] += int(goals_a)
    a["goals_against"] += int(goals_b)
    b["goals_for"] += int(goals_b)
    b["goals_against"] += int(goals_a)
    a["goal_difference"] = a["goals_for"] - a["goals_against"]
    b["goal_difference"] = b["goals_for"] - b["goals_against"]
    if goals_a > goals_b:
        a["points"] += POINTS_WIN
    elif goals_a < goals_b:
        b["points"] += POINTS_WIN
    else:
        a["points"] += POINTS_DRAW
        b["points"] += POINTS_DRAW


def _outcome_points(outcome: str) -> tuple[int, int]:
    if outcome == "a_win":
        return POINTS_WIN, 0
    if outcome == "b_win":
        return 0, POINTS_WIN
    if outcome == "draw":
        return POINTS_DRAW, POINTS_DRAW
    raise ValueError(f"Unknown outcome: {outcome}")


def _fixture_dicts(matches: pd.DataFrame) -> list[dict]:
    fixtures = []
    for _, row in matches.iterrows():
        fixtures.append(
            {
                "match_number": int(row["match_number"]),
                "team_a": str(row["team_a"]),
                "team_b": str(row["team_b"]),
            }
        )
    return fixtures


def _completion_points(
    table: dict[str, dict[str, int]],
    remaining_fixtures: list[dict],
    fixed_outcome_by_match: dict[int, str] | None = None,
) -> list[dict[str, int]]:
    """Enumerate possible final points tables from W/D/L outcomes.

    Ties are treated separately by the state predicates: possible states give
    the labelled team the benefit of a tie-break, while clinched states assume
    tied teams could pass it. This avoids pretending to know future goal margins.
    """

    fixed_outcome_by_match = fixed_outcome_by_match or {}
    variable = [
        fixture
        for fixture in remaining_fixtures
        if int(fixture["match_number"]) not in fixed_outcome_by_match
    ]
    outputs: list[dict[str, int]] = []
    for outcomes in product(("a_win", "draw", "b_win"), repeat=len(variable)):
        points = {team: int(stats["points"]) for team, stats in table.items()}
        for fixture in remaining_fixtures:
            match_number = int(fixture["match_number"])
            outcome = fixed_outcome_by_match.get(match_number)
            if outcome is None:
                outcome = outcomes[variable.index(fixture)]
            pa, pb = _outcome_points(outcome)
            points[fixture["team_a"]] += pa
            points[fixture["team_b"]] += pb
        outputs.append(points)
    return outputs


def _can_top2(points_table: dict[str, int], team: str) -> bool:
    team_points = points_table[team]
    return sum(points > team_points for other, points in points_table.items() if other != team) <= 1


def _guaranteed_top2(points_table: dict[str, int], team: str) -> bool:
    team_points = points_table[team]
    return sum(points >= team_points for other, points in points_table.items() if other != team) <= 1


def _can_first(points_table: dict[str, int], team: str) -> bool:
    team_points = points_table[team]
    return all(points <= team_points for other, points in points_table.items() if other != team)


def _guaranteed_first(points_table: dict[str, int], team: str) -> bool:
    team_points = points_table[team]
    return all(points < team_points for other, points in points_table.items() if other != team)


def _current_outcome_for_team(team: str, current_fixture: dict, team_result: str) -> str:
    if team_result == "draw":
        return "draw"
    is_team_a = str(team) == str(current_fixture["team_a"])
    if team_result == "win":
        return "a_win" if is_team_a else "b_win"
    if team_result == "loss":
        return "b_win" if is_team_a else "a_win"
    raise ValueError(f"Unknown team_result: {team_result}")


def compute_table_from_played(matches: pd.DataFrame, teams: Iterable[str] | None = None) -> dict[str, dict[str, int]]:
    """Build a table from played rows only."""

    if teams is None:
        teams = set(matches["team_a"].astype(str)) | set(matches["team_b"].astype(str))
    table = _empty_table(teams)
    for _, row in matches.iterrows():
        if not _has_result(row):
            continue
        _apply_score(
            table,
            str(row["team_a"]),
            str(row["team_b"]),
            int(row["team_a_goals"]),
            int(row["team_b_goals"]),
        )
    return table


def compute_team_state_from_table(
    table: dict[str, dict[str, int]],
    remaining_fixtures: list[dict],
    current_fixture: dict,
    team: str,
    *,
    best_thirds_supported: bool = False,
    final_group_match: bool | None = None,
) -> dict:
    """Compute one team's pre-match incentive state from current table state."""

    team = str(team)
    if team not in table:
        raise ValueError(f"Team {team!r} is missing from group table.")

    all_points = _completion_points(table, remaining_fixtures)
    if not all_points:
        all_points = [{name: int(stats["points"]) for name, stats in table.items()}]

    can_still_finish_1st = any(_can_first(points, team) for points in all_points)
    has_clinched_1st = all(_guaranteed_first(points, team) for points in all_points)
    has_clinched_top2 = all(_guaranteed_top2(points, team) for points in all_points)
    is_eliminated = not any(_can_top2(points, team) for points in all_points)

    current_match_number = int(current_fixture["match_number"])
    fixed_rank_possibilities = {}
    fixed_rank_guarantees = {}
    for team_result in ("win", "draw", "loss"):
        outcome = _current_outcome_for_team(team, current_fixture, team_result)
        completions = _completion_points(
            table,
            remaining_fixtures,
            fixed_outcome_by_match={current_match_number: outcome},
        )
        fixed_rank_possibilities[team_result] = any(_can_top2(points, team) for points in completions)
        fixed_rank_guarantees[team_result] = all(_guaranteed_top2(points, team) for points in completions)

    must_win_for_top2 = (
        not has_clinched_top2
        and not is_eliminated
        and fixed_rank_possibilities["win"]
        and not fixed_rank_possibilities["draw"]
        and not fixed_rank_possibilities["loss"]
    )
    draw_likely_enough_for_top2 = bool(
        has_clinched_top2 or fixed_rank_guarantees["draw"]
    )

    played_before = int(table[team]["played"])
    if final_group_match is None:
        current_a_played = int(table[str(current_fixture["team_a"])]["played"])
        current_b_played = int(table[str(current_fixture["team_b"])]["played"])
        final_group_match = current_a_played >= 2 and current_b_played >= 2

    has_clinched_advancement = bool(best_thirds_supported and has_clinched_top2)
    low_incentive = bool(
        final_group_match
        and (has_clinched_1st or has_clinched_top2 or has_clinched_advancement or is_eliminated)
    )
    high_incentive = bool(
        final_group_match
        and not has_clinched_top2
        and not is_eliminated
        and (must_win_for_top2 or any(_can_top2(points, team) for points in all_points))
    )

    if has_clinched_1st:
        incentive_score = -2.0
    elif has_clinched_top2 or has_clinched_advancement:
        incentive_score = -1.25
    elif is_eliminated:
        incentive_score = -1.0
    elif must_win_for_top2:
        incentive_score = 2.0
    elif high_incentive:
        incentive_score = 1.0
    elif can_still_finish_1st:
        incentive_score = 0.25
    else:
        incentive_score = 0.0

    return {
        "group_matches_played_before": played_before,
        "group_points_before": int(table[team]["points"]),
        "group_goal_difference_before": int(table[team]["goal_difference"]),
        "group_goals_for_before": int(table[team]["goals_for"]),
        "can_still_finish_1st": bool(can_still_finish_1st),
        "has_clinched_1st": bool(has_clinched_1st),
        "has_clinched_top2": bool(has_clinched_top2),
        "has_clinched_advancement_if_best_thirds_supported": has_clinched_advancement,
        "is_eliminated": bool(is_eliminated),
        "must_win_for_top2": bool(must_win_for_top2),
        "draw_likely_enough_for_top2": bool(draw_likely_enough_for_top2),
        "low_incentive_flag": low_incentive,
        "high_incentive_flag": high_incentive,
        "incentive_score": float(incentive_score),
    }


def _kickoff_series(matches: pd.DataFrame) -> pd.Series:
    if "_kickoff" in matches.columns:
        return pd.to_datetime(matches["_kickoff"])
    return pd.to_datetime(matches["date"])


def compute_team_incentive_state(
    group_matches: pd.DataFrame,
    current_match: pd.Series,
    team: str,
    *,
    best_thirds_supported: bool = False,
) -> dict:
    """Compute one side's pre-match state using only prior played matches."""

    teams = set(group_matches["team_a"].astype(str)) | set(group_matches["team_b"].astype(str))
    current_kickoff = pd.to_datetime(current_match.get("_kickoff", current_match["date"]))
    group = group_matches.copy()
    group["_kickoff"] = _kickoff_series(group)
    prior = group[group["_kickoff"] < current_kickoff].copy()
    table = compute_table_from_played(prior, teams)
    remaining = group[group["_kickoff"] >= current_kickoff].copy()
    remaining_fixtures = _fixture_dicts(remaining)
    current_fixture = {
        "match_number": int(current_match["match_number"]),
        "team_a": str(current_match["team_a"]),
        "team_b": str(current_match["team_b"]),
    }
    final_group_match = (
        int(table[str(current_fixture["team_a"])]["played"]) >= 2
        and int(table[str(current_fixture["team_b"])]["played"]) >= 2
    )
    return compute_team_state_from_table(
        table,
        remaining_fixtures,
        current_fixture,
        team,
        best_thirds_supported=best_thirds_supported,
        final_group_match=final_group_match,
    )


def _favorite_side(row: pd.Series) -> str | None:
    if "home_elo" in row.index and "away_elo" in row.index:
        if not pd.isna(row["home_elo"]) and not pd.isna(row["away_elo"]):
            if float(row["home_elo"]) > float(row["away_elo"]):
                return "team_a"
            if float(row["away_elo"]) > float(row["home_elo"]):
                return "team_b"
    if "model_p_a_win" in row.index and "model_p_b_win" in row.index:
        if not pd.isna(row["model_p_a_win"]) and not pd.isna(row["model_p_b_win"]):
            if float(row["model_p_a_win"]) > float(row["model_p_b_win"]):
                return "team_a"
            if float(row["model_p_b_win"]) > float(row["model_p_a_win"]):
                return "team_b"
    return None


def build_incentive_features_for_matches(
    matches: pd.DataFrame,
    *,
    best_thirds_supported: bool = False,
) -> pd.DataFrame:
    """Return match-level incentive features for group-stage fixtures.

    Required columns: group, match_number, date, team_a, team_b, team_a_goals,
    team_b_goals. Optional rating/model columns are preserved and can be used for
    favourite/underdog labels.
    """

    required = {"group", "match_number", "date", "team_a", "team_b", "team_a_goals", "team_b_goals"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"Group incentive matches missing columns: {sorted(missing)}")

    frame = matches.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["_kickoff"] = pd.to_datetime(frame["date"])
    sort_cols = [col for col in ("year", "tournament_id", "group", "_kickoff", "match_number") if col in frame.columns]
    frame = frame.sort_values(sort_cols).reset_index(drop=True)

    records = []
    for _, current in frame.iterrows():
        group_mask = frame["group"].astype(str).eq(str(current["group"]))
        if "year" in frame.columns:
            group_mask &= frame["year"].eq(current["year"])
        if "tournament_id" in frame.columns:
            group_mask &= frame["tournament_id"].astype(str).eq(str(current["tournament_id"]))
        group_matches = frame[group_mask].copy()

        state_a = compute_team_incentive_state(
            group_matches,
            current,
            str(current["team_a"]),
            best_thirds_supported=best_thirds_supported,
        )
        state_b = compute_team_incentive_state(
            group_matches,
            current,
            str(current["team_b"]),
            best_thirds_supported=best_thirds_supported,
        )
        final_group_match = bool(
            state_a["group_matches_played_before"] >= 2
            and state_b["group_matches_played_before"] >= 2
        )
        favorite = _favorite_side(current)
        underdog = None
        if favorite == "team_a":
            underdog = "team_b"
        elif favorite == "team_b":
            underdog = "team_a"

        record = current.drop(labels=["_kickoff"], errors="ignore").to_dict()
        for prefix, state in (("team_a", state_a), ("team_b", state_b)):
            for key in TEAM_STATE_COLUMNS:
                record[f"{prefix}_{key}"] = state[key]
        record["incentive_diff"] = float(state_a["incentive_score"] - state_b["incentive_score"])
        record["both_low_incentive"] = bool(state_a["low_incentive_flag"] and state_b["low_incentive_flag"])
        record["favourite_side"] = favorite
        record["underdog_side"] = underdog
        record["favourite_low_incentive"] = bool(
            (favorite == "team_a" and state_a["low_incentive_flag"])
            or (favorite == "team_b" and state_b["low_incentive_flag"])
        )
        record["underdog_high_incentive"] = bool(
            (underdog == "team_a" and state_a["high_incentive_flag"])
            or (underdog == "team_b" and state_b["high_incentive_flag"])
        )
        record["final_group_match_flag"] = final_group_match
        records.append(record)
    return pd.DataFrame.from_records(records)


def poisson_score_matrix(lambda_a: float, lambda_b: float, max_goals: int = 10) -> np.ndarray:
    """Independent-Poisson score matrix with truncated tail renormalized."""

    goals = np.arange(max_goals + 1)

    def pmf(lam: float) -> np.ndarray:
        lam = max(float(lam), 1e-6)
        vals = np.array([exp(-lam) * lam**int(k) / factorial(int(k)) for k in goals], dtype=float)
        return vals / vals.sum()

    return np.outer(pmf(lambda_a), pmf(lambda_b))


def outcome_probabilities_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """Return [team_a_win, draw, team_b_win] probabilities."""

    return np.array(
        [
            float(np.tril(matrix, -1).sum()),
            float(np.trace(matrix)),
            float(np.triu(matrix, 1).sum()),
        ]
    )


def expected_goals_from_matrix(matrix: np.ndarray) -> tuple[float, float]:
    goals = np.arange(matrix.shape[0])
    return float((matrix * goals[:, None]).sum()), float((matrix * goals[None, :]).sum())


def _adjust_lambda(lam: float, state: dict, config: IncentiveAdjustmentConfig) -> tuple[float, float]:
    lam = max(float(lam), 1e-6)
    if state.get("low_incentive_flag"):
        raw_shift = lam * float(config.low_xg_factor)
        shift = min(raw_shift, float(config.max_xg_shift))
        return max(1e-6, lam - shift), -shift
    if state.get("high_incentive_flag") and config.high_xg_factor > 0:
        raw_shift = lam * float(config.high_xg_factor)
        shift = min(raw_shift, float(config.max_xg_shift))
        return lam + shift, shift
    return lam, 0.0


def adjust_score_matrix_for_incentives(
    matrix: np.ndarray,
    lambda_a: float,
    lambda_b: float,
    state_a: dict,
    state_b: dict,
    *,
    final_group_match: bool,
    config: IncentiveAdjustmentConfig | None = None,
) -> tuple[np.ndarray, dict]:
    """Apply a capped final-group incentive adjustment to a score matrix.

    The returned metadata records xG and W/D/L probability shifts after capping.
    If the outcome-probability cap would be exceeded, the adjusted matrix is
    blended back toward the original matrix.
    """

    config = config or IncentiveAdjustmentConfig()
    original = np.asarray(matrix, dtype=float)
    original = original / original.sum()
    if config.final_group_only and not final_group_match:
        return original, {
            "applied": False,
            "lambda_a_shift": 0.0,
            "lambda_b_shift": 0.0,
            "max_probability_shift": 0.0,
            "blend_alpha": 0.0,
        }

    adjusted_lambda_a, shift_a = _adjust_lambda(lambda_a, state_a, config)
    adjusted_lambda_b, shift_b = _adjust_lambda(lambda_b, state_b, config)
    if shift_a == 0.0 and shift_b == 0.0:
        return original, {
            "applied": False,
            "lambda_a_shift": 0.0,
            "lambda_b_shift": 0.0,
            "max_probability_shift": 0.0,
            "blend_alpha": 0.0,
        }

    adjusted = poisson_score_matrix(adjusted_lambda_a, adjusted_lambda_b, max_goals=original.shape[0] - 1)
    original_probs = outcome_probabilities_from_matrix(original)
    adjusted_probs = outcome_probabilities_from_matrix(adjusted)
    max_shift = float(np.max(np.abs(adjusted_probs - original_probs)))
    blend_alpha = 1.0
    if max_shift > float(config.max_probability_shift) > 0:
        blend_alpha = float(config.max_probability_shift) / max_shift
        adjusted = original * (1.0 - blend_alpha) + adjusted * blend_alpha
        adjusted = adjusted / adjusted.sum()
        adjusted_probs = outcome_probabilities_from_matrix(adjusted)
        max_shift = float(np.max(np.abs(adjusted_probs - original_probs)))

    return adjusted, {
        "applied": True,
        "lambda_a_shift": float(max(-config.max_xg_shift, min(config.max_xg_shift, shift_a))),
        "lambda_b_shift": float(max(-config.max_xg_shift, min(config.max_xg_shift, shift_b))),
        "max_probability_shift": max_shift,
        "blend_alpha": float(blend_alpha),
    }


def incentive_note_for_state(state: dict, opponent_state: dict | None = None) -> list[str]:
    """Human-readable live diagnostics for a team before a final group match."""

    notes: list[str] = []
    if state.get("has_clinched_1st"):
        notes.append("clinched first")
    elif state.get("low_incentive_flag"):
        notes.append("low incentive / possible rotation")
    if state.get("has_clinched_top2"):
        notes.append("qualified")
    if state.get("is_eliminated"):
        notes.append("eliminated")
    if state.get("must_win_for_top2"):
        notes.append("must win")
    elif state.get("draw_likely_enough_for_top2"):
        notes.append("draw likely enough")
    if opponent_state and opponent_state.get("has_clinched_top2"):
        notes.append("opponent already qualified")
    return notes


def build_live_incentive_diagnostics(scores: pd.DataFrame) -> dict[str, list[dict]]:
    """Build live context diagnostics from the score override frame.

    This never changes predictions; it only labels current tables and remaining
    fixtures from actual played scores.
    """

    group_stage = scores[scores["match_number"].astype(int) <= 72].copy()
    group_stage = group_stage[group_stage["status"] != "void"].copy()
    if group_stage.empty:
        return {"teams": [], "matches": []}

    required_goals = ["team_a_goals", "team_b_goals"]
    for col in required_goals:
        if col not in group_stage.columns:
            group_stage[col] = pd.NA
    features = build_incentive_features_for_matches(group_stage, best_thirds_supported=True)
    remaining = features[features.get("status", "scheduled").astype(str) != "played"].copy()

    team_records: list[dict] = []
    for _, row in remaining.iterrows():
        if not bool(row["final_group_match_flag"]):
            continue
        for side in ("team_a", "team_b"):
            other = "team_b" if side == "team_a" else "team_a"
            state = {key: row[f"{side}_{key}"] for key in TEAM_STATE_COLUMNS}
            opponent_state = {key: row[f"{other}_{key}"] for key in TEAM_STATE_COLUMNS}
            notes = incentive_note_for_state(state, opponent_state)
            if notes:
                team_records.append(
                    {
                        "match_number": int(row["match_number"]),
                        "group": row["group"],
                        "team": row[side],
                        "opponent": row[other],
                        "notes": notes,
                        "incentive_score": float(state["incentive_score"]),
                        "low_incentive_flag": bool(state["low_incentive_flag"]),
                        "high_incentive_flag": bool(state["high_incentive_flag"]),
                    }
                )

    match_records: list[dict] = []
    for _, row in remaining.iterrows():
        if not bool(row["final_group_match_flag"]):
            continue
        state_a = {key: row[f"team_a_{key}"] for key in TEAM_STATE_COLUMNS}
        state_b = {key: row[f"team_b_{key}"] for key in TEAM_STATE_COLUMNS}
        notes_a = incentive_note_for_state(state_a, state_b)
        notes_b = incentive_note_for_state(state_b, state_a)
        if notes_a or notes_b:
            match_records.append(
                {
                    "match_number": int(row["match_number"]),
                    "group": row["group"],
                    "team_a": row["team_a"],
                    "team_b": row["team_b"],
                    "team_a_notes": notes_a,
                    "team_b_notes": notes_b,
                    "both_low_incentive": bool(row["both_low_incentive"]),
                    "incentive_diff": float(row["incentive_diff"]),
                }
            )

    return {"teams": team_records, "matches": match_records}


def adjustment_config_to_dict(config: IncentiveAdjustmentConfig) -> dict:
    return asdict(config)
