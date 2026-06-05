"""Round-by-round knockout exact-score predictions (live, future-only).

The knockout match predictions in the FIF8A game are entered *along the way*: each
round you fill in the previous round's actual results and then predict the next
round's exact scores (rules §4b — correct qualified team x odd, +2 exact score,
+2 shoot-out call). Only the group-stage scores, group standings, and the Last-8
progression block are locked up front.

This module mirrors that flow. It produces, for every knockout match (73-104):

* a **projected** prediction — the model's original up-front gamble for the whole
  bracket, derived from the frozen group-stage projection, and
* a **current** recommendation — refreshed from the ACTUAL participants once they
  are known (group stage complete pins the Round of 32; played knockout matches
  pin later-round participants), so each round you see predictions for the teams
  that are really there.

Submitted group-stage predictions are never read or modified here. Scorelines
reuse the existing Phase 4.5 strength signals via an independent-Poisson matrix;
no model is retrained and nothing hits the network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.baselines import PoissonScoreModel
from src.simulation.full_tournament import source_positions_from_group_view
from src.simulation.knockout_bracket import assign_official_r32_matches
from src.simulation.match_probability import knockout_match_probability
from src.live.tournament_state import (
    actual_round_of_32_from_group_tables,
    group_stage_complete,
)

# Knockout fixtures and the round each match number belongs to (FIFA 48-team bracket).
R32_MATCHES = range(73, 89)
ROUND_BY_MATCH: dict[int, str] = {}
for _n in range(73, 89):
    ROUND_BY_MATCH[_n] = "R32"
for _n in range(89, 97):
    ROUND_BY_MATCH[_n] = "R16"
for _n in range(97, 101):
    ROUND_BY_MATCH[_n] = "QF"
for _n in (101, 102):
    ROUND_BY_MATCH[_n] = "SF"
ROUND_BY_MATCH[103] = "Third-place"
ROUND_BY_MATCH[104] = "Final"

ROUND_ORDER = ["R32", "R16", "QF", "SF", "Third-place", "Final"]
ROUND_LABELS = {
    "R32": "Round of 32",
    "R16": "Round of 16",
    "QF": "Quarter-finals",
    "SF": "Semi-finals",
    "Third-place": "Third-place play-off",
    "Final": "Final",
}

_SCORE_MODEL = PoissonScoreModel(max_goals=10)


def most_probable_scoreline(expected_goals_a: float, expected_goals_b: float) -> tuple[int, int]:
    """Most probable exact scoreline under the independent-Poisson matrix."""
    matrix = _SCORE_MODEL.score_matrix(expected_goals_a, expected_goals_b)
    flat_index = int(np.argmax(matrix))
    goals_a, goals_b = divmod(flat_index, matrix.shape[1])
    return int(goals_a), int(goals_b)


def predict_match(team_a: str, team_b: str, group_view: pd.DataFrame) -> dict:
    """Predicted exact score, advancing team, and shoot-out call for one match."""
    probability = knockout_match_probability(team_a, team_b, group_view)
    goals_a, goals_b = most_probable_scoreline(
        probability.expected_goals_a, probability.expected_goals_b
    )
    if goals_a > goals_b:
        advancing = team_a
        shootout = False
    elif goals_b > goals_a:
        advancing = team_b
        shootout = False
    else:
        # A drawn 90/120-minute scoreline is resolved by the stronger team on the
        # advancement share, and we call the shoot-out (rules §4b +2).
        advancing = team_a if probability.p_team_a_advance >= 0.5 else team_b
        shootout = True
    return {
        "team_a": team_a,
        "team_b": team_b,
        "score": f"{goals_a}-{goals_b}",
        "advancing_team": advancing,
        "shootout": shootout,
        "p_advance_team_a": round(float(probability.p_team_a_advance), 4),
        "expected_goals_a": probability.expected_goals_a,
        "expected_goals_b": probability.expected_goals_b,
    }


def _projected_qualified_third_groups(group_view: pd.DataFrame) -> list[str]:
    third = group_view[group_view["suggested_group_standing"].eq(3)].copy()
    third = third.sort_values(["likely_best_third_signal", "p_top3"], ascending=False)
    return third.head(8)["group"].astype(str).tolist()


def _played_knockout_scores(scores: pd.DataFrame) -> dict[int, tuple[int, int]]:
    """Map of played knockout match_number -> (goals_a, goals_b) from actual results."""
    played: dict[int, tuple[int, int]] = {}
    if scores is None or scores.empty:
        return played
    for _, row in scores.iterrows():
        number = row.get("match_number")
        if number is None or int(number) not in ROUND_BY_MATCH:
            continue
        goals_a = row.get("team_a_goals")
        goals_b = row.get("team_b_goals")
        if pd.isna(goals_a) or pd.isna(goals_b):
            continue
        played[int(number)] = (int(goals_a), int(goals_b))
    return played


def _knockout_advanced_teams(scores: pd.DataFrame) -> dict[int, str]:
    """Map of knockout match_number -> advanced_team (shoot-out winner) if entered."""
    advanced: dict[int, str] = {}
    if scores is None or scores.empty or "advanced_team" not in scores.columns:
        return advanced
    for _, row in scores.iterrows():
        number = row.get("match_number")
        if number is None or int(number) not in ROUND_BY_MATCH:
            continue
        value = row.get("advanced_team")
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        text = str(value).strip()
        if text:
            advanced[int(number)] = text
    return advanced


def _resolve(source: str, winners: dict[int, str], losers: dict[int, str]) -> str | None:
    value = str(source)
    if value.startswith("W"):
        return winners.get(int(value[1:]))
    if value.startswith("L"):
        return losers.get(int(value[1:]))
    return None


def _walk_bracket(
    r32_teams: dict[int, tuple[str, str]],
    progression: pd.DataFrame,
    group_view: pd.DataFrame,
) -> dict[int, dict]:
    """Predict every knockout match by walking the bracket from a set R32.

    ``r32_teams`` maps match numbers 73-88 to (team_a, team_b). Later rounds use
    the predicted advancing team to fill W#/L# sources. Returns match_number ->
    prediction dict (including resolved teams and winner/loser)."""
    predictions: dict[int, dict] = {}
    winners: dict[int, str] = {}
    losers: dict[int, str] = {}

    for match_number in sorted(r32_teams):
        team_a, team_b = r32_teams[match_number]
        pred = predict_match(team_a, team_b, group_view)
        predictions[match_number] = pred
        winners[match_number] = pred["advancing_team"]
        losers[match_number] = team_b if pred["advancing_team"] == team_a else team_a

    for _, row in progression.sort_values("match_number").iterrows():
        match_number = int(row["match_number"])
        team_a = _resolve(row["team_a_source"], winners, losers)
        team_b = _resolve(row["team_b_source"], winners, losers)
        if team_a is None or team_b is None:
            continue
        pred = predict_match(team_a, team_b, group_view)
        predictions[match_number] = pred
        winners[match_number] = pred["advancing_team"]
        losers[match_number] = team_b if pred["advancing_team"] == team_a else team_a

    return predictions


def _actual_winner_loser(
    team_a: str | None,
    team_b: str | None,
    played: tuple[int, int] | None,
    advanced_team: str | None = None,
) -> tuple[str | None, str | None, bool | None]:
    """Actual winner/loser and shoot-out flag from a played knockout score.

    A decisive score determines the winner directly. A level full-time score
    means the tie went to a shoot-out: the winner is taken from ``advanced_team``
    when provided (otherwise it stays undetermined), with the shoot-out flag set."""
    if played is None or team_a is None or team_b is None:
        return None, None, None
    goals_a, goals_b = played
    if goals_a > goals_b:
        return team_a, team_b, False
    if goals_b > goals_a:
        return team_b, team_a, False
    # Level after normal/extra time -> shoot-out; winner from advanced_team if known.
    advanced = (advanced_team or "").strip()
    if advanced == team_a:
        return team_a, team_b, True
    if advanced == team_b:
        return team_b, team_a, True
    return None, None, True


def build_knockout_predictions(
    group_view: pd.DataFrame,
    r32_mapping: pd.DataFrame,
    progression: pd.DataFrame,
    annex: pd.DataFrame,
    scores: pd.DataFrame | None = None,
    actual_group_tables: pd.DataFrame | None = None,
) -> dict:
    """Build projected + current knockout predictions for matches 73-104.

    Returns a JSON-ready dict with one record per knockout match carrying the
    projected up-front gamble, the current recommendation (refreshed from actual
    participants), and any actual result / status. Submitted predictions are not
    touched."""
    # 1) Projected bracket from the frozen group-stage projection.
    source_positions = source_positions_from_group_view(group_view)
    projected_thirds = _projected_qualified_third_groups(group_view)
    projected_r32 = assign_official_r32_matches(
        r32_mapping, source_positions, annex, projected_thirds
    ).sort_values("match_number")
    projected_r32_teams = {
        int(row["match_number"]): (str(row["team_a"]), str(row["team_b"]))
        for _, row in projected_r32.iterrows()
    }
    projected = _walk_bracket(projected_r32_teams, progression, group_view)

    # 2) Actual participants where known (group complete pins the R32).
    group_complete = (
        actual_group_tables is not None and group_stage_complete(actual_group_tables)
    )
    actual_r32_teams: dict[int, tuple[str, str]] = {}
    qualified_third_groups: list[str] = []
    if group_complete:
        actual_r32, qualified_third_groups = actual_round_of_32_from_group_tables(
            actual_group_tables, r32_mapping, annex
        )
        actual_r32_teams = {
            int(row["match_number"]): (str(row["team_a"]), str(row["team_b"]))
            for _, row in actual_r32.iterrows()
        }

    played = _played_knockout_scores(scores)
    advanced_by_match = _knockout_advanced_teams(scores)

    # 3) Resolve actual teams + winners round by round from real results.
    actual_teams: dict[int, tuple[str, str]] = dict(actual_r32_teams)
    actual_winners: dict[int, str] = {}
    actual_losers: dict[int, str] = {}
    actual_shootout: dict[int, bool] = {}

    def _record_actual_result(match_number: int) -> None:
        teams = actual_teams.get(match_number)
        winner, loser, shootout = _actual_winner_loser(
            teams[0] if teams else None,
            teams[1] if teams else None,
            played.get(match_number),
            advanced_by_match.get(match_number),
        )
        if shootout is not None:
            actual_shootout[match_number] = shootout
        if winner is not None:
            actual_winners[match_number] = winner
            actual_losers[match_number] = loser

    for match_number in sorted(actual_r32_teams):
        _record_actual_result(match_number)
    for _, row in progression.sort_values("match_number").iterrows():
        match_number = int(row["match_number"])
        team_a = _resolve(row["team_a_source"], actual_winners, actual_losers)
        team_b = _resolve(row["team_b_source"], actual_winners, actual_losers)
        if team_a is not None and team_b is not None:
            actual_teams[match_number] = (team_a, team_b)
        _record_actual_result(match_number)

    # 4) Assemble per-match records.
    records: list[dict] = []
    for match_number in sorted(ROUND_BY_MATCH):
        round_name = ROUND_BY_MATCH[match_number]
        proj = projected.get(match_number)
        actual_pair = actual_teams.get(match_number)
        score_pair = played.get(match_number)

        # Current recommendation uses the real participants when both are known.
        if actual_pair is not None:
            current = predict_match(actual_pair[0], actual_pair[1], group_view)
            teams_source = "actual"
        elif proj is not None:
            current = proj
            teams_source = "projected"
        else:
            current = None
            teams_source = "pending"

        actual_score = f"{score_pair[0]}-{score_pair[1]}" if score_pair else None
        actual_adv = actual_winners.get(match_number)
        actual_so = actual_shootout.get(match_number)
        if score_pair is not None:
            status = "played"
        elif actual_pair is not None:
            status = "teams_set"
        else:
            status = "projected"

        points = _estimate_points(current, actual_score, actual_adv, actual_so)
        copy_text = None
        if current is not None:
            so_tag = " after shoot-out" if current["shootout"] else ""
            copy_text = (
                f"{match_number}. {current['team_a']} {current['score']} "
                f"{current['team_b']} — adv {current['advancing_team']}{so_tag}"
            )

        records.append(
            {
                "match_number": match_number,
                "round": round_name,
                "round_label": ROUND_LABELS[round_name],
                # Original up-front gamble (projected bracket).
                "projected_team_a": proj["team_a"] if proj else None,
                "projected_team_b": proj["team_b"] if proj else None,
                "projected_score": proj["score"] if proj else None,
                "projected_advancing_team": proj["advancing_team"] if proj else None,
                "projected_shootout": proj["shootout"] if proj else None,
                # Current recommendation (refreshed from actual participants).
                "current_team_a": current["team_a"] if current else None,
                "current_team_b": current["team_b"] if current else None,
                "current_score": current["score"] if current else None,
                "current_advancing_team": current["advancing_team"] if current else None,
                "current_shootout": current["shootout"] if current else None,
                "teams_source": teams_source,
                # Actual result + estimated points once played.
                "actual_score": actual_score,
                "actual_advancing_team": actual_adv,
                "actual_shootout": actual_so,
                "points_earned_estimate": points,
                "status": status,
                "copy_text": copy_text,
            }
        )

    by_round: dict[str, list[dict]] = {name: [] for name in ROUND_ORDER}
    for record in records:
        by_round[record["round"]].append(record)

    copy_lines = [r["copy_text"] for r in records if r["copy_text"]]

    # The "next round to predict" is the earliest knockout round that still has
    # an unplayed match AND at least one match whose two participants are known
    # or projected. The projected bracket always supplies participants, so this
    # is the earliest round not yet fully played; it shows the whole round.
    next_round = None
    for round_name in ROUND_ORDER:
        rows = by_round.get(round_name, [])
        if not rows:
            continue
        has_unplayed = any(r["status"] != "played" for r in rows)
        has_participants = any(r["current_team_a"] and r["current_team_b"] for r in rows)
        if has_unplayed and has_participants:
            next_round = round_name
            break
    next_round_matches = list(by_round.get(next_round, [])) if next_round else []
    next_round_copy = "\n".join(r["copy_text"] for r in next_round_matches if r["copy_text"])

    return {
        "group_stage_complete": bool(group_complete),
        "qualified_third_groups": qualified_third_groups,
        "rounds": ROUND_ORDER,
        "round_labels": ROUND_LABELS,
        "matches": records,
        "matches_by_round": by_round,
        "copy_text": "\n".join(copy_lines),
        "next_round": next_round,
        "next_round_label": ROUND_LABELS.get(next_round) if next_round else None,
        "next_round_matches": next_round_matches,
        "next_round_copy_text": next_round_copy,
        "note": (
            "Knockout match predictions refresh round by round: projected teams "
            "are the up-front gamble; once actual participants are known the "
            "current recommendation uses them. Submitted group-stage predictions "
            "are never modified."
        ),
    }


def _estimate_points(
    current: dict | None,
    actual_score: str | None,
    actual_advancing_team: str | None,
    actual_shootout: bool | None,
) -> float | None:
    """Estimated knockout points for a played match (rules §4b).

    The 6x odd multiplier needs the official knockout template odd, which is not
    in our data, so the correct-team award uses the flat base 6 as an estimate.
    Exact-score (+2) and shoot-out (+2) bonuses are flat and exact."""
    if current is None or actual_score is None:
        return None
    points = 0.0
    if actual_advancing_team is not None and current["advancing_team"] == actual_advancing_team:
        points += 6.0  # base only; odd multiplier unavailable for knockouts
    if current["score"] == actual_score:
        points += 2.0
    if actual_shootout is not None and bool(current["shootout"]) == bool(actual_shootout):
        points += 2.0
    return points
