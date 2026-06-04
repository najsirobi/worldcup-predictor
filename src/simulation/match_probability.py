"""Reusable neutral match probability fallback for knockout simulation.

This is not a retrained model. It converts existing pre-tournament team strength
signals into a conservative W/D/L and advancement probability for arbitrary
team-vs-team knockout pairs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class MatchProbability:
    team_a: str
    team_b: str
    p_team_a_win_in_90: float
    p_draw_90: float
    p_team_b_win_in_90: float
    p_team_a_advance: float
    p_team_b_advance: float
    expected_goals_a: float
    expected_goals_b: float
    notes: str


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _team_strength_from_group_view(group_view: pd.DataFrame) -> dict[str, float]:
    required = {"team", "expected_points", "expected_goal_difference", "p_top2"}
    missing = required - set(group_view.columns)
    if missing:
        raise ValueError(f"Group view missing strength columns: {sorted(missing)}")
    strength = {}
    for _, row in group_view.iterrows():
        strength[str(row["team"])] = (
            float(row["expected_points"]) * 0.55
            + float(row["expected_goal_difference"]) * 0.30
            + float(row["p_top2"]) * 1.50
        )
    return strength


def knockout_match_probability(
    team_a: str,
    team_b: str,
    group_view: pd.DataFrame,
    *,
    base_draw_probability: float = 0.24,
) -> MatchProbability:
    """Return neutral knockout probabilities from existing group-simulation strength.

    Draws after 90 minutes are resolved by a strength-adjusted advancement share.
    This is documented as a pragmatic fallback, not a penalty shootout model.
    """
    strength = _team_strength_from_group_view(group_view)
    if team_a not in strength or team_b not in strength:
        missing = [team for team in [team_a, team_b] if team not in strength]
        raise ValueError(f"Missing team strength for: {missing}")

    diff = strength[team_a] - strength[team_b]
    non_draw_share_a = _sigmoid(diff / 2.4)
    draw = max(0.16, min(0.30, base_draw_probability - min(abs(diff), 5.0) * 0.015))
    non_draw = 1.0 - draw
    p_a_90 = non_draw * non_draw_share_a
    p_b_90 = non_draw * (1.0 - non_draw_share_a)
    draw_adv_share_a = _sigmoid(diff / 3.0)
    p_a_adv = p_a_90 + draw * draw_adv_share_a
    p_b_adv = 1.0 - p_a_adv
    expected_goals_a = max(0.35, 1.25 + diff * 0.10)
    expected_goals_b = max(0.35, 1.25 - diff * 0.10)

    return MatchProbability(
        team_a=team_a,
        team_b=team_b,
        p_team_a_win_in_90=round(float(p_a_90), 6),
        p_draw_90=round(float(draw), 6),
        p_team_b_win_in_90=round(float(p_b_90), 6),
        p_team_a_advance=round(float(p_a_adv), 6),
        p_team_b_advance=round(float(p_b_adv), 6),
        expected_goals_a=round(float(expected_goals_a), 4),
        expected_goals_b=round(float(expected_goals_b), 4),
        notes=(
            "Fallback from existing group-simulation strength; 90-minute draws "
            "resolved by strength-adjusted advancement share, not a penalty model."
        ),
    )
