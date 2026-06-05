"""Draw-aware score selection helpers.

These helpers are deliberately small: they do not generate probabilities, train
models, or mutate final candidate files. They only decide whether an already
computed draw candidate should replace the current selected score under explicit
thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from src.evaluation.auto_consensus import parse_score, score_outcome


@dataclass(frozen=True)
class DrawAwareConfig:
    """Thresholds for the conservative draw-aware hybrid policy."""

    min_draw_probability: float = 0.30
    max_selected_probability_edge: float = 0.08
    min_expected_points_uplift: float = 0.15
    modal_ev_competitive_margin: float = 0.15


def is_draw_score(score: object) -> bool:
    """Return True when a score string has equal Team A / Team B goals."""

    goals_a, goals_b = parse_score(score)
    return goals_a == goals_b


def selected_outcome_probability(row: Mapping[str, object], score: object) -> float:
    """Probability assigned to the W/D/L outcome implied by ``score``."""

    outcome = score_outcome(score)
    column_by_outcome = {
        "a_win": "model_p_a_win",
        "draw": "model_p_draw",
        "b_win": "model_p_b_win",
    }
    return float(row[column_by_outcome[outcome]])


def choose_draw_aware_hybrid_score(
    row: Mapping[str, object],
    current_score: str,
    best_draw_score: str,
    expected_points_current: float,
    expected_points_best_draw: float,
    modal_score: str,
    config: DrawAwareConfig = DrawAwareConfig(),
) -> tuple[str, str]:
    """Return the hybrid score and a machine-readable reason.

    The policy starts from ``current_score`` and only switches to the best draw
    score when one of the explicit draw-aware conditions is met:

    * draw probability is at least 0.30 and close to the selected outcome;
    * draw expected points exceed the current score by a meaningful margin;
    * the draw is modal and expected-points competitive with the current score.
    """

    if is_draw_score(current_score):
        return current_score, "already_draw"

    model_p_draw = float(row["model_p_draw"])
    selected_probability = selected_outcome_probability(row, current_score)
    probability_close = (
        model_p_draw >= config.min_draw_probability
        and selected_probability - model_p_draw <= config.max_selected_probability_edge
    )
    expected_points_uplift = (
        expected_points_best_draw
        >= expected_points_current + config.min_expected_points_uplift
    )
    modal_and_competitive = (
        is_draw_score(modal_score)
        and str(modal_score) == str(best_draw_score)
        and expected_points_best_draw
        >= expected_points_current - config.modal_ev_competitive_margin
    )

    if probability_close:
        return best_draw_score, "draw_probability_close_to_selected_outcome"
    if expected_points_uplift:
        return best_draw_score, "draw_expected_points_uplift"
    if modal_and_competitive:
        return best_draw_score, "draw_modal_and_ev_competitive"
    return current_score, "keep_current"
