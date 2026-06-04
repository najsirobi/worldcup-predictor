"""Scoring functions for the FIF8A World Cup 2026 prediction game.

These implement *how a prediction is scored* under the rules captured in
``data/reference/scoring_rules.yml`` / ``RULES_AND_SCORING.md``. They are the
building blocks for the eventual expected-points objective.

Scope for this phase:
- Deterministic scoring of a given prediction vs. a given actual outcome.
- A helper to compute expected points of a *fixed* prediction under a supplied
  scoreline probability distribution.

Out of scope (NOT implemented here yet):
- Choosing the points-maximising prediction (argmax / optimisation).
- Any trained model producing the probability distributions.
"""
import logging
from typing import Iterable, Mapping

logger = logging.getLogger(__name__)

# Default point values mirroring data/reference/scoring_rules.yml, so these
# functions are usable (and unit-testable) without loading the YAML. Pass an
# explicit ``rules`` dict (e.g. from load_scoring_rules()) to override.
DEFAULT_RULES = {
    "group_match_correct_outcome_base_points": 6,
    "group_match_exact_goal_difference_bonus": 2,
    "group_match_exact_score_bonus": 3,
    "group_top2_any_order_points": 30,
    "group_exact_standing_bonus": 60,
    "qf_team_points": 20,
    "sf_team_points": 40,
    "finalist_points": 60,
    "winner_points": 100,
    "knockout_correct_qualified_team_base_points": 6,
    "knockout_exact_score_bonus": 2,
    "knockout_penalty_shootout_bonus": 2,
}


def _rules(rules: Mapping = None) -> Mapping:
    return DEFAULT_RULES if rules is None else rules


def match_outcome(home_score: int, away_score: int) -> str:
    """Map a scoreline to an outcome label: 'home', 'draw' or 'away'."""
    if home_score > away_score:
        return "home"
    if home_score < away_score:
        return "away"
    return "draw"


def score_group_match_prediction(
    pred_home: int,
    pred_away: int,
    actual_home: int,
    actual_away: int,
    odds: Mapping,
    rules: Mapping = None,
) -> float:
    """Score one group-stage match prediction.

    Args:
        pred_home, pred_away: predicted scoreline.
        actual_home, actual_away: realised scoreline.
        odds: mapping with keys 'home', 'draw', 'away' = template odd (rate) for
            each outcome.
        rules: optional scoring-rule overrides; defaults to DEFAULT_RULES.

    Points: ``base * odd`` for a correct outcome, plus a flat goal-difference
    bonus and a flat exact-score bonus — both only when the outcome is correct.
    """
    r = _rules(rules)
    pred_outcome = match_outcome(pred_home, pred_away)
    actual_outcome = match_outcome(actual_home, actual_away)

    if pred_outcome != actual_outcome:
        return 0.0

    points = float(r["group_match_correct_outcome_base_points"]) * float(odds[pred_outcome])
    if (pred_home - pred_away) == (actual_home - actual_away):
        points += float(r["group_match_exact_goal_difference_bonus"])
    if pred_home == actual_home and pred_away == actual_away:
        points += float(r["group_match_exact_score_bonus"])
    return points


def expected_group_match_points(
    pred_home: int,
    pred_away: int,
    scoreline_probs: Mapping,
    odds: Mapping,
    rules: Mapping = None,
) -> float:
    """Expected points of a FIXED prediction under a scoreline distribution.

    Args:
        pred_home, pred_away: the (already chosen) predicted scoreline.
        scoreline_probs: mapping {(home, away): probability} over actual results.
        odds: template odds mapping ('home'/'draw'/'away').

    This evaluates a given prediction; it does NOT search for the best one.
    """
    return float(
        sum(
            prob * score_group_match_prediction(
                pred_home, pred_away, ah, aa, odds, rules
            )
            for (ah, aa), prob in scoreline_probs.items()
        )
    )


def score_group_standing_prediction(
    pred_order: list,
    actual_order: list,
    rules: Mapping = None,
) -> float:
    """Score a predicted group standing (list of 4 teams, position 1..4).

    30 points for the correct top-2 (any order); +60 if the full 1-2-3-4 order
    is exact (so a perfect group = 90).
    """
    r = _rules(rules)
    points = 0.0
    if set(pred_order[:2]) == set(actual_order[:2]):
        points += float(r["group_top2_any_order_points"])
    if list(pred_order) == list(actual_order):
        points += float(r["group_exact_standing_bonus"])
    return points


def score_last8_prediction(
    pred: Mapping,
    actual: Mapping,
    rules: Mapping = None,
) -> float:
    """Score the last-8 progression block.

    ``pred`` / ``actual`` are mappings with keys:
        'quarter_finalists' (8 teams), 'semi_finalists' (4), 'finalists' (2),
        'winner' (single team).
    Points are awarded per correctly-named team at each stage, plus the winner.
    """
    r = _rules(rules)
    points = 0.0
    points += float(r["qf_team_points"]) * len(
        set(pred.get("quarter_finalists", [])) & set(actual.get("quarter_finalists", []))
    )
    points += float(r["sf_team_points"]) * len(
        set(pred.get("semi_finalists", [])) & set(actual.get("semi_finalists", []))
    )
    points += float(r["finalist_points"]) * len(
        set(pred.get("finalists", [])) & set(actual.get("finalists", []))
    )
    if pred.get("winner") is not None and pred.get("winner") == actual.get("winner"):
        points += float(r["winner_points"])
    return points


def score_knockout_match_prediction(
    pred_qualified_team,
    actual_qualified_team,
    exact_score_correct: bool = False,
    penalty_shootout_correct: bool = False,
    odd: float = 1.0,
    rules: Mapping = None,
) -> float:
    """Score one knockout-stage match prediction (used in a later phase).

    base * odd for naming the correct qualifying team, plus a flat exact-score
    bonus (end of extra time) and a flat correct-penalty-shoot-out-call bonus.
    The score/shoot-out bonuses are passed as pre-evaluated booleans to keep
    this skeleton independent of the (not-yet-defined) knockout scoreline format.
    """
    r = _rules(rules)
    points = 0.0
    if pred_qualified_team == actual_qualified_team:
        points += float(r["knockout_correct_qualified_team_base_points"]) * float(odd)
    if exact_score_correct:
        points += float(r["knockout_exact_score_bonus"])
    if penalty_shootout_correct:
        points += float(r["knockout_penalty_shootout_bonus"])
    return points
