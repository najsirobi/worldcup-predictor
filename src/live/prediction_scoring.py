"""Score the active candidate's predicted group scores vs actual results (Task E).

Pure functions implementing the group-stage scoring of RULES_AND_SCORING.md
(machine spec: ``data/reference/scoring_rules.yml``):

* **Correct outcome** (home win / draw / away win) pays ``base (6) x template odd``
  of the *predicted* outcome. Rate A = team_a win, Rate Draw = draw, Rate B =
  team_b win (from ``fif8a_group_stage_template.csv``).
* **Exact goal difference** adds a flat ``+2`` -- only when the outcome is correct.
* **Exact score** adds a flat ``+3`` -- only when the outcome is correct.

If the outcome is wrong, all three awards are zero.

``max_possible_points_for_match`` is what a perfect call would have scored:
predicting the exact actual score, i.e. ``6 x odd(actual outcome) + 2 + 3``.
No model is trained or changed here -- this only compares frozen predictions to
manually entered scores.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_PATH = ROOT / "data" / "reference" / "fif8a_group_stage_template.csv"
SCORING_RULES_PATH = ROOT / "data" / "reference" / "scoring_rules.yml"

DETAIL_COLUMNS = [
    "match_number",
    "group",
    "team_a",
    "team_b",
    "status",
    "submitted_score",
    "predicted_score",
    "predicted_team_a_goals",
    "predicted_team_b_goals",
    "actual_score",
    "actual_team_a_goals",
    "actual_team_b_goals",
    "predicted_outcome",
    "actual_outcome",
    "outcome_correct",
    "predicted_goal_difference",
    "actual_goal_difference",
    "goal_difference_correct",
    "exact_score_correct",
    "applicable_odd",
    "outcome_points",
    "goal_difference_bonus",
    "exact_score_bonus",
    "points_earned",
    "total_points",
    "max_possible_points_for_match",
    "points_missed",
    "scoring_explanation",
]


class ScoringRules:
    """Thin holder for the three group-match scoring constants."""

    def __init__(self, base: float, gd_bonus: float, exact_bonus: float):
        self.base = base
        self.gd_bonus = gd_bonus
        self.exact_bonus = exact_bonus


def load_scoring_rules(path: Path = SCORING_RULES_PATH) -> ScoringRules:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ScoringRules(
        base=float(raw["group_match_correct_outcome_base_points"]),
        gd_bonus=float(raw["group_match_exact_goal_difference_bonus"]),
        exact_bonus=float(raw["group_match_exact_score_bonus"]),
    )


def load_template_odds(path: Path = TEMPLATE_PATH) -> dict[int, dict[str, float]]:
    """Map match_number -> {'team_a': rate_a, 'draw': rate_draw, 'team_b': rate_b}."""
    template = pd.read_csv(path)
    odds: dict[int, dict[str, float]] = {}
    for _, row in template.iterrows():
        odds[int(row["match_number"])] = {
            "team_a": float(row["rate_a"]),
            "draw": float(row["rate_draw"]),
            "team_b": float(row["rate_b"]),
        }
    return odds


def parse_score(text: str) -> tuple[int, int]:
    """Parse a scoreline like ``2-1`` (also tolerates en/em dashes) to ints."""
    s = str(text).strip().replace("–", "-").replace("—", "-")
    a, b = s.split("-")
    return int(a.strip()), int(b.strip())


def outcome_of(team_a_goals: int, team_b_goals: int) -> str:
    if team_a_goals > team_b_goals:
        return "team_a"
    if team_a_goals < team_b_goals:
        return "team_b"
    return "draw"


def _round_points(value: float) -> float:
    return round(float(value), 4)


def score_match(
    predicted: tuple[int, int],
    actual: tuple[int, int],
    odds: dict[str, float],
    rules: ScoringRules,
) -> dict:
    """Score one match. ``odds`` keyed by 'team_a'/'draw'/'team_b'."""
    pa, pb = predicted
    aa, ab = actual
    predicted_outcome = outcome_of(pa, pb)
    actual_outcome = outcome_of(aa, ab)
    outcome_correct = predicted_outcome == actual_outcome

    predicted_gd = pa - pb
    actual_gd = aa - ab
    gd_correct = predicted_gd == actual_gd
    exact_correct = (pa == aa) and (pb == ab)

    applicable_odd = odds[predicted_outcome]

    if outcome_correct:
        outcome_points = rules.base * applicable_odd
        gd_bonus = rules.gd_bonus if gd_correct else 0.0
        exact_bonus = rules.exact_bonus if exact_correct else 0.0
    else:
        outcome_points = 0.0
        gd_bonus = 0.0
        exact_bonus = 0.0

    total = outcome_points + gd_bonus + exact_bonus
    # Best achievable: predict the exact actual score -> correct outcome (at the
    # actual outcome's odd) plus both bonuses.
    max_possible = rules.base * odds[actual_outcome] + rules.gd_bonus + rules.exact_bonus
    points_missed = max_possible - total

    explanation = _explain(
        predicted_outcome,
        actual_outcome,
        outcome_correct,
        gd_correct,
        exact_correct,
        rules,
        applicable_odd,
        outcome_points,
        gd_bonus,
        exact_bonus,
    )

    return {
        "predicted_team_a_goals": pa,
        "predicted_team_b_goals": pb,
        "actual_team_a_goals": aa,
        "actual_team_b_goals": ab,
        "predicted_score": f"{pa}-{pb}",
        "actual_score": f"{aa}-{ab}",
        "predicted_outcome": predicted_outcome,
        "actual_outcome": actual_outcome,
        "outcome_correct": bool(outcome_correct),
        "predicted_goal_difference": predicted_gd,
        "actual_goal_difference": actual_gd,
        "goal_difference_correct": bool(gd_correct),
        "exact_score_correct": bool(exact_correct),
        "applicable_odd": applicable_odd,
        "outcome_points": _round_points(outcome_points),
        "goal_difference_bonus": _round_points(gd_bonus),
        "exact_score_bonus": _round_points(exact_bonus),
        "total_points": _round_points(total),
        "max_possible_points_for_match": _round_points(max_possible),
        "points_missed": _round_points(points_missed),
        "scoring_explanation": explanation,
    }


def _explain(
    predicted_outcome,
    actual_outcome,
    outcome_correct,
    gd_correct,
    exact_correct,
    rules,
    odd,
    outcome_points,
    gd_bonus,
    exact_bonus,
) -> str:
    if not outcome_correct:
        return (
            f"Outcome wrong (predicted {predicted_outcome}, actual {actual_outcome}): "
            "0 points; goal-difference and exact-score bonuses do not apply."
        )
    parts = [
        f"Outcome correct ({actual_outcome}): {rules.base:g} x odd {odd:g} = "
        f"{_round_points(outcome_points):g}"
    ]
    if exact_correct:
        parts.append(f"exact score +{rules.exact_bonus:g}")
        parts.append(f"exact goal difference +{rules.gd_bonus:g}")
    elif gd_correct:
        parts.append(f"exact goal difference +{rules.gd_bonus:g}")
    else:
        parts.append("no goal-difference or exact-score bonus")
    return "; ".join(parts) + f" = {_round_points(outcome_points + gd_bonus + exact_bonus):g} pts."


def score_predictions_vs_actuals(
    predictions: pd.DataFrame,
    scores: pd.DataFrame,
    odds: dict[int, dict[str, float]],
    rules: ScoringRules,
) -> pd.DataFrame:
    """Build the per-match prediction-vs-actual detail frame for played matches."""
    pred_by_match = {int(r["match_number"]): r for _, r in predictions.iterrows()}
    played = scores[scores["status"] == "played"].copy()

    rows: list[dict] = []
    for _, s in played.sort_values("match_number").iterrows():
        mn = int(s["match_number"])
        if mn not in pred_by_match:
            continue
        pred_row = pred_by_match[mn]
        if mn not in odds:
            raise ValueError(f"No template odds for match {mn}.")
        predicted = parse_score(pred_row["final_recommended_score"])
        actual = (int(s["team_a_goals"]), int(s["team_b_goals"]))
        scored = score_match(predicted, actual, odds[mn], rules)
        scored.update(
            {
                "match_number": mn,
                "group": s["group"],
                "team_a": s["team_a"],
                "team_b": s["team_b"],
                "status": "locked/submitted",
                "submitted_score": scored["predicted_score"],
                "points_earned": scored["total_points"],
            }
        )
        rows.append(scored)

    return pd.DataFrame(rows, columns=DETAIL_COLUMNS)


def summarise(detail: pd.DataFrame) -> dict:
    """Aggregate the detail frame into the scoring summary payload."""
    played = len(detail)
    total_points = float(detail["total_points"].sum()) if played else 0.0
    possible = float(detail["max_possible_points_for_match"].sum()) if played else 0.0
    missed = float(detail["points_missed"].sum()) if played else 0.0

    by_group = {}
    if played:
        grouped = detail.groupby("group").agg(
            played_matches=("match_number", "count"),
            total_points=("total_points", "sum"),
            possible_points=("max_possible_points_for_match", "sum"),
            outcomes_correct=("outcome_correct", "sum"),
            goal_differences_correct=("goal_difference_correct", "sum"),
            exact_scores_correct=("exact_score_correct", "sum"),
        )
        for group, row in grouped.iterrows():
            by_group[str(group)] = {
                "played_matches": int(row["played_matches"]),
                "total_points": _round_points(row["total_points"]),
                "possible_points": _round_points(row["possible_points"]),
                "points_missed": _round_points(row["possible_points"] - row["total_points"]),
                "outcomes_correct": int(row["outcomes_correct"]),
                "goal_differences_correct": int(row["goal_differences_correct"]),
                "exact_scores_correct": int(row["exact_scores_correct"]),
            }

    return {
        "played_matches": played,
        "exact_scores_correct": int(detail["exact_score_correct"].sum()) if played else 0,
        "goal_differences_correct": int(detail["goal_difference_correct"].sum()) if played else 0,
        "outcomes_correct": int(detail["outcome_correct"].sum()) if played else 0,
        "total_points": _round_points(total_points),
        "possible_points_for_played_matches": _round_points(possible),
        "points_missed": _round_points(missed),
        "average_points_per_played_match": _round_points(total_points / played) if played else 0.0,
        "total_by_group": by_group,
    }
