"""Load and validate the FIF8A scoring/objective specification.

The scoring rules live in ``data/reference/scoring_rules.yml`` and are derived
from ``Rules of the game/RULES_AND_SCORING.md``. The model's objective is to
maximise expected points under these rules; this module makes them available as
structured, validated configuration.
"""
import logging
from pathlib import Path

import yaml

from .common import get_data_dir

logger = logging.getLogger(__name__)

# Required numeric scoring fields (every one must be present and a number).
REQUIRED_NUMERIC_FIELDS = [
    "group_match_correct_outcome_base_points",
    "group_match_exact_goal_difference_bonus",
    "group_match_exact_score_bonus",
    "group_top2_any_order_points",
    "group_exact_standing_bonus",
    "qf_team_points",
    "sf_team_points",
    "finalist_points",
    "winner_points",
    "knockout_correct_qualified_team_base_points",
    "knockout_exact_score_bonus",
    "knockout_penalty_shootout_bonus",
]

REQUIRED_META_FIELDS = [
    "odds_are_template_derived",
    "odds_source",
]


def load_scoring_rules(path: Path = None) -> dict:
    """Load the scoring rules YAML into a dict.

    Raises FileNotFoundError if the spec file is missing.
    """
    if path is None:
        path = get_data_dir("reference") / "scoring_rules.yml"
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Scoring rules file not found: {path}. "
            "Expected the structured spec derived from RULES_AND_SCORING.md."
        )
    with open(path) as f:
        rules = yaml.safe_load(f)
    if not isinstance(rules, dict):
        raise ValueError(f"Scoring rules file {path} did not parse to a mapping")
    logger.info(f"Loaded scoring rules from {path}")
    return rules


def validate_scoring_rules(rules: dict) -> None:
    """Validate that all required scoring fields are present and well-typed.

    Raises ValueError listing every missing/invalid field, so the spec can be
    fixed in one pass.
    """
    problems = []

    for field in REQUIRED_NUMERIC_FIELDS:
        if field not in rules:
            problems.append(f"missing required field '{field}'")
        elif not isinstance(rules[field], (int, float)) or isinstance(rules[field], bool):
            problems.append(f"field '{field}' must be a number, got {rules[field]!r}")
        elif rules[field] < 0:
            problems.append(f"field '{field}' must be non-negative, got {rules[field]!r}")

    for field in REQUIRED_META_FIELDS:
        if field not in rules:
            problems.append(f"missing required field '{field}'")

    if "odds_are_template_derived" in rules and not isinstance(
        rules["odds_are_template_derived"], bool
    ):
        problems.append("field 'odds_are_template_derived' must be a boolean")

    if problems:
        raise ValueError(
            "Invalid scoring rules specification:\n  - " + "\n  - ".join(problems)
        )
