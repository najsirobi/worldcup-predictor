"""Test the scoring-rules config loader/validator."""
import pytest

from src.ingest.rules_and_scoring import (
    load_scoring_rules,
    validate_scoring_rules,
    REQUIRED_NUMERIC_FIELDS,
)


def _valid_rules():
    rules = {f: 1 for f in REQUIRED_NUMERIC_FIELDS}
    rules["odds_are_template_derived"] = True
    rules["odds_source"] = "test"
    return rules


def test_valid_rules_pass():
    validate_scoring_rules(_valid_rules())  # should not raise


def test_missing_field_raises():
    rules = _valid_rules()
    del rules["winner_points"]
    with pytest.raises(ValueError, match="winner_points"):
        validate_scoring_rules(rules)


def test_non_numeric_field_raises():
    rules = _valid_rules()
    rules["qf_team_points"] = "twenty"
    with pytest.raises(ValueError, match="qf_team_points"):
        validate_scoring_rules(rules)


def test_boolean_not_accepted_as_number():
    rules = _valid_rules()
    rules["sf_team_points"] = True  # bool must not count as a numeric score
    with pytest.raises(ValueError, match="sf_team_points"):
        validate_scoring_rules(rules)


def test_real_scoring_rules_file_loads_and_validates():
    """The committed data/reference/scoring_rules.yml must be valid."""
    rules = load_scoring_rules()
    validate_scoring_rules(rules)
    # spot-check the headline values from RULES_AND_SCORING.md
    assert rules["group_match_correct_outcome_base_points"] == 6
    assert rules["group_match_exact_score_bonus"] == 3
    assert rules["group_top2_any_order_points"] == 30
    assert rules["group_exact_standing_bonus"] == 60
    assert rules["winner_points"] == 100
    assert rules["odds_are_template_derived"] is True
