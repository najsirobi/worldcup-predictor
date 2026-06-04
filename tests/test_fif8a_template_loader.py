"""Test the FIF8A group-stage template validator with synthetic data."""
import pandas as pd
import pytest

from src.ingest.fif8a_template import (
    load_fif8a_group_template,
    validate_fif8a_group_template,
    TEMPLATE_COLUMNS,
    GROUPS,
)


def test_real_committed_template_still_validates():
    """The committed data/reference/fif8a_group_stage_template.csv must stay valid."""
    df = load_fif8a_group_template()
    validate_fif8a_group_template(df, require_full=True)
    assert len(df) == 72


def _synthetic_full_template():
    """Build a structurally-valid 72-match template (12 groups × 6)."""
    rows = []
    n = 1
    for g in GROUPS:
        for _ in range(6):
            rows.append({
                "match_number": n, "group": g, "date": "2026-06-11",
                "team_a": f"{g}_TeamA", "rate_a": 1.5, "rate_draw": 3.0,
                "rate_b": 2.5, "team_b": f"{g}_TeamB",
                "source": "test", "source_date": "2026-06-03", "notes": "",
            })
            n += 1
    return pd.DataFrame(rows, columns=TEMPLATE_COLUMNS)


def test_full_synthetic_template_passes():
    validate_fif8a_group_template(_synthetic_full_template(), require_full=True)


def test_missing_column_raises():
    df = _synthetic_full_template().drop(columns=["rate_draw"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_fif8a_group_template(df)


def test_negative_odd_raises():
    df = _synthetic_full_template()
    df.loc[0, "rate_a"] = -1.0
    with pytest.raises(ValueError, match="non-positive odds"):
        validate_fif8a_group_template(df)


def test_null_team_raises():
    df = _synthetic_full_template()
    df.loc[0, "team_b"] = None
    with pytest.raises(ValueError, match="null team names"):
        validate_fif8a_group_template(df)


def test_wrong_match_count_raises_when_full_required():
    df = _synthetic_full_template().iloc[:-1]  # 71 matches
    with pytest.raises(ValueError, match="expected 72 matches"):
        validate_fif8a_group_template(df, require_full=True)


def test_duplicate_match_number_raises():
    df = _synthetic_full_template()
    df.loc[1, "match_number"] = df.loc[0, "match_number"]
    with pytest.raises(ValueError, match="duplicate match_number"):
        validate_fif8a_group_template(df)


def test_partial_allowed_when_not_full():
    # 2 groups only, but structurally valid -> passes with require_full=False
    df = _synthetic_full_template()
    df = df[df["group"].isin(["A", "B"])].reset_index(drop=True)
    validate_fif8a_group_template(df, require_full=False)
