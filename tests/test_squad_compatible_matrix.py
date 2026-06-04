"""Tests for the squad-compatible model matrix builder (Phase 5C, Task F/J)."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
SCRIPT = ROOT / "scripts" / "build_model_matrix_squad_compatible.py"
MATRIX = ROOT / "data" / "processed" / "model_matrix_squad_compatible.parquet"
TARGETS = ["home_score", "away_score", "result_label", "home_goals", "away_goals"]


def _load_builder():
    spec = importlib.util.spec_from_file_location("squad_matrix_builder", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_squad_diff_cols_are_comparable_only():
    from src.features.historical_squad_features import COMPARABLE_FEATURE_COLUMNS
    mod = _load_builder()
    # The matrix may only carry the comparable (age + position) squad features.
    assert mod.SQUAD_DIFF_COLS == COMPARABLE_FEATURE_COLUMNS
    banned = ("height", "club", "domestic", "foreign", "market", "value")
    for col in mod.SQUAD_DIFF_COLS:
        assert not any(b in col for b in banned), col


def test_tournament_name_only_for_world_cup():
    mod = _load_builder()
    wc = pd.Series({"tournament": "FIFA World Cup", "match_year": 2018})
    fr = pd.Series({"tournament": "Friendly", "match_year": 2018})
    assert mod._tournament_name(wc) == "2018 FIFA World Cup"
    assert mod._tournament_name(fr) == ""


def test_join_preserves_nulls_for_missing_side():
    mod = _load_builder()
    base = pd.DataFrame({
        "tournament_name": ["2018 FIFA World Cup", ""],
        "home_team": ["A", "X"],
    })
    feats = pd.DataFrame({
        "tournament_name": ["2018 FIFA World Cup"],
        "team": ["A"],
        "squad_avg_age": [27.0],
        "has_historical_squad_features": [True],
    })
    out = mod._join_side(base, feats, "home", ["tournament_name"],
                         ["squad_avg_age", "has_historical_squad_features"], "home_squad_")
    assert out.loc[0, "home_squad_squad_avg_age"] == 27.0
    # unmatched (non-WC) row stays null, not zero
    assert pd.isna(out.loc[1, "home_squad_squad_avg_age"])


@pytest.mark.skipif(not MATRIX.exists(), reason="matrix not built yet")
def test_built_matrix_preserves_targets_and_has_no_current_only_cols():
    df = pd.read_parquet(MATRIX)
    for t in TARGETS:
        assert t in df.columns, t
    forbidden = [c for c in df.columns if any(b in c for b in ("height", "club", "domestic", "foreign"))]
    assert forbidden == [], forbidden


@pytest.mark.skipif(not MATRIX.exists(), reason="matrix not built yet")
def test_non_world_cup_rows_have_no_squad_features():
    df = pd.read_parquet(MATRIX)
    non_wc = df[df["tournament"] != "FIFA World Cup"]
    assert not non_wc["has_squad_features"].fillna(False).any()
