"""Tests proving rolling form has no future/current-match leakage."""
import numpy as np
import pandas as pd

from src.features.model_matrix import compute_rolling_form


def _matches():
    # Team A plays 3 dated matches (home), known points; one unrelated match.
    return pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"]),
        "home_team": ["A", "A", "A", "X"],
        "away_team": ["B", "C", "D", "Y"],
        "home_score": [2, 0, 3, 1],
        "away_score": [0, 0, 1, 1],
        "home_points": [3, 1, 3, 1],
        "away_points": [0, 1, 0, 1],
    })


def test_rolling_uses_only_previous_matches():
    out = compute_rolling_form(_matches())
    a = out[out["home_team"] == "A"].sort_values("date").reset_index(drop=True)
    # match 1: no prior -> NaN
    assert pd.isna(a.loc[0, "home_ppm_10"])
    # match 2: prior = {match1 (3 pts)} -> ppm = 3
    assert a.loc[1, "home_ppm_10"] == 3.0
    # match 3: prior = {3, 1} -> ppm = 2.0  (current match 3 EXCLUDED)
    assert a.loc[2, "home_ppm_10"] == 2.0


def test_current_match_excluded_from_its_own_form():
    out = compute_rolling_form(_matches())
    a = out[out["home_team"] == "A"].sort_values("date").reset_index(drop=True)
    # match 3 scored 3 pts; if current were included ppm would be (3+1+3)/3=2.33, not 2.0
    assert a.loc[2, "home_ppm_10"] != (3 + 1 + 3) / 3


def test_future_match_does_not_change_past_features():
    base = compute_rolling_form(_matches())
    a_base = base[base["home_team"] == "A"].sort_values("date")["home_ppm_10"].tolist()

    extended = _matches()
    extended = pd.concat([extended, pd.DataFrame({
        "date": [pd.Timestamp("2021-01-01")], "home_team": ["A"], "away_team": ["Z"],
        "home_score": [5], "away_score": [0], "home_points": [3], "away_points": [0],
    })], ignore_index=True)
    ext = compute_rolling_form(extended)
    a_ext = ext[ext["home_team"] == "A"].sort_values("date")["home_ppm_10"].tolist()
    # the three original matches keep identical form despite a later match existing
    np.testing.assert_array_equal(np.array(a_base), np.array(a_ext[:3]))
