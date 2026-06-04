"""Model-matrix feature engineering (Phase 4).

Builds a model-ready table from the rated match backbone:
- tournament category / flags
- time trend
- strict no-future-leakage rolling pre-match form (uses only matches with
  date < current match date; the current match is always excluded).

No model is trained here. No future fixtures are used.
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Rolling-form columns produced per team (then emitted as home_/away_/_diff).
FORM_STATS = [
    "ppm_5", "ppm_10", "gf_5", "ga_5", "gf_10", "ga_10", "gd_10",
    "win_rate_10", "draw_rate_10", "loss_rate_10",
    "clean_sheet_rate_10", "failed_to_score_rate_10",
]


def categorize_tournament(name: str) -> str:
    """Map a raw tournament string to a coarse category."""
    n = str(name).lower()
    if n == "friendly":
        return "friendly"
    if "world cup qualification" in n:
        return "world_cup_qualifier"
    if n == "fifa world cup":
        return "world_cup"
    if "qualification" in n:
        return "continental_qualifier"
    if "nations league" in n:
        return "nations_league"
    continental = [
        "uefa euro", "copa américa", "copa america", "african cup of nations",
        "afc asian cup", "gold cup", "concacaf", "ofc nations",
    ]
    if any(k in n for k in continental):
        return "continental_championship"
    return "other"


def add_tournament_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["tournament_category"] = out["tournament"].map(categorize_tournament)
    out["is_world_cup"] = (out["tournament"] == "FIFA World Cup").astype(int)
    out["is_world_cup_qualifier"] = out["tournament_category"].eq("world_cup_qualifier").astype(int)
    out["is_continental_championship"] = out["tournament_category"].eq("continental_championship").astype(int)
    out["is_friendly"] = out["tournament_category"].eq("friendly").astype(int)
    return out


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["match_year"] = out["date"].dt.year
    out["days_since_2000"] = (out["date"] - pd.Timestamp("2000-01-01")).dt.days
    return out


def add_context_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Hypothesis 1: home/away effect interacted with match context.

    Home advantage should NOT be transferred blindly into (neutral) World Cup
    games, so we expose per-context home-advantage indicators the models can
    weight independently (and we report the empirical effect by context).
    """
    out = df.copy()
    nonneutral = (~out["neutral"].astype(bool)).astype(int)
    out["nonneutral"] = nonneutral
    out["home_adv_friendly"] = nonneutral * out["is_friendly"]
    out["home_adv_world_cup_qualifier"] = nonneutral * out["is_world_cup_qualifier"]
    out["home_adv_continental"] = nonneutral * out["is_continental_championship"]
    out["home_adv_world_cup"] = nonneutral * out["is_world_cup"]
    out["neutral_world_cup_context"] = out["neutral"].astype(int) * out["is_world_cup"]
    return out


def _elo_expected_score(home_elo, away_elo):
    """Elo expected score for home team in [0,1] (win=1, draw=0.5, loss=0)."""
    return 1.0 / (1.0 + 10 ** ((away_elo - home_elo) / 400.0))


def compute_rolling_form(matches: pd.DataFrame) -> pd.DataFrame:
    """Compute strict pre-match rolling form for home & away teams.

    Returns the input frame with home_/away_/diff form columns added. The
    current match is excluded via shift(1); rolling windows only look backward,
    so no current-match or future information enters a feature.
    """
    m = matches.copy().reset_index(drop=True)
    m["match_id"] = np.arange(len(m))
    m["date"] = pd.to_datetime(m["date"])

    # Hypothesis 3: Elo-expected score residual (measurable over/under-performance).
    if "home_elo" in m.columns and "away_elo" in m.columns:
        exp_h = _elo_expected_score(m["home_elo"], m["away_elo"])
        m["home_elo_resid"] = (m["home_points"].map({3: 1.0, 1: 0.5, 0: 0.0}) - exp_h)
        m["away_elo_resid"] = (m["away_points"].map({3: 1.0, 1: 0.5, 0: 0.0}) - (1 - exp_h))
    else:
        m["home_elo_resid"] = np.nan
        m["away_elo_resid"] = np.nan

    home = m[["match_id", "date", "home_team", "home_score", "away_score", "home_points", "home_elo_resid"]].rename(
        columns={"home_team": "team", "home_score": "gf", "away_score": "ga", "home_points": "pts",
                 "home_elo_resid": "elo_resid"})
    home["side"] = "home"
    away = m[["match_id", "date", "away_team", "away_score", "home_score", "away_points", "away_elo_resid"]].rename(
        columns={"away_team": "team", "away_score": "gf", "home_score": "ga", "away_points": "pts",
                 "away_elo_resid": "elo_resid"})
    away["side"] = "away"
    lng = pd.concat([home, away], ignore_index=True)

    lng["win"] = (lng["pts"] == 3).astype(float)
    lng["draw"] = (lng["pts"] == 1).astype(float)
    lng["loss"] = (lng["pts"] == 0).astype(float)
    lng["clean_sheet"] = (lng["ga"] == 0).astype(float)
    lng["fts"] = (lng["gf"] == 0).astype(float)

    # Deterministic chronological order per team; current match excluded by shift(1).
    lng = lng.sort_values(["team", "date", "match_id"]).reset_index(drop=True)
    g = lng.groupby("team", sort=False)

    def roll(col, w, fn="mean"):
        return g[col].transform(lambda s: getattr(s.shift(1).rolling(w, min_periods=1), fn)())

    lng["ppm_5"] = roll("pts", 5)
    lng["ppm_10"] = roll("pts", 10)
    lng["gf_5"] = roll("gf", 5)
    lng["ga_5"] = roll("ga", 5)
    lng["gf_10"] = roll("gf", 10)
    lng["ga_10"] = roll("ga", 10)
    lng["gd_10"] = lng["gf_10"] - lng["ga_10"]
    lng["win_rate_10"] = roll("win", 10)
    lng["draw_rate_10"] = roll("draw", 10)
    lng["loss_rate_10"] = roll("loss", 10)
    lng["clean_sheet_rate_10"] = roll("clean_sheet", 10)
    lng["failed_to_score_rate_10"] = roll("fts", 10)
    # Hypothesis 3: recent over/under-performance vs Elo expectation (prior 10).
    lng["overperf_elo_10"] = roll("elo_resid", 10)
    # number of prior matches actually available (for missingness reporting)
    lng["prior_matches"] = g.cumcount()

    extra = FORM_STATS + ["overperf_elo_10", "prior_matches"]
    home_f = lng[lng["side"] == "home"].set_index("match_id")[extra].add_prefix("home_")
    away_f = lng[lng["side"] == "away"].set_index("match_id")[extra].add_prefix("away_")

    out = m.set_index("match_id").join(home_f).join(away_f).reset_index(drop=True)

    for s in FORM_STATS + ["overperf_elo_10"]:
        out[f"{s}_diff"] = out[f"home_{s}"] - out[f"away_{s}"]
    return out


def build_model_matrix(rated: pd.DataFrame) -> pd.DataFrame:
    """Assemble the full model matrix from the rated match backbone."""
    df = add_tournament_features(rated)
    df = add_time_features(df)
    df = add_context_interactions(df)
    df = compute_rolling_form(df)
    return df
