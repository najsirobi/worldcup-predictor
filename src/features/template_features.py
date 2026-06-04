"""Shared helpers to build template-match feature rows and scoreline matrices.

Used by prediction, audit, and simulation scripts so Team A / Team B orientation
stays consistent across every Phase 4.5 output. WC matches are treated as
neutral, matching the Phase 4 modelling choice.
"""
import numpy as np
import pandas as pd

from src.models.baselines import POISSON_HOME_FEATURES, POISSON_AWAY_FEATURES

# Template (FIFA-style) name -> backbone name. Explicit, documented; not inferred.
TEMPLATE_TO_BACKBONE = {
    "Korea Republic": "South Korea", "Czechia": "Czech Republic", "Türkiye": "Turkey",
    "Côte d'Ivoire": "Ivory Coast", "IR Iran": "Iran", "Cabo Verde": "Cape Verde",
    "Congo DR": "DR Congo", "USA": "United States",
}


def build_team_snapshots(mm: pd.DataFrame) -> pd.DataFrame:
    """Latest pre-cutoff feature snapshot per backbone team from the model matrix."""
    def side(s):
        cols = {
            "date": "date",
            f"{s}_team": "team",
            f"{s}_elo": "elo",
            f"{s}_fifa_points": "fifa_points",
            f"{s}_fifa_rank": "fifa_rank",
            f"{s}_ppm_5": "ppm_5",
            f"{s}_ppm_10": "ppm_10",
            f"{s}_gf_10": "gf_10",
            f"{s}_ga_10": "ga_10",
            f"{s}_gd_10": "gd_10",
            f"{s}_win_rate_10": "win_rate_10",
            f"{s}_overperf_elo_10": "overperf_elo_10",
            f"{s}_fifa_ranking_date": "fifa_ranking_date",
        }
        return mm[list(cols)].rename(columns=cols)
    lng = pd.concat([side("home"), side("away")], ignore_index=True).sort_values("date")
    return lng.groupby("team").tail(1).set_index("team")


def feature_row(a_snap, b_snap, match_date) -> pd.DataFrame:
    days = (pd.Timestamp(match_date) - pd.Timestamp("2000-01-01")).days
    return pd.DataFrame([{
        "home_elo": a_snap["elo"], "away_elo": b_snap["elo"],
        "elo_diff": a_snap["elo"] - b_snap["elo"],
        "home_fifa_points": a_snap["fifa_points"], "away_fifa_points": b_snap["fifa_points"],
        "fifa_points_diff": a_snap["fifa_points"] - b_snap["fifa_points"],
        "home_fifa_rank": a_snap["fifa_rank"], "away_fifa_rank": b_snap["fifa_rank"],
        "fifa_rank_diff": a_snap["fifa_rank"] - b_snap["fifa_rank"],
        "home_ppm_5": a_snap["ppm_5"], "away_ppm_5": b_snap["ppm_5"],
        "ppm_5_diff": a_snap["ppm_5"] - b_snap["ppm_5"],
        "home_gf_10": a_snap["gf_10"], "away_ga_10": b_snap["ga_10"],
        "away_gf_10": b_snap["gf_10"], "home_ga_10": a_snap["ga_10"],
        "home_ppm_10": a_snap["ppm_10"], "away_ppm_10": b_snap["ppm_10"],
        "ppm_10_diff": a_snap["ppm_10"] - b_snap["ppm_10"],
        "gf_10_diff": a_snap["gf_10"] - b_snap["gf_10"],
        "ga_10_diff": a_snap["ga_10"] - b_snap["ga_10"],
        "gd_10_diff": a_snap["gd_10"] - b_snap["gd_10"],
        "win_rate_10_diff": a_snap["win_rate_10"] - b_snap["win_rate_10"],
        "overperf_elo_10_diff": a_snap["overperf_elo_10"] - b_snap["overperf_elo_10"],
        "neutral": True,
        "is_world_cup": 1,
        "is_friendly": 0,
        "is_world_cup_qualifier": 0,
        "is_continental_championship": 0,
        "nonneutral": 0,
        "home_adv_friendly": 0,
        "home_adv_world_cup_qualifier": 0,
        "home_adv_continental": 0,
        "home_adv_world_cup": 0,
        "neutral_world_cup_context": 1,
        "tournament_category": "world_cup",
        "days_since_2000": days,
    }])


def resolve(name, snaps):
    b = TEMPLATE_TO_BACKBONE.get(name, name)
    return b if b in snaps.index else None


def match_lambdas_matrix(poisson, snaps, team_a, team_b, match_date):
    """Return (lam_home, lam_away, scoreline_matrix) or None if a team is unresolved."""
    ba, bb = resolve(team_a, snaps), resolve(team_b, snaps)
    if ba is None or bb is None:
        return None
    X = feature_row(snaps.loc[ba], snaps.loc[bb], match_date)
    lh = float(np.clip(poisson.home_pipe.predict(X[POISSON_HOME_FEATURES]), 1e-3, 12)[0])
    la = float(np.clip(poisson.away_pipe.predict(X[POISSON_AWAY_FEATURES]), 1e-3, 12)[0])
    return lh, la, poisson.score_matrix(lh, la)
