#!/usr/bin/env python3
"""Analyse country-level residual over/underperformance vs model expectations.

This is an analysis/reporting script only. It does not modify production models,
bracket mappings, final prediction files, or final submission outputs.

Expectations are generated with expanding, year-start Poisson baseline models:
each match in year Y is predicted by a model fit only on played matches before
January 1 of year Y. This is conservative versus a true pre-match rolling model
and avoids using later matches from the same calendar year.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.baselines import CLASSES, PoissonScoreModel


ROOT = Path(__file__).parent.parent
MATRIX = ROOT / "data" / "processed" / "model_matrix_baseline.parquet"
RESIDUALS_OUT = ROOT / "data" / "interim" / "country_match_residuals.parquet"
TABLE_OUT = ROOT / "outputs" / "predictions" / "country_overperformance_table.csv"

REPORT_DATA = ROOT / "outputs" / "reports" / "country_residuals_data_report.md"
REPORT_COUNTRY = ROOT / "outputs" / "reports" / "country_overperformance_report.md"
REPORT_STABILITY = ROOT / "outputs" / "reports" / "country_residual_stability_report.md"
REPORT_EXPLAIN = ROOT / "outputs" / "reports" / "country_overperformance_explanation_hypotheses.md"
REPORT_RECS = ROOT / "outputs" / "reports" / "country_residual_model_recommendations.md"
REPORT_LIFT = ROOT / "outputs" / "reports" / "country_residual_feature_lift_report.md"

RNG = np.random.default_rng(20260604)


def md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    """Small markdown table helper without relying on optional tabulate."""
    if max_rows is not None:
        df = df.head(max_rows)
    if df.empty:
        return "_No rows._\n"
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in df.itertuples(index=False):
        vals = []
        for v in row:
            if pd.isna(v):
                vals.append("")
            elif isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out) + "\n"


def load_name_map() -> dict[str, str]:
    path = ROOT / "data" / "reference" / "team_name_map.csv"
    if not path.exists():
        return {}
    m = pd.read_csv(path)
    mapping = dict(zip(m["raw_name"], m["canonical_team_name"]))
    mapping.update(dict(zip(m["canonical_team_name"], m["canonical_team_name"])))
    return mapping


def _prep_matrix() -> pd.DataFrame:
    df = pd.read_parquet(MATRIX)
    df["date"] = pd.to_datetime(df["date"])
    df = df[
        (df["match_year"] >= 2000)
        & df["elo_diff"].notna()
        & df["home_goals"].notna()
        & df["away_goals"].notna()
        & df["result_label"].isin(CLASSES)
    ].copy()
    df = df.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    df["analysis_match_id"] = np.arange(len(df))
    return df


def _world_cup_metadata(name_map: dict[str, str]) -> pd.DataFrame:
    path = ROOT / "data" / "raw" / "kaggle" / "world_cup_database" / "matches.csv"
    if not path.exists():
        return pd.DataFrame()
    wc = pd.read_csv(path)
    wc["date"] = pd.to_datetime(wc["match_date"])
    wc["home_team"] = wc["home_team_name"].map(name_map).fillna(wc["home_team_name"])
    wc["away_team"] = wc["away_team_name"].map(name_map).fillna(wc["away_team_name"])
    keep = [
        "date",
        "home_team",
        "away_team",
        "home_team_score",
        "away_team_score",
        "stage_name",
        "group_stage",
        "knockout_stage",
        "extra_time",
        "penalty_shootout",
        "home_team_score_penalties",
        "away_team_score_penalties",
    ]
    wc = wc[keep].rename(
        columns={
            "home_team_score": "home_goals",
            "away_team_score": "away_goals",
            "stage_name": "wc_stage_name",
            "group_stage": "is_group_stage",
            "knockout_stage": "is_knockout",
            "penalty_shootout": "went_to_penalties",
        }
    )
    for c in ["home_goals", "away_goals"]:
        wc[c] = pd.to_numeric(wc[c], errors="coerce")
    return wc


def _team_metadata(name_map: dict[str, str]) -> pd.DataFrame:
    path = ROOT / "data" / "raw" / "kaggle" / "world_cup_database" / "teams.csv"
    if not path.exists():
        return pd.DataFrame(columns=["team", "confederation_code", "region_name"])
    teams = pd.read_csv(path)
    teams["team"] = teams["team_name"].map(name_map).fillna(teams["team_name"])
    return teams[["team", "confederation_code", "region_name"]].drop_duplicates("team")


def _squad_features() -> pd.DataFrame:
    path = ROOT / "data" / "interim" / "historical_squad_features.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _coach_features() -> pd.DataFrame:
    path = ROOT / "data" / "interim" / "historical_coach_features.parquet"
    if not path.exists():
        return pd.DataFrame()
    c = pd.read_parquet(path)
    c["match_date"] = pd.to_datetime(c["match_date"])
    return c


def add_wc_metadata(df: pd.DataFrame) -> pd.DataFrame:
    name_map = load_name_map()
    wc = _world_cup_metadata(name_map)
    if wc.empty:
        out = df.copy()
        out["is_knockout"] = np.nan
        out["is_group_stage"] = np.nan
        out["wc_stage_name"] = pd.NA
        out["went_to_penalties"] = np.nan
        return out

    keys = ["date", "home_team", "away_team", "home_goals", "away_goals"]
    out = df.merge(wc, on=keys, how="left")
    out["is_knockout"] = out["is_knockout"].fillna(0).astype(int)
    out["is_group_stage"] = out["is_group_stage"].fillna(out["tournament"].eq("FIFA World Cup").astype(int)).astype(int)
    out["went_to_penalties"] = out["went_to_penalties"].fillna(0).astype(int)
    return out


def predict_year_start(df: pd.DataFrame, min_train_rows: int = 500) -> pd.DataFrame:
    rows = []
    years = sorted(df["match_year"].dropna().astype(int).unique())
    for year in years:
        cutoff = pd.Timestamp(year=year, month=1, day=1)
        train = df[df["date"] < cutoff].copy()
        test = df[df["match_year"] == year].copy()
        if len(train) < min_train_rows or test.empty:
            continue
        model = PoissonScoreModel().fit(train)
        lh, la = model.predict_lambdas(test)
        probs = model.predict_proba(test)
        pred = test[["analysis_match_id"]].copy()
        pred["lambda_home"] = lh
        pred["lambda_away"] = la
        pred["p_home_win"] = probs[:, 0]
        pred["p_draw"] = probs[:, 1]
        pred["p_away_win"] = probs[:, 2]
        pred["prediction_train_through"] = str((cutoff - pd.Timedelta(days=1)).date())
        pred["prediction_train_rows"] = len(train)
        pred["source_model_used"] = "expanding_year_start_poisson_baseline"
        rows.append(pred)
    if not rows:
        raise RuntimeError("No year-start predictions were generated; check input coverage.")
    return pd.concat(rows, ignore_index=True)


def build_residual_rows(df: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    m = df.merge(pred, on="analysis_match_id", how="inner").copy()
    team_rows = []
    side_cols = {
        "home": {
            "team": "home_team",
            "opponent": "away_team",
            "goals_for": "home_goals",
            "goals_against": "away_goals",
            "points": "home_points",
            "p_win": "p_home_win",
            "p_loss": "p_away_win",
            "elo": "home_elo",
            "opp_elo": "away_elo",
            "fifa_points": "home_fifa_points",
            "opp_fifa_points": "away_fifa_points",
            "expected_gf": "lambda_home",
            "expected_ga": "lambda_away",
            "prefix": "home_",
        },
        "away": {
            "team": "away_team",
            "opponent": "home_team",
            "goals_for": "away_goals",
            "goals_against": "home_goals",
            "points": "away_points",
            "p_win": "p_away_win",
            "p_loss": "p_home_win",
            "elo": "away_elo",
            "opp_elo": "home_elo",
            "fifa_points": "away_fifa_points",
            "opp_fifa_points": "home_fifa_points",
            "expected_gf": "lambda_away",
            "expected_ga": "lambda_home",
            "prefix": "away_",
        },
    }
    form_names = [
        "ppm_5",
        "ppm_10",
        "gf_5",
        "ga_5",
        "gf_10",
        "ga_10",
        "gd_10",
        "win_rate_10",
        "draw_rate_10",
        "loss_rate_10",
        "clean_sheet_rate_10",
        "failed_to_score_rate_10",
        "overperf_elo_10",
        "prior_matches",
    ]
    base_cols = [
        "analysis_match_id",
        "date",
        "tournament",
        "tournament_category",
        "is_world_cup",
        "is_knockout",
        "is_group_stage",
        "wc_stage_name",
        "went_to_penalties",
        "neutral",
        "prediction_train_through",
        "prediction_train_rows",
        "source_model_used",
    ]
    for side, spec in side_cols.items():
        r = m[base_cols].copy()
        r["side"] = side
        r["team"] = m[spec["team"]]
        r["opponent"] = m[spec["opponent"]]
        r["team_goals"] = m[spec["goals_for"]]
        r["opponent_goals"] = m[spec["goals_against"]]
        r["actual_points"] = m[spec["points"]]
        r["model_p_win"] = m[spec["p_win"]]
        r["model_p_draw"] = m["p_draw"]
        r["model_p_loss"] = m[spec["p_loss"]]
        r["expected_points"] = 3 * r["model_p_win"] + r["model_p_draw"]
        r["points_residual"] = r["actual_points"] - r["expected_points"]
        r["actual_goal_diff"] = r["team_goals"] - r["opponent_goals"]
        r["expected_goals_for"] = m[spec["expected_gf"]]
        r["expected_goals_against"] = m[spec["expected_ga"]]
        r["expected_goal_diff"] = r["expected_goals_for"] - r["expected_goals_against"]
        r["goals_for_residual"] = r["team_goals"] - r["expected_goals_for"]
        r["goals_against_residual"] = r["opponent_goals"] - r["expected_goals_against"]
        r["goal_diff_residual"] = r["actual_goal_diff"] - r["expected_goal_diff"]
        r["predicted_result_probability"] = np.select(
            [r["actual_points"].eq(3), r["actual_points"].eq(1), r["actual_points"].eq(0)],
            [r["model_p_win"], r["model_p_draw"], r["model_p_loss"]],
            default=np.nan,
        )
        r["outcome_surprise"] = 1 - r["predicted_result_probability"]
        r["elo"] = m[spec["elo"]]
        r["fifa_points"] = m[spec["fifa_points"]]
        r["opponent_strength"] = m[spec["opp_elo"]]
        r["opponent_fifa_points"] = m[spec["opp_fifa_points"]]
        for name in form_names:
            r[name] = m.get(spec["prefix"] + name)
        team_rows.append(r)
    out = pd.concat(team_rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out["match_year"] = out["date"].dt.year
    out["era"] = pd.cut(
        out["match_year"],
        bins=[1999, 2009, 2019, 2026],
        labels=["2000-2009", "2010-2019", "2020-2026"],
    )
    out["world_cup_year"] = np.where(out["is_world_cup"].eq(1), out["match_year"], np.nan)
    out["is_favourite"] = out["expected_points"] > 1.5
    out["is_underdog"] = out["expected_points"] < 1.2
    return out.sort_values(["date", "analysis_match_id", "side"]).reset_index(drop=True)


def bootstrap_ci(vals: pd.Series, n_boot: int = 500) -> tuple[float, float]:
    x = vals.dropna().to_numpy(dtype=float)
    if len(x) < 3:
        return (np.nan, np.nan)
    idx = RNG.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def aggregate_country(res: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for team, g in res.groupby("team", sort=False):
        wc = g[g["is_world_cup"].eq(1)]
        ci = bootstrap_ci(g["points_residual"])
        wc_ci = bootstrap_ci(wc["points_residual"])
        matches = len(g)
        wc_matches = len(wc)
        raw = g["points_residual"].mean()
        wc_raw = wc["points_residual"].mean() if wc_matches else np.nan
        rows.append(
            {
                "team": team,
                "matches": matches,
                "world_cup_matches": wc_matches,
                "avg_expected_points": g["expected_points"].mean(),
                "avg_actual_points": g["actual_points"].mean(),
                "total_points_residual": g["points_residual"].sum(),
                "points_residual_per_match_raw": raw,
                "points_residual_per_match_shrunk": raw * matches / (matches + 20),
                "points_residual_ci_low": ci[0],
                "points_residual_ci_high": ci[1],
                "world_cup_points_residual_per_match_raw": wc_raw,
                "world_cup_points_residual_per_match_shrunk": (
                    wc_raw * wc_matches / (wc_matches + 8) if wc_matches else np.nan
                ),
                "world_cup_points_residual_ci_low": wc_ci[0],
                "world_cup_points_residual_ci_high": wc_ci[1],
                "goal_diff_residual_per_match": g["goal_diff_residual"].mean(),
                "world_cup_goal_diff_residual_per_match": wc["goal_diff_residual"].mean() if wc_matches else np.nan,
                "outcome_surprise_avg": g["outcome_surprise"].mean(),
                "exact_score_or_outcome_residual_available": "outcome_only; exact score residual not stored",
                "all_tournaments_sample_flag": "ok" if matches >= 15 else "small_sample",
                "world_cup_sample_flag": "ok" if wc_matches >= 8 else "small_sample",
            }
        )
    out = pd.DataFrame(rows)
    out["overperformance_rank"] = out["points_residual_per_match_shrunk"].rank(ascending=False, method="min")
    out["underperformance_rank"] = out["points_residual_per_match_shrunk"].rank(ascending=True, method="min")
    return out.sort_values("points_residual_per_match_shrunk", ascending=False).reset_index(drop=True)


def slice_summary(res: pd.DataFrame, keys: list[str], min_n: int = 3) -> pd.DataFrame:
    g = res.groupby(keys + ["team"], dropna=False)["points_residual"].agg(["count", "mean"]).reset_index()
    return g[g["count"] >= min_n].rename(columns={"count": "matches", "mean": "points_residual_per_match"})


def consistent_by_slice(s: pd.DataFrame, slice_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for team, g in s.groupby("team"):
        signs = np.sign(g["points_residual_per_match"])
        pos = int((signs > 0).sum())
        neg = int((signs < 0).sum())
        rows.append(
            {
                "team": team,
                "slices": len(g),
                "positive_slices": pos,
                "negative_slices": neg,
                "avg_slice_residual": g["points_residual_per_match"].mean(),
                "slice_detail": "; ".join(
                    f"{row[slice_col]}={row['points_residual_per_match']:.2f} ({int(row['matches'])})"
                    for _, row in g.iterrows()
                ),
            }
        )
    d = pd.DataFrame(rows)
    over = d[(d["slices"] >= 2) & (d["positive_slices"] >= 2) & (d["negative_slices"] == 0)].sort_values(
        "avg_slice_residual", ascending=False
    )
    under = d[(d["slices"] >= 2) & (d["negative_slices"] >= 2) & (d["positive_slices"] == 0)].sort_values(
        "avg_slice_residual"
    )
    return over, under


def one_tournament_artefacts(res: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    wc = res[res["is_world_cup"].eq(1)].copy()
    rows = []
    for team, g in wc.groupby("team"):
        if len(g) < 8:
            continue
        total = g["points_residual"].sum()
        by_year = g.groupby("world_cup_year")["points_residual"].sum()
        if total == 0 or by_year.empty:
            continue
        biggest_year = by_year.abs().idxmax()
        share = abs(by_year.loc[biggest_year]) / max(by_year.abs().sum(), 1e-9)
        rows.append(
            {
                "team": team,
                "world_cup_matches": len(g),
                "world_cup_total_residual": total,
                "largest_driver_year": int(biggest_year),
                "largest_driver_residual": by_year.loc[biggest_year],
                "share_of_absolute_wc_residual": share,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out[out["share_of_absolute_wc_residual"] >= 0.65].sort_values("share_of_absolute_wc_residual", ascending=False)


def explanation_metrics(res: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    candidates = pd.concat([table.head(10), table.tail(10)]).drop_duplicates("team")
    rows = []
    teams_meta = _team_metadata(load_name_map())
    squads = _squad_features()
    coaches = _coach_features()
    for team in candidates["team"]:
        g = res[res["team"].eq(team)]
        wc = g[g["is_world_cup"].eq(1)]
        ko = wc[wc["is_knockout"].eq(1)]
        group = wc[wc["is_group_stage"].eq(1)]
        squad_team = pd.DataFrame()
        if not squads.empty:
            squad_team = squads[squads["team"].eq(team)]
        coach_team = pd.DataFrame()
        if not coaches.empty:
            coach_team = coaches[coaches["team"].eq(team)]
        rows.append(
            {
                "team": team,
                "matches": len(g),
                "shrunk_points_residual": table.loc[table["team"].eq(team), "points_residual_per_match_shrunk"].iloc[0],
                "goals_for_residual_per_match": g["goals_for_residual"].mean(),
                "goals_against_residual_per_match": g["goals_against_residual"].mean(),
                "goal_diff_residual_per_match": g["goal_diff_residual"].mean(),
                "actual_total_goals_avg": (g["team_goals"] + g["opponent_goals"]).mean(),
                "world_cup_residual_per_match": wc["points_residual"].mean() if len(wc) else np.nan,
                "group_stage_residual_per_match": group["points_residual"].mean() if len(group) else np.nan,
                "knockout_residual_per_match": ko["points_residual"].mean() if len(ko) else np.nan,
                "penalty_shootout_matches": int(wc["went_to_penalties"].sum()) if len(wc) else 0,
                "neutral_residual_per_match": g[g["neutral"].astype(bool)]["points_residual"].mean(),
                "rating_momentum_avg": g["overperf_elo_10"].mean(),
                "favourite_residual_per_match": g[g["is_favourite"]]["points_residual"].mean(),
                "underdog_residual_per_match": g[g["is_underdog"]]["points_residual"].mean(),
                "wc_squad_avg_age": squad_team["squad_avg_age"].mean() if not squad_team.empty else np.nan,
                "wc_squad_fw_share": squad_team["squad_fw_share"].mean() if not squad_team.empty else np.nan,
                "wc_coach_tenure_days_avg": (
                    coach_team["coach_tenure_days_at_tournament_start"].mean() if not coach_team.empty else np.nan
                ),
                "wc_coach_prior_experience_avg": (
                    coach_team["prior_world_cup_experience"].mean() if not coach_team.empty else np.nan
                ),
            }
        )
    out = pd.DataFrame(rows)
    if not teams_meta.empty:
        out = out.merge(teams_meta, on="team", how="left")
    return out.sort_values("shrunk_points_residual", ascending=False)


def feature_lift_prototype(res: pd.DataFrame) -> pd.DataFrame:
    """Lightweight experimental country residual feature on held-out WCs.

    The feature for each team before a match is an expanding, prior-match-only
    average points residual, shrunk by n/(n+30). A simple logistic residual model
    is fitted on baseline Poisson log-odds plus residual-difference for WC2018
    and WC2022. This is diagnostic only and is not saved as a production model.
    """
    match = res[res["side"].eq("home")].copy()
    match["result_label"] = np.select(
        [match["actual_points"].eq(3), match["actual_points"].eq(1), match["actual_points"].eq(0)],
        ["home_win", "draw", "away_win"],
        default=pd.NA,
    )
    match = match.rename(
        columns={
            "date": "date_home",
            "match_year": "match_year_home",
            "model_p_win": "model_p_win_home",
            "model_p_draw": "model_p_draw_home",
            "model_p_loss": "model_p_loss_home",
        }
    ).sort_values("date_home")

    long = res.sort_values(["team", "date", "analysis_match_id"]).copy()
    long["prior_sum"] = long.groupby("team")["points_residual"].cumsum() - long["points_residual"]
    long["prior_n"] = long.groupby("team").cumcount()
    long["country_residual_rating"] = (long["prior_sum"] / long["prior_n"].replace(0, np.nan)).fillna(0)
    long["country_residual_rating"] *= long["prior_n"] / (long["prior_n"] + 30)
    feat = long.pivot_table(
        index="analysis_match_id", columns="side", values="country_residual_rating", aggfunc="first"
    ).reset_index()
    feat["country_residual_rating_diff"] = feat["home"] - feat["away"]
    match = match.merge(feat[["analysis_match_id", "country_residual_rating_diff"]], on="analysis_match_id", how="left")

    rows = []
    eps = 1e-6
    for year in [2018, 2022]:
        train = match[(pd.to_datetime(match["date_home"]) < pd.Timestamp(f"{year}-01-01")) & match["result_label"].isin(CLASSES)]
        test = match[(match["tournament"].eq("FIFA World Cup")) & (match["match_year_home"].eq(year))]
        if len(test) == 0 or len(train) < 500:
            continue
        for c in ["model_p_win_home", "model_p_draw_home", "model_p_loss_home"]:
            train[c] = train[c].clip(eps, 1 - eps)
            test[c] = test[c].clip(eps, 1 - eps)
        X_base = pd.DataFrame(
            {
                "logit_home": np.log(train["model_p_win_home"] / train["model_p_loss_home"]),
                "logit_draw": np.log(train["model_p_draw_home"] / train["model_p_loss_home"]),
            }
        )
        X_plus = X_base.assign(country_residual_rating_diff=train["country_residual_rating_diff"].fillna(0))
        Xt_base = pd.DataFrame(
            {
                "logit_home": np.log(test["model_p_win_home"] / test["model_p_loss_home"]),
                "logit_draw": np.log(test["model_p_draw_home"] / test["model_p_loss_home"]),
            }
        )
        Xt_plus = Xt_base.assign(country_residual_rating_diff=test["country_residual_rating_diff"].fillna(0))

        y_train = train["result_label"]
        y_test = test["result_label"]
        base_probs = test[["model_p_win_home", "model_p_draw_home", "model_p_loss_home"]].to_numpy()
        lr = Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, C=0.2)),
            ]
        ).fit(X_base, y_train)
        lr_plus = Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, C=0.2)),
            ]
        ).fit(X_plus, y_train)
        probs_lr = _proba_ordered(lr, Xt_base)
        probs_plus = _proba_ordered(lr_plus, Xt_plus)
        rows.extend(
            [
                {
                    "backtest": f"WC{year}",
                    "model": "year_start_poisson_baseline",
                    "log_loss": _ordered_log_loss(y_test, base_probs),
                    "avg_p_actual": _avg_p_actual(y_test, base_probs),
                    "notes": "no extra training; raw year-start Poisson probabilities",
                },
                {
                    "backtest": f"WC{year}",
                    "model": "calibrated_baseline_logit",
                    "log_loss": _ordered_log_loss(y_test, probs_lr),
                    "avg_p_actual": _avg_p_actual(y_test, probs_lr),
                    "notes": "diagnostic calibration model",
                },
                {
                    "backtest": f"WC{year}",
                    "model": "calibrated_plus_country_residual_rating",
                    "log_loss": _ordered_log_loss(y_test, probs_plus),
                    "avg_p_actual": _avg_p_actual(y_test, probs_plus),
                    "notes": "experimental; prior-match-only, shrunk n/(n+30)",
                },
            ]
        )
    return pd.DataFrame(rows)


def current_wc2026_review_shortlist(
    table: pd.DataFrame,
    era_over: pd.DataFrame,
    era_under: pd.DataFrame,
    wc_over: pd.DataFrame,
    wc_under: pd.DataFrame,
) -> pd.DataFrame:
    """Flag current WC2026 teams whose historical residuals merit manual review."""
    sim_path = ROOT / "outputs" / "predictions" / "full_tournament_simulation_summary.csv"
    group_path = ROOT / "outputs" / "predictions" / "final_group_score_predictions.csv"
    teams: set[str] = set()
    if sim_path.exists():
        teams.update(pd.read_csv(sim_path)["team"].dropna().astype(str).tolist())
    if group_path.exists():
        gp = pd.read_csv(group_path)
        for col in ["team_a", "team_b"]:
            if col in gp.columns:
                teams.update(gp[col].dropna().astype(str).tolist())
    if not teams:
        return pd.DataFrame(columns=["team", "residual_flag", "points_residual_per_match_shrunk", "world_cup_points_residual_per_match_shrunk"])

    over_era = set(era_over["team"])
    under_era = set(era_under["team"])
    over_wc = set(wc_over["team"])
    under_wc = set(wc_under["team"])
    rows = []
    sub = table[table["team"].isin(teams)].copy()
    for _, row in sub.iterrows():
        flags = []
        team = row["team"]
        if team in over_era:
            flags.append("stable_all-era_overperformer")
        if team in under_era:
            flags.append("stable_all-era_underperformer")
        if team in over_wc:
            flags.append("stable_wc_overperformer")
        if team in under_wc:
            flags.append("stable_wc_underperformer")
        if abs(row["points_residual_per_match_shrunk"]) >= 0.10:
            flags.append("large_shrunk_all-match_residual")
        wc_val = row.get("world_cup_points_residual_per_match_shrunk")
        if pd.notna(wc_val) and abs(wc_val) >= 0.20:
            flags.append("large_shrunk_world-cup_residual")
        if flags:
            rows.append(
                {
                    "team": team,
                    "residual_flag": "; ".join(flags),
                    "matches": int(row["matches"]),
                    "world_cup_matches": int(row["world_cup_matches"]),
                    "points_residual_per_match_shrunk": row["points_residual_per_match_shrunk"],
                    "world_cup_points_residual_per_match_shrunk": wc_val,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("points_residual_per_match_shrunk", ascending=False).head(30)


def _proba_ordered(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    classes = list(model.named_steps["clf"].classes_)
    p = model.predict_proba(X)
    return p[:, [classes.index(c) for c in CLASSES]]


def _avg_p_actual(y: pd.Series, p: np.ndarray) -> float:
    idx = {c: i for i, c in enumerate(CLASSES)}
    return float(np.mean([p[i, idx[v]] for i, v in enumerate(y)]))


def _ordered_log_loss(y: pd.Series, p: np.ndarray, eps: float = 1e-15) -> float:
    idx = {c: i for i, c in enumerate(CLASSES)}
    clipped = np.clip(p, eps, 1 - eps)
    clipped = clipped / clipped.sum(axis=1, keepdims=True)
    vals = [clipped[i, idx[v]] for i, v in enumerate(y)]
    return float(-np.mean(np.log(vals)))


def write_reports(res: pd.DataFrame, table: pd.DataFrame, lift: pd.DataFrame) -> None:
    RESIDUALS_OUT.parent.mkdir(parents=True, exist_ok=True)
    TABLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_DATA.parent.mkdir(parents=True, exist_ok=True)

    top_raw = table.sort_values("points_residual_per_match_raw", ascending=False).head(10)
    top_shrunk = table.sort_values("points_residual_per_match_shrunk", ascending=False).head(10)
    bottom_raw = table.sort_values("points_residual_per_match_raw").head(10)
    bottom_shrunk = table.sort_values("points_residual_per_match_shrunk").head(10)
    era = slice_summary(res, ["era"], min_n=5)
    wc_year = slice_summary(res[res["world_cup_year"].isin([2010, 2014, 2018, 2022])], ["world_cup_year"], min_n=2)
    stage = slice_summary(res[res["is_world_cup"].eq(1)], ["is_group_stage", "is_knockout"], min_n=2)
    fav = slice_summary(
        res.assign(
            favourite_bucket=np.where(res["is_favourite"], "favourite", np.where(res["is_underdog"], "underdog", "balanced"))
        ),
        ["favourite_bucket"],
        min_n=5,
    )
    tournament_cat = slice_summary(res[~res["tournament_category"].eq("friendly")], ["tournament_category"], min_n=8)
    era_over, era_under = consistent_by_slice(era, "era")
    wc_over, wc_under = consistent_by_slice(wc_year, "world_cup_year")
    artefacts = one_tournament_artefacts(res, table)
    current_2026 = current_wc2026_review_shortlist(table, era_over, era_under, wc_over, wc_under)

    with open(REPORT_DATA, "w") as f:
        f.write("# Country Residuals Data Report\n\n")
        f.write("## Method\n\n")
        f.write(
            "- Built one row per team per historical match from `data/processed/model_matrix_baseline.parquet`.\n"
            "- Included played matches from 2000 onward with Elo features and generated model probabilities.\n"
            "- Expectations use expanding year-start Poisson baseline models. For every year Y, models are fit only on matches before January 1 of Y.\n"
            "- This is analysis-time scoring, not a production model refresh and not a change to final predictions.\n"
            "- World Cup group/knockout/stage metadata is joined from `data/raw/kaggle/world_cup_database/matches.csv` where names and scores match explicitly.\n\n"
        )
        f.write("## Output Coverage\n\n")
        f.write(f"- Residual rows: **{len(res)}**\n")
        f.write(f"- Unique teams: **{res['team'].nunique()}**\n")
        f.write(f"- Match date range: **{res['date'].min().date()}** to **{res['date'].max().date()}**\n")
        f.write(f"- World Cup team-rows: **{int(res['is_world_cup'].sum())}**\n")
        f.write(f"- World Cup knockout team-rows: **{int((res['is_world_cup'].eq(1) & res['is_knockout'].eq(1)).sum())}**\n")
        f.write(f"- Group-stage team-rows: **{int(res['is_group_stage'].sum())}**\n")
        f.write("\n## Columns\n\n")
        f.write(", ".join(res.columns) + "\n")

    with open(REPORT_COUNTRY, "w") as f:
        f.write("# Country Overperformance Report\n\n")
        f.write("Positive residual means the country earned more points than the model expected. Shrinkage is `raw_mean * n/(n+k)`, with k=20 for all matches and k=8 for World Cup-only summaries.\n\n")
        f.write("## Top 10 Raw Overperformers\n\n")
        f.write(md_table(top_raw[["team", "matches", "points_residual_per_match_raw", "points_residual_per_match_shrunk", "goal_diff_residual_per_match"]]))
        f.write("\n## Top 10 Shrunk Overperformers\n\n")
        f.write(md_table(top_shrunk[["team", "matches", "points_residual_per_match_raw", "points_residual_per_match_shrunk", "points_residual_ci_low", "points_residual_ci_high"]]))
        f.write("\n## Top 10 Raw Underperformers\n\n")
        f.write(md_table(bottom_raw[["team", "matches", "points_residual_per_match_raw", "points_residual_per_match_shrunk", "goal_diff_residual_per_match"]]))
        f.write("\n## Top 10 Shrunk Underperformers\n\n")
        f.write(md_table(bottom_shrunk[["team", "matches", "points_residual_per_match_raw", "points_residual_per_match_shrunk", "points_residual_ci_low", "points_residual_ci_high"]]))
        f.write("\n## Interpretation Guardrails\n\n")
        f.write("- Countries below 15 all-match rows or below 8 World Cup rows are explicitly flagged as small-sample.\n")
        f.write("- Confidence intervals are bootstrap intervals over team-match residuals, not proof of persistent skill.\n")
        f.write("- Residuals can reflect model misspecification, rating lag, schedule composition, home/neutral context, or noise.\n")
        f.write("\n## Final Summary\n\n")
        f.write("Stable all-era overperformers include: " + ", ".join(era_over.head(10)["team"].astype(str).tolist()) + ".\n\n")
        f.write("Stable all-era underperformers include: " + ", ".join(era_under.head(10)["team"].astype(str).tolist()) + ".\n\n")
        f.write("Stable World Cup 2010-2022 overperformers include: " + ", ".join(wc_over.head(10)["team"].astype(str).tolist()) + ".\n\n")
        f.write("Stable World Cup 2010-2022 underperformers include: " + ", ".join(wc_under.head(10)["team"].astype(str).tolist()) + ".\n\n")
        f.write("One-tournament artefact flags are listed in the stability report; they are based on one World Cup contributing at least 65% of absolute WC residual movement.\n\n")
        f.write("Likely explanations to investigate are rating lag, defensive/goals-against residuals, tournament-stage splits, confederation calibration, and path/penalty effects. None are causal claims.\n\n")
        f.write("A country residual feature is worth testing further, but the lightweight prototype does not justify promotion or current prediction changes.\n\n")
        f.write("WC2026 manual-review shortlist from residual history only:\n\n")
        f.write(md_table(current_2026))

    with open(REPORT_STABILITY, "w") as f:
        f.write("# Country Residual Stability Report\n\n")
        f.write("## Consistent Across Eras\n\n")
        f.write("Overperformers:\n\n")
        f.write(md_table(era_over.head(15)))
        f.write("\nUnderperformers:\n\n")
        f.write(md_table(era_under.head(15)))
        f.write("\n## Consistent Across World Cups 2010-2022\n\n")
        f.write("Overperformers:\n\n")
        f.write(md_table(wc_over.head(15)))
        f.write("\nUnderperformers:\n\n")
        f.write(md_table(wc_under.head(15)))
        f.write("\n## Potential One-Tournament Artefacts\n\n")
        f.write(md_table(artefacts.head(20) if not artefacts.empty else artefacts))
        f.write("\n## Group Stage vs Knockout Stage\n\n")
        f.write(md_table(stage.sort_values(["team", "is_knockout"]).head(40)))
        f.write("\n## Favourite vs Underdog Matches\n\n")
        f.write(md_table(fav.sort_values(["team", "favourite_bucket"]).head(40)))
        f.write("\n## Tournament / High-Pressure Categories\n\n")
        f.write(md_table(tournament_cat.sort_values(["team", "tournament_category"]).head(60)))
        f.write("\n## Read\n\n")
        f.write("- A country is labelled stable only if the residual sign repeats across at least two qualified slices without an opposite qualified slice.\n")
        f.write("- Knockout samples are small; use them as flags for review, not as standalone evidence.\n")

    expl = explanation_metrics(res, table)
    with open(REPORT_EXPLAIN, "w") as f:
        f.write("# Country Overperformance Explanation Hypotheses\n\n")
        f.write("These are possible explanations consistent with the available features, not causal claims.\n\n")
        f.write("## Candidate Metrics For Top Over/Underperformers\n\n")
        f.write(md_table(expl))
        f.write("\n## Hypothesis Read\n\n")
        f.write("- Defensive/tactical style: look for positive goal-difference residual with negative goals-against residual. That means fewer conceded than the Poisson expectation and better match control than raw rating implied.\n")
        f.write("- Knockout/pressure effect: compare World Cup group-stage and knockout residuals. Large knockout values usually have tiny samples and may reflect penalties or path effects.\n")
        f.write("- Goalkeeper/penalty effect: `penalty_shootout_matches` counts available World Cup shootouts only; it is a contribution check, not a complete goalkeeper metric.\n")
        f.write("- Coach continuity and squad age/experience: historical coach/squad features exist for World Cups, but coverage is tournament-specific. Treat any association as descriptive.\n")
        f.write("- Rating lag: high recent `overperf_elo_10` is consistent with the model lagging an improving team.\n")
        f.write("- Confederation calibration: confederation fields are available mainly through World Cup team metadata, so broad regional claims should be tested separately before use.\n")
        f.write("- Bracket/path effect: deep runs can come from draw structure; per-match residuals reduce but do not eliminate this issue.\n")

    with open(REPORT_RECS, "w") as f:
        f.write("# Country Residual Model Recommendations\n\n")
        f.write("## Answers\n\n")
        f.write("1. Country fixed effects: **do not add by default**. They are high-variance and can encode reputation or historical noise.\n")
        f.write("2. Confederation calibration: **worth testing before country effects** because it has fewer degrees of freedom and targets plausible rating-calibration gaps.\n")
        f.write("3. Tournament pedigree/residual features: **test only as heavily shrunk, pre-match features** with strict tournament cutoffs.\n")
        f.write("4. Shrinkage-adjusted country residual feature: **experimental only** unless it improves WC2018 and WC2022 backtests without calibration damage.\n")
        f.write("5. Overfitting risk: **material**. World Cup samples are small, knockout samples are smaller, and several apparent effects are one-tournament artefacts.\n")
        f.write("6. Current WC2026 predictions: **do not change automatically** from this analysis. Use the report to create a manual-review shortlist only.\n\n")
        f.write("## Promotion Rule\n\n")
        f.write("Promote a residual feature only if it improves WC2018 and WC2022 log loss/Brier, is stable across tournaments, is shrunk strongly toward zero, and does not create suspicious high-confidence predictions.\n")

    with open(REPORT_LIFT, "w") as f:
        f.write("# Country Residual Feature Lift Report\n\n")
        f.write("Diagnostic prototype only. No production model or final prediction output was changed.\n\n")
        if lift.empty:
            f.write("_Prototype could not be evaluated._\n")
        else:
            f.write(md_table(lift))
            f.write("\n## Decision\n\n")
            f.write("Do not promote by default. This prototype is a lightweight screen; any serious promotion needs the same gates as the main model backtests plus calibration and high-confidence error checks.\n")


def main() -> None:
    df = add_wc_metadata(_prep_matrix())
    pred = predict_year_start(df)
    residuals = build_residual_rows(df, pred)
    table = aggregate_country(residuals)
    lift = feature_lift_prototype(residuals)

    RESIDUALS_OUT.parent.mkdir(parents=True, exist_ok=True)
    TABLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    residuals.to_parquet(RESIDUALS_OUT, index=False)
    table.to_csv(TABLE_OUT, index=False)
    write_reports(residuals, table, lift)

    print(f"Wrote {RESIDUALS_OUT} ({len(residuals)} rows)")
    print(f"Wrote {TABLE_OUT} ({len(table)} teams)")
    for p in [REPORT_DATA, REPORT_COUNTRY, REPORT_STABILITY, REPORT_EXPLAIN, REPORT_RECS, REPORT_LIFT]:
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
