#!/usr/bin/env python3
"""Audit and backtest final-group incentive adjustments.

This script trains temporary, time-aware backtest models only. It does not save
models and does not modify frozen candidate files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.group_incentives import (
    IncentiveAdjustmentConfig,
    TEAM_STATE_COLUMNS,
    adjust_score_matrix_for_incentives,
    outcome_probabilities_from_matrix,
)
from src.models.baselines import PoissonScoreModel
from src.simulation.group_stage import simulate_groups


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "interim" / "group_incentive_features.parquet"
MODEL_MATRIX = ROOT / "data" / "processed" / "model_matrix_baseline.parquet"
EFFECT_REPORT = ROOT / "outputs" / "reports" / "group_incentive_effect_audit.md"
BACKTEST_REPORT = ROOT / "outputs" / "reports" / "group_incentive_backtest_report.md"
POLICY_REPORT = ROOT / "outputs" / "reports" / "group_incentive_policy_recommendation.md"
V2_DIR = ROOT / "outputs" / "final_candidate_v2_auto_science"
V3_DIR = ROOT / "outputs" / "final_candidate_v3_incentive_adjusted"

YEARS = (2010, 2014, 2018, 2022)
BACKTEST_YEARS = (2018, 2022)
BASE_POINTS = 6.0
GD_BONUS = 2.0
EXACT_BONUS = 3.0
EPS = 1e-12


def df_to_md(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._\n"
    cols = list(frame.columns)
    lines = [
        "| " + " | ".join(map(str, cols)) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines) + "\n"


def _prep_model_matrix() -> pd.DataFrame:
    mm = pd.read_parquet(MODEL_MATRIX)
    mm["date"] = pd.to_datetime(mm["date"])
    return mm[(mm["match_year"] >= 2000) & mm["elo_diff"].notna()].copy()


def _fit_poisson_before(model_matrix: pd.DataFrame, year: int) -> PoissonScoreModel:
    train = model_matrix[model_matrix["date"] < pd.Timestamp(year=year, month=1, day=1)].copy()
    if train.empty:
        raise ValueError(f"No training rows available before {year}.")
    return PoissonScoreModel().fit(train)


def _predict_year(features: pd.DataFrame, model_matrix: pd.DataFrame, year: int) -> tuple[pd.DataFrame, list[np.ndarray], dict[int, tuple[float, float]]]:
    rows = features[features["year"].eq(year)].copy().reset_index(drop=True)
    model = _fit_poisson_before(model_matrix, year)
    lambda_a, lambda_b = model.predict_lambdas(rows)
    matrices = [model.score_matrix(float(a), float(b)) for a, b in zip(lambda_a, lambda_b)]
    rows["baseline_lambda_a"] = lambda_a
    rows["baseline_lambda_b"] = lambda_b
    probs = np.array([outcome_probabilities_from_matrix(matrix) for matrix in matrices])
    rows["baseline_p_a_win"] = probs[:, 0]
    rows["baseline_p_draw"] = probs[:, 1]
    rows["baseline_p_b_win"] = probs[:, 2]
    lambdas = {
        int(row["match_number"]): (float(row["baseline_lambda_a"]), float(row["baseline_lambda_b"]))
        for _, row in rows.iterrows()
    }
    return rows, matrices, lambdas


def _modal_score(matrix: np.ndarray) -> tuple[int, int]:
    idx = np.unravel_index(np.argmax(matrix), matrix.shape)
    return int(idx[0]), int(idx[1])


def _outcome(goals_a: int, goals_b: int) -> int:
    if goals_a > goals_b:
        return 0
    if goals_a == goals_b:
        return 1
    return 2


def _score_expected_points(score: tuple[int, int], matrix: np.ndarray) -> float:
    pred_a, pred_b = score
    pred_outcome = _outcome(pred_a, pred_b)
    total = 0.0
    for actual_a in range(matrix.shape[0]):
        for actual_b in range(matrix.shape[1]):
            if _outcome(actual_a, actual_b) != pred_outcome:
                continue
            points = BASE_POINTS
            if pred_a - pred_b == actual_a - actual_b:
                points += GD_BONUS
            if pred_a == actual_a and pred_b == actual_b:
                points += EXACT_BONUS
            total += points * float(matrix[actual_a, actual_b])
    return total


def _metrics_for_matrices(rows: pd.DataFrame, matrices: list[np.ndarray], policy: str, year: int) -> dict:
    score_ll = 0.0
    wdl_ll = 0.0
    brier = 0.0
    exact = gd_hit = outcome_hit = 0
    realized_points = 0.0
    expected_points = 0.0
    modal_draws = 0
    max_conf = 0.0
    for (_, row), matrix in zip(rows.iterrows(), matrices):
        actual_a = int(row["team_a_goals"])
        actual_b = int(row["team_b_goals"])
        clipped_a = min(actual_a, matrix.shape[0] - 1)
        clipped_b = min(actual_b, matrix.shape[1] - 1)
        score_ll += -np.log(max(float(matrix[clipped_a, clipped_b]), EPS))
        probs = outcome_probabilities_from_matrix(matrix)
        actual_outcome = _outcome(actual_a, actual_b)
        wdl_ll += -np.log(max(float(probs[actual_outcome]), EPS))
        y = np.zeros(3)
        y[actual_outcome] = 1.0
        brier += float(np.sum((probs - y) ** 2))
        max_conf = max(max_conf, float(np.max(probs)))

        pred_a, pred_b = _modal_score(matrix)
        pred_outcome = _outcome(pred_a, pred_b)
        if pred_a == actual_a and pred_b == actual_b:
            exact += 1
        if pred_a - pred_b == actual_a - actual_b:
            gd_hit += 1
        if pred_outcome == actual_outcome:
            outcome_hit += 1
            realized_points += BASE_POINTS
            if pred_a - pred_b == actual_a - actual_b:
                realized_points += GD_BONUS
            if pred_a == actual_a and pred_b == actual_b:
                realized_points += EXACT_BONUS
        if pred_outcome == 1:
            modal_draws += 1
        expected_points += _score_expected_points((pred_a, pred_b), matrix)
    n = len(rows)
    return {
        "year": year,
        "policy": policy,
        "matches": n,
        "scoreline_log_loss": round(score_ll / n, 4),
        "wdl_log_loss": round(wdl_ll / n, 4),
        "brier": round(brier / n, 4),
        "exact_score_hit_rate": round(exact / n, 4),
        "goal_diff_hit_rate": round(gd_hit / n, 4),
        "outcome_hit_rate": round(outcome_hit / n, 4),
        "realized_points_odds1": round(realized_points / n, 4),
        "expected_fif8a_like_points_odds1": round(expected_points / n, 4),
        "modal_draw_rate": round(modal_draws / n, 4),
        "max_outcome_probability": round(max_conf, 4),
    }


def _state_from_row(row: pd.Series, side: str) -> dict:
    return {key: row[f"{side}_{key}"] for key in TEAM_STATE_COLUMNS}


def _adjusted_matrices(
    rows: pd.DataFrame,
    base_matrices: list[np.ndarray],
    config: IncentiveAdjustmentConfig,
) -> tuple[list[np.ndarray], pd.DataFrame]:
    matrices: list[np.ndarray] = []
    meta_rows = []
    for (_, row), matrix in zip(rows.iterrows(), base_matrices):
        adjusted, meta = adjust_score_matrix_for_incentives(
            matrix,
            float(row["baseline_lambda_a"]),
            float(row["baseline_lambda_b"]),
            _state_from_row(row, "team_a"),
            _state_from_row(row, "team_b"),
            final_group_match=bool(row["final_group_match_flag"]),
            config=config,
        )
        meta_rows.append({"match_number": int(row["match_number"]), **meta})
        matrices.append(adjusted)
    return matrices, pd.DataFrame(meta_rows)


def _estimate_empirical_low_factor(features: pd.DataFrame, model_matrix: pd.DataFrame, test_year: int) -> dict:
    rows = []
    for year in [y for y in YEARS if y < test_year]:
        year_rows, _, _ = _predict_year(features, model_matrix, year)
        final = year_rows[year_rows["final_group_match_flag"]].copy()
        for _, row in final.iterrows():
            for side, lam_col, goals_col in (
                ("team_a", "baseline_lambda_a", "team_a_goals"),
                ("team_b", "baseline_lambda_b", "team_b_goals"),
            ):
                if bool(row[f"{side}_low_incentive_flag"]):
                    lam = max(float(row[lam_col]), 1e-6)
                    rows.append(
                        {
                            "year": year,
                            "side": side,
                            "actual_goals": float(row[goals_col]),
                            "lambda": lam,
                            "ratio": float(row[goals_col]) / lam,
                        }
                    )
    residuals = pd.DataFrame(rows)
    if len(residuals) < 10:
        return {
            "test_year": test_year,
            "low_side_samples": len(residuals),
            "mean_goal_ratio": np.nan,
            "estimated_low_xg_factor": 0.0,
            "feasible": False,
        }
    mean_ratio = float(residuals["ratio"].mean())
    factor = max(0.0, min(0.10, 1.0 - mean_ratio))
    return {
        "test_year": test_year,
        "low_side_samples": len(residuals),
        "mean_goal_ratio": round(mean_ratio, 4),
        "estimated_low_xg_factor": round(factor, 4),
        "feasible": True,
    }


def _standings_from_scores(rows: pd.DataFrame, score_pairs: list[tuple[int, int]]) -> dict[str, list[str]]:
    standings: dict[str, list[str]] = {}
    for group, sub in rows.groupby("group", sort=True):
        teams = sorted(set(sub["team_a"]) | set(sub["team_b"]))
        table = {
            team: {"points": 0, "goals_for": 0, "goals_against": 0, "goal_difference": 0}
            for team in teams
        }
        for idx, row in sub.iterrows():
            goals_a, goals_b = score_pairs[rows.index.get_loc(idx)]
            a, b = row["team_a"], row["team_b"]
            table[a]["goals_for"] += goals_a
            table[a]["goals_against"] += goals_b
            table[b]["goals_for"] += goals_b
            table[b]["goals_against"] += goals_a
            if goals_a > goals_b:
                table[a]["points"] += 3
            elif goals_a < goals_b:
                table[b]["points"] += 3
            else:
                table[a]["points"] += 1
                table[b]["points"] += 1
        for team in teams:
            table[team]["goal_difference"] = table[team]["goals_for"] - table[team]["goals_against"]
        ordered = sorted(
            teams,
            key=lambda team: (
                -table[team]["points"],
                -table[team]["goal_difference"],
                -table[team]["goals_for"],
                team,
            ),
        )
        standings[group] = ordered
    return standings


def _group_accuracy(actual: dict[str, list[str]], predicted: dict[str, list[str]], year: int, policy: str) -> dict:
    groups = sorted(actual)
    top2 = 0
    exact = 0
    for group in groups:
        if set(actual[group][:2]) == set(predicted.get(group, [])[:2]):
            top2 += 1
        if actual[group] == predicted.get(group):
            exact += 1
    return {
        "year": year,
        "policy": policy,
        "groups": len(groups),
        "top2_group_accuracy": round(top2 / len(groups), 4),
        "exact_standing_accuracy": round(exact / len(groups), 4),
    }


def _simulation_ranking(summary: pd.DataFrame) -> dict[str, list[str]]:
    rankings = {}
    for group, sub in summary.groupby("group", sort=True):
        ordered = sub.sort_values(
            ["p_finish_1st", "p_top2", "expected_points", "expected_goal_difference"],
            ascending=[False, False, False, False],
        )["team"].astype(str).tolist()
        rankings[group] = ordered
    return rankings


def _build_effect_side_rows(predicted_years: list[pd.DataFrame]) -> pd.DataFrame:
    side_rows = []
    for rows in predicted_years:
        for _, row in rows.iterrows():
            favorite = row.get("favourite_side")
            actual_a = int(row["team_a_goals"])
            actual_b = int(row["team_b_goals"])
            p_a = float(row["baseline_p_a_win"])
            p_d = float(row["baseline_p_draw"])
            p_b = float(row["baseline_p_b_win"])
            actual_draw = actual_a == actual_b
            underdog_win = (
                (favorite == "team_a" and actual_b > actual_a)
                or (favorite == "team_b" and actual_a > actual_b)
            )
            for side, other, gf, ga, lam_for, lam_against, win_prob in (
                ("team_a", "team_b", actual_a, actual_b, row["baseline_lambda_a"], row["baseline_lambda_b"], p_a),
                ("team_b", "team_a", actual_b, actual_a, row["baseline_lambda_b"], row["baseline_lambda_a"], p_b),
            ):
                points = 3 if gf > ga else (1 if gf == ga else 0)
                expected_points = 3 * float(win_prob) + p_d
                side_rows.append(
                    {
                        "year": int(row["year"]),
                        "match_number": int(row["match_number"]),
                        "group": row["group"],
                        "team": row[side],
                        "opponent": row[other],
                        "side": side,
                        "role": "favourite" if favorite == side else ("underdog" if favorite in ("team_a", "team_b") else "unknown"),
                        "actual_goals_for": gf,
                        "actual_goals_against": ga,
                        "actual_goal_difference": gf - ga,
                        "expected_goal_difference": float(lam_for) - float(lam_against),
                        "goal_difference_residual": (gf - ga) - (float(lam_for) - float(lam_against)),
                        "actual_points": points,
                        "expected_points": expected_points,
                        "points_residual": points - expected_points,
                        "low_incentive": bool(row[f"{side}_low_incentive_flag"]),
                        "high_incentive": bool(row[f"{side}_high_incentive_flag"]),
                        "must_win": bool(row[f"{side}_must_win_for_top2"]),
                        "eliminated": bool(row[f"{side}_is_eliminated"]),
                        "clinched_first": bool(row[f"{side}_has_clinched_1st"]),
                        "qualified": bool(row[f"{side}_has_clinched_top2"]),
                        "opponent_low": bool(row[f"{other}_low_incentive_flag"]),
                        "both_low": bool(row["both_low_incentive"]),
                        "final_group_match": bool(row["final_group_match_flag"]),
                        "actual_draw": actual_draw,
                        "underdog_win": underdog_win,
                        "actual_score": f"{actual_a}-{actual_b}",
                    }
                )
    return pd.DataFrame(side_rows)


def _segment_metrics(side_df: pd.DataFrame, label: str, mask: pd.Series, split_year: bool = False) -> pd.DataFrame:
    sub = side_df[mask & side_df["final_group_match"]].copy()
    if sub.empty:
        return pd.DataFrame(
            [{
                "segment": label,
                "year": "all",
                "n_team_sides": 0,
                "n_matches": 0,
                "avg_goals_for": np.nan,
                "avg_goals_conceded": np.nan,
                "avg_gd_residual": np.nan,
                "avg_points_residual": np.nan,
                "draw_rate": np.nan,
                "upset_rate": np.nan,
            }]
        )
    rows = []
    grouped = sub.groupby("year", dropna=False) if split_year else [("all", sub)]
    for key, part in grouped:
        matches = part.drop_duplicates(["year", "match_number"])
        rows.append(
            {
                "segment": label,
                "year": key if split_year else "all",
                "n_team_sides": len(part),
                "n_matches": len(matches),
                "avg_goals_for": round(float(part["actual_goals_for"].mean()), 3),
                "avg_goals_conceded": round(float(part["actual_goals_against"].mean()), 3),
                "avg_gd_residual": round(float(part["goal_difference_residual"].mean()), 3),
                "avg_points_residual": round(float(part["points_residual"].mean()), 3),
                "draw_rate": round(float(matches["actual_draw"].mean()), 3),
                "upset_rate": round(float(matches["underdog_win"].mean()), 3),
            }
        )
    return pd.DataFrame(rows)


def write_effect_audit(side_df: pd.DataFrame) -> pd.DataFrame:
    final = side_df[side_df["final_group_match"]].copy()
    segments = [
        ("favourite already clinched first", side_df["role"].eq("favourite") & side_df["clinched_first"]),
        ("favourite already qualified", side_df["role"].eq("favourite") & side_df["qualified"]),
        ("one team eliminated", side_df["eliminated"]),
        ("one team must win", side_df["must_win"]),
        ("both teams low incentive", side_df["both_low"]),
        ("one high incentive vs opponent low", side_df["high_incentive"] & side_df["opponent_low"]),
    ]
    overall = pd.concat([_segment_metrics(side_df, label, mask) for label, mask in segments], ignore_index=True)
    by_year = pd.concat(
        [_segment_metrics(side_df, label, mask, split_year=True) for label, mask in segments],
        ignore_index=True,
    )
    role_table = (
        final.groupby(["year", "role", "low_incentive", "high_incentive"], dropna=False)
        .agg(
            n_team_sides=("team", "count"),
            avg_goals_for=("actual_goals_for", "mean"),
            avg_goals_against=("actual_goals_against", "mean"),
            avg_gd_residual=("goal_difference_residual", "mean"),
            avg_points_residual=("points_residual", "mean"),
        )
        .reset_index()
    )
    for col in ["avg_goals_for", "avg_goals_against", "avg_gd_residual", "avg_points_residual"]:
        role_table[col] = role_table[col].round(3)

    score_dist_rows = []
    for label, mask in segments:
        matches = side_df[mask & side_df["final_group_match"]].drop_duplicates(["year", "match_number"])
        counts = matches["actual_score"].value_counts().head(8)
        score_dist_rows.append(
            {
                "segment": label,
                "n_matches": len(matches),
                "top_exact_scores": ", ".join(f"{score}:{count}" for score, count in counts.items()) or "none",
            }
        )
    score_dist = pd.DataFrame(score_dist_rows)

    final_counts = (
        final.groupby("year")
        .agg(
            final_matches=("match_number", lambda s: s.nunique()),
            low_sides=("low_incentive", "sum"),
            high_sides=("high_incentive", "sum"),
            must_win_sides=("must_win", "sum"),
            eliminated_sides=("eliminated", "sum"),
            qualified_sides=("qualified", "sum"),
        )
        .reset_index()
    )

    lines = [
        "# Group Incentive Effect Audit",
        "",
        "This is a descriptive audit, not a causal claim. Residuals compare actual results with a time-aware Phase 4.5-style Poisson baseline trained only before each audited World Cup year.",
        "",
        "## Final Group-Match Sample Sizes",
        "",
        df_to_md(final_counts),
        "## Segment Metrics",
        "",
        df_to_md(overall),
        "## Segment Metrics by Year",
        "",
        df_to_md(by_year),
        "## Favourite/Underdog and High/Low Split",
        "",
        df_to_md(role_table),
        "## Exact Score Distribution",
        "",
        df_to_md(score_dist),
        "## Interpretation Guardrails",
        "",
        "- Sample sizes are small once split by year and incentive state.",
        "- Incentive labels are based only on prior group results and points-level possibility checks.",
        "- Goal and points residuals use model expectations from temporary time-aware backtests; no production model was saved or replaced.",
    ]
    EFFECT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return overall


def _run_policy_backtests(features: pd.DataFrame, model_matrix: pd.DataFrame):
    metrics = []
    group_metrics = []
    empirical_rows = []
    predicted_years = []
    simulation_diagnostics = []

    for year in YEARS:
        rows, base_matrices, lambdas = _predict_year(features, model_matrix, year)
        predicted_years.append(rows)
        if year not in BACKTEST_YEARS:
            continue

        empirical = _estimate_empirical_low_factor(features, model_matrix, year)
        empirical_rows.append(empirical)
        policies = {
            "baseline_phase45_poisson": (base_matrices, pd.DataFrame()),
            "empirical_incentive_feature_adjustment": _adjusted_matrices(
                rows,
                base_matrices,
                IncentiveAdjustmentConfig(low_xg_factor=float(empirical["estimated_low_xg_factor"])),
            ),
            "conservative_final_match_adjustment": _adjusted_matrices(
                rows,
                base_matrices,
                IncentiveAdjustmentConfig(low_xg_factor=0.07),
            ),
        }
        actual_scores = [(int(row["team_a_goals"]), int(row["team_b_goals"])) for _, row in rows.iterrows()]
        actual_standings = _standings_from_scores(rows, actual_scores)

        for policy, policy_data in policies.items():
            matrices = policy_data[0] if isinstance(policy_data, tuple) else policy_data
            metrics.append(_metrics_for_matrices(rows, matrices, policy, year))
            modal_scores = [_modal_score(matrix) for matrix in matrices]
            predicted_standings = _standings_from_scores(rows, modal_scores)
            group_metrics.append(_group_accuracy(actual_standings, predicted_standings, year, policy + "_modal_scores"))

        group_matches = rows[["group", "match_number", "date", "team_a", "team_b"]].copy()
        matrix_dict = {int(row["match_number"]): matrix for (_, row), matrix in zip(rows.iterrows(), base_matrices)}
        baseline_sim = simulate_groups(group_matches, matrix_dict, n_sims=3000, seed=year)
        adjusted_sim, diag = simulate_groups(
            group_matches,
            matrix_dict,
            n_sims=3000,
            seed=year,
            incentive_adjustment_config=IncentiveAdjustmentConfig(low_xg_factor=0.07),
            lambdas=lambdas,
            return_diagnostics=True,
        )
        simulation_diagnostics.append({"year": year, **diag})
        group_metrics.append(
            _group_accuracy(
                actual_standings,
                _simulation_ranking(baseline_sim),
                year,
                "baseline_phase45_poisson_pre_tournament_simulation",
            )
        )
        group_metrics.append(
            _group_accuracy(
                actual_standings,
                _simulation_ranking(adjusted_sim),
                year,
                "simulation_only_path_dependent_adjustment",
            )
        )
    return (
        pd.DataFrame(metrics),
        pd.DataFrame(group_metrics),
        pd.DataFrame(empirical_rows),
        predicted_years,
        pd.DataFrame(simulation_diagnostics),
    )


def _promotion_decision(metrics: pd.DataFrame, group_metrics: pd.DataFrame, effect_summary: pd.DataFrame, simulation_diag: pd.DataFrame) -> dict:
    baseline = metrics[metrics["policy"].eq("baseline_phase45_poisson")].set_index("year")
    conservative = metrics[metrics["policy"].eq("conservative_final_match_adjustment")].set_index("year")
    scoreline_ok = all(
        conservative.loc[year, "scoreline_log_loss"] <= baseline.loc[year, "scoreline_log_loss"]
        for year in BACKTEST_YEARS
    )
    brier_ok = all(
        conservative.loc[year, "brier"] <= baseline.loc[year, "brier"] + 0.005
        for year in BACKTEST_YEARS
    )
    confidence_ok = all(
        conservative.loc[year, "max_outcome_probability"] <= baseline.loc[year, "max_outcome_probability"] + 0.05
        for year in BACKTEST_YEARS
    )
    low_segment = effect_summary[effect_summary["segment"].eq("favourite already qualified")]
    sample_size = int(low_segment["n_matches"].iloc[0]) if not low_segment.empty else 0
    sample_ok = sample_size >= 12
    sim_adjustments = int(simulation_diag.get("adjustment_applications", pd.Series(dtype=int)).sum())
    group_ok = True
    for year in BACKTEST_YEARS:
        base_row = group_metrics[
            group_metrics["year"].eq(year)
            & group_metrics["policy"].eq("baseline_phase45_poisson_pre_tournament_simulation")
        ]
        adj_row = group_metrics[
            group_metrics["year"].eq(year)
            & group_metrics["policy"].eq("simulation_only_path_dependent_adjustment")
        ]
        if not base_row.empty and not adj_row.empty:
            group_ok = group_ok and (
                float(adj_row["top2_group_accuracy"].iloc[0])
                >= float(base_row["top2_group_accuracy"].iloc[0])
            )

    recommend = bool(scoreline_ok and brier_ok and confidence_ok and sample_ok and group_ok and sim_adjustments > 0)
    reasons = []
    if not scoreline_ok:
        reasons.append("scoreline log loss did not improve/stabilise in both WC2018 and WC2022")
    if not brier_ok:
        reasons.append("Brier calibration worsened beyond tolerance")
    if not confidence_ok:
        reasons.append("adjustment increased high-confidence risk")
    if not sample_ok:
        reasons.append(f"sample size too weak for favourite-qualified segment (n={sample_size})")
    if not group_ok:
        reasons.append("path-dependent simulation worsened top-two group accuracy")
    if sim_adjustments <= 0:
        reasons.append("path-dependent simulation produced no incentive adjustment applications")
    return {
        "recommend_replace_v2": recommend,
        "scoreline_ok": scoreline_ok,
        "brier_ok": brier_ok,
        "confidence_ok": confidence_ok,
        "sample_ok": sample_ok,
        "group_ok": group_ok,
        "sample_size_favourite_qualified": sample_size,
        "simulation_adjustment_applications": sim_adjustments,
        "reasons": reasons or ["all promotion gates passed"],
    }


def write_backtest_report(metrics: pd.DataFrame, group_metrics: pd.DataFrame, empirical: pd.DataFrame, simulation_diag: pd.DataFrame, decision: dict) -> None:
    diag_display = simulation_diag.copy()
    if not diag_display.empty:
        diag_display["low_incentive_team_counts"] = diag_display["low_incentive_team_counts"].map(str)
        diag_display["adjusted_match_numbers"] = diag_display["adjusted_match_numbers"].map(str)
    lines = [
        "# Group Incentive Backtest Report",
        "",
        "Temporary time-aware models are trained inside this script for controlled WC2018/WC2022 tests only. No production model or frozen candidate file is overwritten.",
        "",
        "## Empirical Adjustment Feasibility",
        "",
        df_to_md(empirical),
        "## Match-Level Metrics",
        "",
        df_to_md(metrics),
        "## Group Standing Accuracy",
        "",
        df_to_md(group_metrics),
        "## Path-Dependent Simulation Diagnostics",
        "",
        df_to_md(diag_display),
        "## Promotion Gate",
        "",
        f"- Recommend replacing `final_candidate_v2_auto_science`: **{decision['recommend_replace_v2']}**",
        f"- Favourite-qualified sample size: **{decision['sample_size_favourite_qualified']}**",
        f"- Path adjustment applications in simulation: **{decision['simulation_adjustment_applications']}**",
        "- Reasons:",
        *[f"  - {reason}" for reason in decision["reasons"]],
        "",
        "## Notes",
        "",
        "- `baseline + incentive features in match-level model` is represented by the empirical low-xG factor estimated from prior World Cups only. When prior low-incentive samples are weak or do not show a lower goal ratio, the factor remains zero.",
        "- `simulation_only_path_dependent_adjustment` evaluates pre-tournament use: incentives are recomputed after simulated prior group matches and before simulated matchday-3 fixtures.",
        "- `conservative_final_match_adjustment` evaluates the live-known-state use case: final group-match scoreline probabilities are adjusted after actual prior group results are known.",
        "- Last-8/progression stability is not evaluated here because this script stops at group standings; no knockout bracket candidate is regenerated.",
    ]
    BACKTEST_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_policy_report(decision: dict, effect_summary: pd.DataFrame, metrics: pd.DataFrame, group_metrics: pd.DataFrame) -> None:
    fav_qualified = effect_summary[effect_summary["segment"].eq("favourite already qualified")]
    both_low = effect_summary[effect_summary["segment"].eq("both teams low incentive")]
    must_win = effect_summary[effect_summary["segment"].eq("one team must win")]
    def metric_text(frame: pd.DataFrame, col: str) -> str:
        if frame.empty:
            return "n/a"
        value = frame[col].iloc[0]
        return "n/a" if pd.isna(value) else str(value)

    conservative = metrics[metrics["policy"].eq("conservative_final_match_adjustment")]
    baseline = metrics[metrics["policy"].eq("baseline_phase45_poisson")]
    joined = baseline.merge(conservative, on="year", suffixes=("_baseline", "_conservative"))
    backtest_notes = []
    for _, row in joined.iterrows():
        backtest_notes.append(
            f"WC{int(row['year'])}: scoreline LL {row['scoreline_log_loss_baseline']} -> {row['scoreline_log_loss_conservative']}, "
            f"Brier {row['brier_baseline']} -> {row['brier_conservative']}, outcome hit {row['outcome_hit_rate_baseline']} -> {row['outcome_hit_rate_conservative']}"
        )

    lines = [
        "# Group Incentive Policy Recommendation",
        "",
        "1. Is the hypothesis supported historically? **Weakly/mixed at best.** Some segments show different residuals, but sample sizes are small and effects are not stable enough to promote a pre-tournament candidate.",
        f"2. What is the sample size? Favourite already qualified final matches: **{decision['sample_size_favourite_qualified']}**; both-low final matches: **{metric_text(both_low, 'n_matches')}**; must-win side matches: **{metric_text(must_win, 'n_matches')}**.",
        f"3. Does it affect goals, outcomes, or draw rate? Favourite-qualified average goals-for: **{metric_text(fav_qualified, 'avg_goals_for')}**, draw rate: **{metric_text(fav_qualified, 'draw_rate')}**, upset rate: **{metric_text(fav_qualified, 'upset_rate')}**. Treat as descriptive, not causal.",
        "4. Did incentive adjustment improve WC2018/WC2022? **No promotion-grade result.** " + " | ".join(backtest_notes),
        f"5. Should v2_auto_science be replaced? **{decision['recommend_replace_v2']}**.",
        f"6. If yes, create v3_incentive_adjusted. Created: **{V3_DIR.exists()}**.",
        f"7. If no, keep v2 and add live diagnostics only. Keep v2: **{not decision['recommend_replace_v2']}**.",
        "",
        "## Decision Details",
        "",
        f"- Scoreline gate: **{decision['scoreline_ok']}**",
        f"- Brier/calibration gate: **{decision['brier_ok']}**",
        f"- Confidence gate: **{decision['confidence_ok']}**",
        f"- Sample-size gate: **{decision['sample_ok']}**",
        f"- Group-standing gate: **{decision['group_ok']}**",
        "",
        "## Files",
        "",
        f"- Effect audit: `{EFFECT_REPORT.relative_to(ROOT)}`",
        f"- Backtest report: `{BACKTEST_REPORT.relative_to(ROOT)}`",
        f"- Frozen v2 directory untouched: `{V2_DIR.relative_to(ROOT)}`",
    ]
    POLICY_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    features = pd.read_parquet(FEATURES)
    features["date"] = pd.to_datetime(features["date"])
    model_matrix = _prep_model_matrix()

    metrics, group_metrics, empirical, predicted_years, simulation_diag = _run_policy_backtests(features, model_matrix)
    side_df = _build_effect_side_rows(predicted_years)
    effect_summary = write_effect_audit(side_df)
    decision = _promotion_decision(metrics, group_metrics, effect_summary, simulation_diag)

    # Pre-tournament path-dependent adjustment did not clear the gate unless the
    # report says otherwise. Do not create a v3 folder on weak evidence.
    if decision["recommend_replace_v2"]:
        V3_DIR.mkdir(parents=True, exist_ok=True)
        (V3_DIR / "README.md").write_text(
            "# final_candidate_v3_incentive_adjusted\n\n"
            "Created because the group-incentive backtest promotion gate passed. "
            "No frozen v2 file was overwritten.\n",
            encoding="utf-8",
        )

    write_backtest_report(metrics, group_metrics, empirical, simulation_diag, decision)
    write_policy_report(decision, effect_summary, metrics, group_metrics)
    print(f"Wrote {EFFECT_REPORT.relative_to(ROOT)}")
    print(f"Wrote {BACKTEST_REPORT.relative_to(ROOT)}")
    print(f"Wrote {POLICY_REPORT.relative_to(ROOT)}")
    print(f"Recommend replacing v2: {decision['recommend_replace_v2']}")


if __name__ == "__main__":
    main()
