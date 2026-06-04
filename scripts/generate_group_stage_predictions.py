#!/usr/bin/env python3
"""Generate FIF8A group-stage predictions (Phase 4.5 audit refresh).

For each of the 72 template matches: model W/D/L probs + scoreline distribution
(Poisson), expected FIF8A points per candidate prediction (using template odds),
and safe vs EV recommendations.

Team features come from each team's latest pre-cutoff snapshot in the model
matrix. WC group matches are treated as NEUTRAL (no home advantage transferred;
see RULES_AND_SCORING.md §7 and Hypothesis 1 in the report).
"""
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.ingest.rules_and_scoring import load_scoring_rules
from src.models.baselines import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    BINARY_FEATURES,
    make_logit_pipeline,
    make_hgb_pipeline,
    PoissonScoreModel,
    proba_in_class_order,
)
from src.features.template_features import build_team_snapshots, feature_row
from src.evaluation.backtest import time_split
from src.evaluation.expected_points import (
    outcome_probs_from_matrix, most_probable_score, ev_max_score,
    expected_points_for_score, expected_points_for_outcome,
)
from src.evaluation.group_stage_predictions import (
    OUTCOME_KEYS,
    add_score_columns,
    most_probable_score_for_outcome,
    named_outcome_from_key,
    score_to_string,
)
from src.evaluation.metrics import all_wdl_metrics

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
MATRIX = ROOT / "data" / "processed" / "model_matrix_baseline.parquet"
TEMPLATE = ROOT / "data" / "reference" / "fif8a_group_stage_template.csv"
MODELS = ROOT / "outputs" / "models" / "final_models.pkl"
OUT_CSV = ROOT / "outputs" / "predictions" / "fif8a_group_stage_predictions.csv"
OUT_CSV_ENSEMBLE = ROOT / "outputs" / "predictions" / "fif8a_group_stage_predictions_ensemble.csv"
REPORT = ROOT / "outputs" / "reports" / "fif8a_group_stage_prediction_report.md"
ENSEMBLE_REPORT = ROOT / "outputs" / "reports" / "ensemble_sanity_report.md"

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
TEMPLATE_ALIASES = {
    "Korea Republic": "South Korea",
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
    "Côte d'Ivoire": "Ivory Coast",
    "IR Iran": "Iran",
    "Cabo Verde": "Cape Verde",
    "Congo DR": "DR Congo",
    "USA": "United States",
}


def resolve_team(name, snaps):
    team = TEMPLATE_ALIASES.get(name, name)
    return team if team in snaps.index else None


def build_match_context(poisson, snaps, template_row):
    team_a = template_row["team_a"]
    team_b = template_row["team_b"]
    resolved_a = resolve_team(team_a, snaps)
    resolved_b = resolve_team(team_b, snaps)
    if resolved_a is None or resolved_b is None:
        return None

    X = feature_row(snaps.loc[resolved_a], snaps.loc[resolved_b], template_row["date"])
    lam_a, lam_b = poisson.predict_lambdas(X)
    M = poisson.score_matrix(float(lam_a[0]), float(lam_b[0]))
    return {
        "team_a": team_a,
        "team_b": team_b,
        "features": X,
        "matrix": M,
    }


def build_prediction_row(template_row, model_probs, score_matrix, rules, notes_prefix=None):
    team_a = template_row["team_a"]
    team_b = template_row["team_b"]
    odds = {"a_win": template_row["rate_a"], "draw": template_row["rate_draw"], "b_win": template_row["rate_b"]}
    inv = np.array([1 / template_row["rate_a"], 1 / template_row["rate_draw"], 1 / template_row["rate_b"]])
    template_probs = inv / inv.sum()

    safe_outcome_key = OUTCOME_KEYS[int(np.argmax(model_probs))]
    safe_score = most_probable_score_for_outcome(score_matrix, safe_outcome_key)
    raw_mode_score = most_probable_score(score_matrix)
    ev_score, ev_points = ev_max_score(score_matrix, odds, rules)
    ev_outcome_values = {
        outcome_key: expected_points_for_outcome(outcome_key, score_matrix, odds, rules)[0]
        for outcome_key in OUTCOME_KEYS
    }
    ev_outcome_key = max(ev_outcome_values, key=ev_outcome_values.get)
    safe_points = expected_points_for_score(safe_score[0], safe_score[1], score_matrix, odds, rules)

    edge = model_probs - template_probs
    contrarian = (ev_outcome_key != safe_outcome_key) and (odds[ev_outcome_key] > odds[safe_outcome_key])
    notes = []
    if notes_prefix:
        notes.extend(notes_prefix)
    if contrarian:
        notes.append("CONTRARIAN: EV backs higher-odds outcome vs most-probable")
    if score_matrix.max() < 0.12:
        notes.append("HIGH_VARIANCE: flat scoreline distribution")
    if abs(edge).max() > 0.10:
        notes.append("VALUE_EDGE>0.10 vs template")
    if safe_score != raw_mode_score:
        notes.append("SAFE_SCORE_ALIGNED_TO_MODEL_OUTCOME")

    return {
        "match_number": int(template_row["match_number"]),
        "group": template_row["group"],
        "date": template_row["date"],
        "team_a": team_a,
        "team_b": team_b,
        "rate_a": template_row["rate_a"],
        "rate_draw": template_row["rate_draw"],
        "rate_b": template_row["rate_b"],
        "model_p_a_win": round(float(model_probs[0]), 4),
        "model_p_draw": round(float(model_probs[1]), 4),
        "model_p_b_win": round(float(model_probs[2]), 4),
        "template_p_a_win": round(float(template_probs[0]), 4),
        "template_p_draw": round(float(template_probs[1]), 4),
        "template_p_b_win": round(float(template_probs[2]), 4),
        "value_edge_a": round(float(edge[0]), 4),
        "value_edge_draw": round(float(edge[1]), 4),
        "value_edge_b": round(float(edge[2]), 4),
        "most_probable_score": score_to_string(raw_mode_score),
        "ev_max_score": score_to_string(ev_score),
        "most_probable_outcome": named_outcome_from_key(safe_outcome_key, team_a, team_b),
        "ev_max_outcome": named_outcome_from_key(ev_outcome_key, team_a, team_b),
        "recommended_score_safe": score_to_string(safe_score),
        "recommended_score_ev": score_to_string(ev_score),
        "expected_points_safe": round(float(safe_points), 3),
        "expected_points_ev": round(float(ev_points), 3),
        "notes": "; ".join(notes),
    }


def _prep_backtest_matrix(mm):
    mm = mm.copy()
    mm["date"] = pd.to_datetime(mm["date"])
    return mm[(mm["match_year"] >= 2000) & mm["elo_diff"].notna()].copy()


def _fit_backtest_models(train):
    return {
        "logit": make_logit_pipeline().fit(train[ALL_FEATURES], train["result_label"]),
        "poisson": PoissonScoreModel().fit(train),
        "hgb": make_hgb_pipeline().fit(train[ALL_FEATURES], train["result_label"]),
    }


def _wdl_probabilities(models, test):
    return {
        "logit": proba_in_class_order(models["logit"], test[ALL_FEATURES]),
        "poisson": models["poisson"].predict_proba(test),
        "hgb": proba_in_class_order(models["hgb"], test[ALL_FEATURES]),
    }


def write_ensemble_report(mm):
    eval_df = _prep_backtest_matrix(mm)
    backtests = [
        ("train<=2014 -> WC2018", "2014-12-31", (eval_df["tournament"] == "FIFA World Cup") & (eval_df["match_year"] == 2018)),
        ("train<=2018 -> WC2022", "2018-12-31", (eval_df["tournament"] == "FIFA World Cup") & (eval_df["match_year"] == 2022)),
    ]

    rows = []
    improved = True
    for label, train_end, mask in backtests:
        train, test = time_split(eval_df, train_end, mask)
        models = _fit_backtest_models(train)
        probs = _wdl_probabilities(models, test)
        ensemble = (probs["logit"] + probs["poisson"] + probs["hgb"]) / 3.0

        poisson_metrics = all_wdl_metrics(test["result_label"].values, probs["poisson"])
        ensemble_metrics = all_wdl_metrics(test["result_label"].values, ensemble)
        improved = improved and ensemble_metrics["log_loss"] <= poisson_metrics["log_loss"] and ensemble_metrics["brier"] <= poisson_metrics["brier"]
        rows.append({
            "label": label,
            "poisson_log_loss": poisson_metrics["log_loss"],
            "poisson_brier": poisson_metrics["brier"],
            "ensemble_log_loss": ensemble_metrics["log_loss"],
            "ensemble_brier": ensemble_metrics["brier"],
            "ensemble_accuracy": ensemble_metrics["accuracy"],
        })

    ENSEMBLE_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(ENSEMBLE_REPORT, "w") as f:
        f.write("# Ensemble Sanity Report\n\n")
        f.write("- Candidate ensemble: equal-weight average of logit, HGB, and Poisson W/D/L probabilities.\n")
        f.write("- Scoreline engine remains the Poisson matrix; only W/D/L probabilities are blended.\n")
        f.write("- Selection rule: implement only if the ensemble does not worsen Poisson log loss or Brier on WC2018 or WC2022.\n\n")
        f.write("| Backtest | Poisson log loss | Ensemble log loss | Poisson Brier | Ensemble Brier | Ensemble accuracy |\n")
        f.write("|---|--:|--:|--:|--:|--:|\n")
        for row in rows:
            f.write(
                f"| {row['label']} | {row['poisson_log_loss']} | {row['ensemble_log_loss']} | "
                f"{row['poisson_brier']} | {row['ensemble_brier']} | {row['ensemble_accuracy']} |\n"
            )
        if improved:
            f.write("\nConclusion: the equal-weight ensemble slightly improves or matches Poisson on both held-out World Cups, so an optional ensemble predictions file is written.\n")
        else:
            f.write("\nConclusion: the ensemble does not clear the no-worse-on-both-cups rule, so Poisson remains the only prediction output.\n")
    logger.info(f"✓ Wrote {ENSEMBLE_REPORT}")
    return improved


def main():
    rules = load_scoring_rules()
    mm = pd.read_parquet(MATRIX)
    mm["date"] = pd.to_datetime(mm["date"])
    snaps = build_team_snapshots(mm)
    with open(MODELS, "rb") as f:
        model_bundle = pickle.load(f)
    poisson = model_bundle["poisson"]
    logit = model_bundle["logit"]
    hgb = model_bundle["hgb"]
    tmpl = pd.read_csv(TEMPLATE)

    poisson_rows = []
    ensemble_rows = []
    unresolved = set()
    for _, t in tmpl.iterrows():
        context = build_match_context(poisson, snaps, t)
        if context is None:
            unresolved.add(t["team_a"] if resolve_team(t["team_a"], snaps) is None else t["team_b"])
            continue
        X = context["features"]
        M = context["matrix"]
        poisson_probs = outcome_probs_from_matrix(M)
        logit_probs = proba_in_class_order(logit, X[ALL_FEATURES])[0]
        hgb_probs = proba_in_class_order(hgb, X[ALL_FEATURES])[0]
        ensemble_probs = (poisson_probs + logit_probs + hgb_probs) / 3.0

        poisson_rows.append(build_prediction_row(t, poisson_probs, M, rules))
        ensemble_rows.append(build_prediction_row(t, ensemble_probs, M, rules, notes_prefix=["WDL_SOURCE=equal_weight_ensemble"]))

    pred = add_score_columns(pd.DataFrame(poisson_rows).sort_values("match_number"))
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(OUT_CSV, index=False)
    logger.info(f"✓ Wrote {OUT_CSV} ({len(pred)} rows; unresolved teams: {sorted(unresolved) or 'none'})")

    _report(pred, unresolved)
    if write_ensemble_report(mm):
        ensemble_pred = add_score_columns(pd.DataFrame(ensemble_rows).sort_values("match_number"))
        ensemble_pred.to_csv(OUT_CSV_ENSEMBLE, index=False)
        logger.info(f"✓ Wrote {OUT_CSV_ENSEMBLE} ({len(ensemble_pred)} rows)")


def _report(pred, unresolved):
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    edges = pred.assign(max_edge=pred[["value_edge_a", "value_edge_draw", "value_edge_b"]].abs().max(axis=1))
    top_edges = edges.sort_values("max_edge", ascending=False).head(10)
    contrarian = pred[pred["notes"].str.contains("CONTRARIAN")]
    safe_conf = pred.assign(conf=pred[["model_p_a_win", "model_p_draw", "model_p_b_win"]].max(axis=1)) \
                    .sort_values("conf", ascending=False).head(10)
    with open(REPORT, "w") as f:
        f.write("# FIF8A Group-Stage Prediction Report\n\n")
        f.write(f"- Matches predicted: **{len(pred)}** / 72\n")
        f.write(f"- Unresolved teams: {sorted(unresolved) or 'none'}\n")
        f.write("- Scoreline engine: **Poisson** goal model (Model 2); WC matches treated as "
                "**neutral** (no home advantage transferred — RULES §7 / Hypothesis 1).\n")
        f.write("- Odds in EV = template `rate_*` (FIFA-ranking Poisson odds), NOT bookmaker odds.\n\n")
        f.write("## Recommendation logic\n\n")
        f.write("- `most_probable_score` = raw Poisson modal scoreline, regardless of W/D/L aggregation.\n")
        f.write("- `recommended_score_safe` = most probable Poisson scoreline consistent with the highest model W/D/L probability.\n")
        f.write("- `recommended_score_ev` = expected-FIF8A-points-maximising scoreline.\n")
        f.write("- `recommended_score_*_display` keeps Team A / Team B order explicit as `Team A vs Team B: a-b`.\n")
        f.write("- Contrarian flag = EV backs a higher-odds outcome than the most probable one "
                "(do not chase odds mechanically; these are flagged, not defaulted).\n\n")
        f.write("## Top 10 model-vs-template value edges\n\n")
        f.write("| # | match | model vs template (edge) |\n|---|---|---|\n")
        for _, r in top_edges.iterrows():
            f.write(f"| {r['match_number']} | {r['team_a']} v {r['team_b']} | "
                    f"a {r['value_edge_a']:+.2f}, draw {r['value_edge_draw']:+.2f}, b {r['value_edge_b']:+.2f} |\n")
        f.write("\n## Highest-confidence safe predictions\n\n")
        f.write("| # | match | pick | p | safe score |\n|---|---|---|--:|---|\n")
        for _, r in safe_conf.iterrows():
            conf = max(r["model_p_a_win"], r["model_p_draw"], r["model_p_b_win"])
            f.write(f"| {r['match_number']} | {r['team_a']} v {r['team_b']} | {r['most_probable_outcome']} | "
                    f"{conf:.2f} | {r['recommended_score_safe']} |\n")
        f.write(f"\n## High-variance / contrarian picks ({len(contrarian)})\n\n")
        if len(contrarian):
            f.write("| # | match | safe outcome | EV outcome | notes |\n|---|---|---|---|---|\n")
            for _, r in contrarian.iterrows():
                f.write(f"| {r['match_number']} | {r['team_a']} v {r['team_b']} | "
                        f"{r['most_probable_outcome']} | {r['ev_max_outcome']} | {r['notes']} |\n")
        else:
            f.write("None — EV recommendation never backed a higher-odds outcome over the most "
                    "probable one (expected behaviour under near-fair template odds).\n")
        f.write("\n⚠️ No model has been re-trained here; predictions use the saved Poisson model. "
                "Refresh after final friendlies (see model_selection_summary.md).\n")
    logger.info(f"✓ Wrote {REPORT}")


if __name__ == "__main__":
    main()
