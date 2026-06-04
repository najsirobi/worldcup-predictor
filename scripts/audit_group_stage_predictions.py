#!/usr/bin/env python3
"""Audit and sanity-check the Phase 4 group-stage prediction outputs."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from src.ingest.rules_and_scoring import load_scoring_rules
from src.features.template_features import build_team_snapshots, feature_row, resolve
from src.evaluation.expected_points import outcome_probs_from_matrix, expected_points_for_outcome
from src.evaluation.group_stage_predictions import (
    OUTCOME_KEYS,
    add_score_columns,
    named_outcome_from_key,
    outcome_key_from_score,
    parse_score,
    probability_sum_error,
    score_display,
)

ROOT = Path(__file__).parent.parent
MATRIX = ROOT / "data" / "processed" / "model_matrix_baseline.parquet"
TEMPLATE = ROOT / "data" / "reference" / "fif8a_group_stage_template.csv"
PREDICTIONS = ROOT / "outputs" / "predictions" / "fif8a_group_stage_predictions.csv"
SIMULATION = ROOT / "outputs" / "predictions" / "group_stage_simulation_summary.csv"
MODELS = ROOT / "outputs" / "models" / "final_models.pkl"
AUDIT_REPORT = ROOT / "outputs" / "reports" / "prediction_sanity_audit.md"
FLAGS_REPORT = ROOT / "outputs" / "reports" / "top_prediction_flags.md"

SPECIFIC_MATCHES = [8, 26, 18, 52, 37]


def _load_inputs():
    rules = load_scoring_rules()
    template = pd.read_csv(TEMPLATE)
    predictions = add_score_columns(pd.read_csv(PREDICTIONS))
    simulation = pd.read_csv(SIMULATION)
    matrix = pd.read_parquet(MATRIX)
    matrix["date"] = pd.to_datetime(matrix["date"])
    with open(MODELS, "rb") as handle:
        models = pickle.load(handle)
    snapshots = build_team_snapshots(matrix)
    return rules, template, predictions, simulation, models, snapshots


def _build_match_matrix(poisson_model, snapshots, team_a, team_b, match_date):
    resolved_a = resolve(team_a, snapshots)
    resolved_b = resolve(team_b, snapshots)
    if resolved_a is None or resolved_b is None:
        raise ValueError(f"Unresolved team in audit: {team_a} vs {team_b}")
    X = feature_row(snapshots.loc[resolved_a], snapshots.loc[resolved_b], match_date)
    lam_a, lam_b = poisson_model.predict_lambdas(X)
    M = poisson_model.score_matrix(float(lam_a[0]), float(lam_b[0]))
    return snapshots.loc[resolved_a], snapshots.loc[resolved_b], M


def _validate(template, predictions, simulation, rules, poisson_model, snapshots):
    issues = []

    if len(template) != 72:
        issues.append(f"Template row count is {len(template)}, expected 72.")
    if len(predictions) != len(template):
        issues.append(f"Predictions row count {len(predictions)} does not match template {len(template)}.")
    if len(simulation) != 48:
        issues.append(f"Simulation summary row count is {len(simulation)}, expected 48 teams.")

    merged = predictions.merge(
        template[["match_number", "team_a", "team_b", "rate_a", "rate_draw", "rate_b", "group", "date"]],
        on="match_number",
        suffixes=("_pred", "_template"),
    )
    orientation_ok = (
        merged["team_a_pred"].equals(merged["team_a_template"])
        and merged["team_b_pred"].equals(merged["team_b_template"])
    )
    if not orientation_ok:
        issues.append("Prediction rows do not preserve Team A / Team B template orientation.")

    for column in ["recommended_score_safe", "recommended_score_ev", "most_probable_score", "ev_max_score"]:
        try:
            predictions[column].map(parse_score)
        except Exception as exc:
            issues.append(f"Could not parse `{column}` as `team_a_goals-team_b_goals`: {exc}")
            break

    safe_display_expected = [
        score_display(team_a, team_b, parse_score(score))
        for team_a, team_b, score in zip(
            predictions["team_a"], predictions["team_b"], predictions["recommended_score_safe"]
        )
    ]
    if safe_display_expected != predictions["recommended_score_safe_display"].tolist():
        issues.append("Safe display strings are not Team A / Team B oriented.")

    ev_display_expected = [
        score_display(team_a, team_b, parse_score(score))
        for team_a, team_b, score in zip(
            predictions["team_a"], predictions["team_b"], predictions["recommended_score_ev"]
        )
    ]
    if ev_display_expected != predictions["recommended_score_ev_display"].tolist():
        issues.append("EV display strings are not Team A / Team B oriented.")

    if float(probability_sum_error(predictions, ["model_p_a_win", "model_p_draw", "model_p_b_win"]).max()) > 5e-4:
        issues.append("Model probabilities do not sum to ~1 within tolerance.")
    if float(probability_sum_error(predictions, ["template_p_a_win", "template_p_draw", "template_p_b_win"]).max()) > 5e-4:
        issues.append("Template implied probabilities do not sum to ~1 within tolerance.")

    recomputed = []
    for _, row in template.iterrows():
        _, _, M = _build_match_matrix(poisson_model, snapshots, row["team_a"], row["team_b"], row["date"])
        if abs(float(M.sum()) - 1.0) > 1e-9:
            issues.append(f"Scoreline matrix for match {int(row['match_number'])} does not sum to 1.")
            break
        outcome_probs = outcome_probs_from_matrix(M)
        predicted_outcome = OUTCOME_KEYS[int(outcome_probs.argmax())]
        ev_outcome = max(
            OUTCOME_KEYS,
            key=lambda key: expected_points_for_outcome(
                key,
                M,
                {"a_win": row["rate_a"], "draw": row["rate_draw"], "b_win": row["rate_b"]},
                rules,
            )[0],
        )
        recomputed.append(
            {
                "match_number": int(row["match_number"]),
                "poisson_outcome_name": named_outcome_from_key(predicted_outcome, row["team_a"], row["team_b"]),
                "ev_outcome_name": named_outcome_from_key(ev_outcome, row["team_a"], row["team_b"]),
            }
        )

    recomputed = pd.DataFrame(recomputed)
    check = predictions.merge(recomputed, on="match_number")
    if not check["most_probable_outcome"].equals(check["poisson_outcome_name"]):
        issues.append("`most_probable_outcome` does not match the highest model probability.")
    if not check["ev_max_outcome"].equals(check["ev_outcome_name"]):
        issues.append("`ev_max_outcome` does not match the highest expected-points outcome.")

    for _, row in predictions.iterrows():
        safe_score = parse_score(row["recommended_score_safe"])
        safe_outcome = named_outcome_from_key(outcome_key_from_score(safe_score), row["team_a"], row["team_b"])
        if safe_outcome != row["most_probable_outcome"]:
            issues.append(
                f"Safe recommendation outcome mismatch on match {row['match_number']}: "
                f"{row['recommended_score_safe']} vs {row['most_probable_outcome']}."
            )
            break

    group_checks = []
    for group, sub in template.groupby("group"):
        teams = sorted(set(sub["team_a"]) | set(sub["team_b"]))
        group_checks.append((group, len(teams), len(sub)))
        if len(teams) != 4 or len(sub) != 6:
            issues.append(f"Group {group} has {len(teams)} teams and {len(sub)} matches; expected 4 and 6.")
        counts = pd.concat([sub["team_a"], sub["team_b"]]).value_counts()
        invalid = counts[counts != 3]
        if not invalid.empty:
            issues.append(f"Group {group} team appearance counts are invalid: {invalid.to_dict()}")
    return issues, group_checks


def _template_probs(row):
    return f"A {row['template_p_a_win']:.3f} / D {row['template_p_draw']:.3f} / B {row['template_p_b_win']:.3f}"


def _model_probs(row):
    return f"A {row['model_p_a_win']:.3f} / D {row['model_p_draw']:.3f} / B {row['model_p_b_win']:.3f}"


def _edge_text(row):
    edges = {
        row["team_a"]: row["value_edge_a"],
        "Draw": row["value_edge_draw"],
        row["team_b"]: row["value_edge_b"],
    }
    name, value = max(edges.items(), key=lambda item: abs(item[1]))
    return f"{name} {value:+.3f}"


def _classify(row, snap_a, snap_b):
    if pd.isna(snap_a["fifa_rank"]) or pd.isna(snap_b["fifa_rank"]):
        return "likely data/model artefact"
    if "CONTRARIAN" in str(row["notes"]) or row["recommended_score_safe"] != row["recommended_score_ev"]:
        return "contrarian"
    return "safe"


def _reason(row, snap_a, snap_b):
    if pd.isna(snap_a["fifa_rank"]) or pd.isna(snap_b["fifa_rank"]):
        return "Missing FIFA rank/points in the latest snapshot makes the rating picture incomplete."
    if "SAFE_SCORE_ALIGNED_TO_MODEL_OUTCOME" in str(row["notes"]):
        return "Outcome probability and raw Poisson mode pointed in different directions, so the safe score was aligned to the model outcome."
    if "CONTRARIAN" in str(row["notes"]):
        return "Expected-points optimisation prefers a higher-odds outcome than the safest call."
    return "Model and template broadly agree on direction; the flag is mainly about confidence or edge size."


def _write_prediction_audit(predictions, template, snapshots):
    flagged_ids = set(SPECIFIC_MATCHES)
    flagged_ids |= set(
        predictions.loc[
            predictions[["value_edge_a", "value_edge_draw", "value_edge_b"]].abs().max(axis=1) > 0.20,
            "match_number",
        ]
    )
    flagged_ids |= set(predictions.loc[predictions["recommended_score_safe"] != predictions["recommended_score_ev"], "match_number"])
    flagged_ids |= set(
        predictions.loc[
            predictions[["model_p_a_win", "model_p_draw", "model_p_b_win"]].max(axis=1) > 0.80,
            "match_number",
        ]
    )

    flagged = predictions[predictions["match_number"].isin(sorted(flagged_ids))].copy()
    flagged = flagged.sort_values("match_number")

    specific_rows = predictions[predictions["match_number"].isin(SPECIFIC_MATCHES)].sort_values("match_number")
    over_20 = predictions[predictions[["value_edge_a", "value_edge_draw", "value_edge_b"]].abs().max(axis=1) > 0.20]
    safe_ev = predictions[predictions["recommended_score_safe"] != predictions["recommended_score_ev"]]
    high_conf = predictions[predictions[["model_p_a_win", "model_p_draw", "model_p_b_win"]].max(axis=1) > 0.80]

    AUDIT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_REPORT, "w") as handle:
        handle.write("# Prediction Sanity Audit\n\n")
        handle.write("## Key findings\n\n")
        handle.write("- Team A / Team B orientation matches the FIF8A template for all 72 rows.\n")
        handle.write("- Score displays are explicit Team A first strings: `Team A vs Team B: a-b`.\n")
        handle.write("- Qatar vs Switzerland is a real model prediction, not a formatting inversion: `Qatar vs Switzerland: 0-2` means Qatar 0, Switzerland 2.\n")
        handle.write(f"- Matches with >20 percentage-point model/template divergence: **{len(over_20)}**.\n")
        handle.write(f"- Matches where safe and EV recommendations differ: **{len(safe_ev)}**.\n")
        handle.write(f"- Matches above 80% model confidence: **{len(high_conf)}**.\n\n")

        handle.write("## Requested spot checks\n\n")
        handle.write("| # | Match | Template probs | Model probs | Safe score | EV score | Value edge | Label | Reason |\n")
        handle.write("|---|---|---|---|---|---|---|---|---|\n")
        for _, row in specific_rows.iterrows():
            snap_a = snapshots.loc[resolve(row["team_a"], snapshots)]
            snap_b = snapshots.loc[resolve(row["team_b"], snapshots)]
            handle.write(
                f"| {row['match_number']} | {row['team_a']} vs {row['team_b']} | {_template_probs(row)} | "
                f"{_model_probs(row)} | {row['recommended_score_safe_display']} | {row['recommended_score_ev_display']} | "
                f"{_edge_text(row)} | {_classify(row, snap_a, snap_b)} | {_reason(row, snap_a, snap_b)} |\n"
            )

        handle.write("\n## All flagged matches\n\n")
        handle.write("| # | Match | Template probs | Model probs | Safe score | EV score | Value edge | Label |\n")
        handle.write("|---|---|---|---|---|---|---|---|\n")
        for _, row in flagged.iterrows():
            snap_a = snapshots.loc[resolve(row["team_a"], snapshots)]
            snap_b = snapshots.loc[resolve(row["team_b"], snapshots)]
            handle.write(
                f"| {row['match_number']} | {row['team_a']} vs {row['team_b']} | {_template_probs(row)} | "
                f"{_model_probs(row)} | {row['recommended_score_safe_display']} | {row['recommended_score_ev_display']} | "
                f"{_edge_text(row)} | {_classify(row, snap_a, snap_b)} |\n"
            )


def _write_top_flags(predictions, template, snapshots):
    records = []
    for _, row in predictions.iterrows():
        match = template.loc[template["match_number"] == row["match_number"]].iloc[0]
        snap_a = snapshots.loc[resolve(row["team_a"], snapshots)]
        snap_b = snapshots.loc[resolve(row["team_b"], snapshots)]
        confidence = max(row["model_p_a_win"], row["model_p_draw"], row["model_p_b_win"])
        stronger_team = row["team_a"] if snap_a["elo"] > snap_b["elo"] else row["team_b"]
        predicted_team = row["most_probable_outcome"]

        if pd.isna(snap_a["fifa_rank"]) or pd.isna(snap_b["fifa_rank"]):
            records.append(
                {
                    "match_number": row["match_number"],
                    "match": f"{row['team_a']} vs {row['team_b']}",
                    "flag": "missing_fifa_snapshot",
                    "reason": "Latest pre-match snapshot is missing FIFA ranking data for one side.",
                }
            )

        if (
            abs(float(snap_a["elo"] - snap_b["elo"])) >= 150
            and predicted_team not in {"Draw", stronger_team}
            and confidence > 0.65
        ):
            records.append(
                {
                    "match_number": row["match_number"],
                    "match": f"{row['team_a']} vs {row['team_b']}",
                    "flag": "strong_team_predicted_to_lose",
                    "reason": "Model opposes a large Elo advantage with >65% confidence.",
                }
            )

        underdog_a = snap_a["elo"] < snap_b["elo"] and snap_a["fifa_rank"] > snap_b["fifa_rank"]
        underdog_b = snap_b["elo"] < snap_a["elo"] and snap_b["fifa_rank"] > snap_a["fifa_rank"]
        if underdog_a and row["model_p_a_win"] > 0.60:
            records.append(
                {
                    "match_number": row["match_number"],
                    "match": f"{row['team_a']} vs {row['team_b']}",
                    "flag": "underdog_a_above_60",
                    "reason": "Team A is behind on Elo and FIFA rank but still clears 60% win probability.",
                }
            )
        if underdog_b and row["model_p_b_win"] > 0.60:
            records.append(
                {
                    "match_number": row["match_number"],
                    "match": f"{row['team_a']} vs {row['team_b']}",
                    "flag": "underdog_b_above_60",
                    "reason": "Team B is behind on Elo and FIFA rank but still clears 60% win probability.",
                }
            )

        template_favorite = row["team_a"] if row["template_p_a_win"] > row["template_p_b_win"] else row["team_b"]
        if (
            abs(float(snap_a["elo"] - snap_b["elo"])) >= 75
            and predicted_team not in {"Draw", stronger_team}
            and predicted_team != template_favorite
        ):
            records.append(
                {
                    "match_number": row["match_number"],
                    "match": f"{row['team_a']} vs {row['team_b']}",
                    "flag": "disagrees_with_elo_and_template",
                    "reason": "Prediction opposes both the Elo gap and the template favourite.",
                }
            )

    flagged = pd.DataFrame(records).drop_duplicates().sort_values(["match_number", "flag"])

    FLAGS_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(FLAGS_REPORT, "w") as handle:
        handle.write("# Top Prediction Flags\n\n")
        if flagged.empty:
            handle.write("No intuition flags cleared the audit thresholds.\n")
            return
        handle.write("| # | Match | Flag | Reason |\n")
        handle.write("|---|---|---|---|\n")
        for _, row in flagged.iterrows():
            handle.write(f"| {row['match_number']} | {row['match']} | {row['flag']} | {row['reason']} |\n")


def main():
    rules, template, predictions, simulation, models, snapshots = _load_inputs()
    issues, _ = _validate(template, predictions, simulation, rules, models["poisson"], snapshots)
    if issues:
        raise ValueError("Audit failed:\n- " + "\n- ".join(issues))
    _write_prediction_audit(predictions, template, snapshots)
    _write_top_flags(predictions, template, snapshots)


if __name__ == "__main__":
    main()
