#!/usr/bin/env python3
"""Build final human-readable submission readiness reports.

This script is intentionally reporting-only. It reads validated prediction and
simulation outputs and writes submission audit/export artifacts without training
models or changing the Phase 4.5 prediction files.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
PREDICTIONS = ROOT / "outputs" / "predictions" / "fif8a_group_stage_predictions.csv"
ENSEMBLE = ROOT / "outputs" / "predictions" / "fif8a_group_stage_predictions_ensemble.csv"
SQUAD_OVERLAY = ROOT / "outputs" / "predictions" / "fif8a_group_stage_predictions_squad_overlay.csv"
SIMULATION = ROOT / "outputs" / "predictions" / "group_stage_simulation_summary.csv"
TEMPLATE = ROOT / "data" / "reference" / "fif8a_group_stage_template.csv"
TOP_FLAGS = ROOT / "outputs" / "reports" / "top_prediction_flags.md"
FINAL_SOURCE_STATUS = ROOT / "outputs" / "reports" / "final_prediction_source_status.md"

SUBMISSION_AUDIT = ROOT / "outputs" / "reports" / "submission_prediction_audit.md"
DECISION_MD = ROOT / "outputs" / "reports" / "safe_vs_ev_decision_table.md"
DECISION_CSV = ROOT / "outputs" / "predictions" / "submission_decision_table.csv"
MANUAL_REVIEW = ROOT / "outputs" / "reports" / "manual_review_shortlist.md"
GROUP_MD = ROOT / "outputs" / "reports" / "group_submission_view.md"
GROUP_CSV = ROOT / "outputs" / "predictions" / "group_submission_view.csv"
LAST8 = ROOT / "outputs" / "reports" / "last8_progression_readiness.md"
CHECKLIST = ROOT / "outputs" / "reports" / "final_submission_checklist.md"

SCORE_RE = re.compile(r"^\d{1,2}-\d{1,2}$")
SPECIAL_TEAMS = {
    "Qatar",
    "Switzerland",
    "Canada",
    "Iraq",
    "Norway",
    "Bosnia and Herzegovina",
}


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def parse_score(score: object) -> tuple[int, int] | None:
    if not isinstance(score, str) or not SCORE_RE.match(score):
        return None
    left, right = score.split("-", maxsplit=1)
    return int(left), int(right)


def score_outcome(score: object) -> str:
    parsed = parse_score(score)
    if parsed is None:
        return "invalid"
    team_a_goals, team_b_goals = parsed
    if team_a_goals > team_b_goals:
        return "a_win"
    if team_a_goals < team_b_goals:
        return "b_win"
    return "draw"


def outcome_label(row: pd.Series, outcome: str) -> str:
    if outcome == "a_win":
        return str(row["team_a"])
    if outcome == "b_win":
        return str(row["team_b"])
    if outcome == "draw":
        return "Draw"
    return "Invalid"


def model_probs(row: pd.Series) -> dict[str, float]:
    return {
        "a_win": float(row["model_p_a_win"]),
        "draw": float(row["model_p_draw"]),
        "b_win": float(row["model_p_b_win"]),
    }


def template_probs(row: pd.Series) -> dict[str, float]:
    return {
        "a_win": float(row["template_p_a_win"]),
        "draw": float(row["template_p_draw"]),
        "b_win": float(row["template_p_b_win"]),
    }


def top_outcome(probabilities: dict[str, float]) -> str:
    return max(probabilities, key=probabilities.get)


def read_suspicious_match_numbers() -> set[int]:
    if not TOP_FLAGS.exists():
        return set()
    numbers: set[int] = set()
    for line in TOP_FLAGS.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\|\s*(\d+)\s*\|", line)
        if match:
            numbers.add(int(match.group(1)))
    return numbers


def build_prediction_audit(pred: pd.DataFrame, template: pd.DataFrame) -> dict[str, object]:
    groups = pred.groupby("group").size().to_dict()
    fixtures = pred[["team_a", "team_b"]].apply(lambda row: tuple(sorted(row)), axis=1)
    checks = {
        "row_count": len(pred),
        "row_count_ok": len(pred) == 72,
        "group_match_counts": groups,
        "all_groups_have_6_matches": all(groups.get(group) == 6 for group in list("ABCDEFGHIJKL")),
        "unique_match_numbers": pred["match_number"].is_unique,
        "template_orientation_matches": pred[["match_number", "team_a", "team_b"]].equals(
            template[["match_number", "team_a", "team_b"]]
        ),
        "safe_scores_parseable": pred["recommended_score_safe"].map(parse_score).notna().all(),
        "ev_scores_parseable": pred["recommended_score_ev"].map(parse_score).notna().all(),
        "missing_model_probabilities": int(pred[["model_p_a_win", "model_p_draw", "model_p_b_win"]].isna().sum().sum()),
        "max_model_probability_sum_error": float(
            (pred[["model_p_a_win", "model_p_draw", "model_p_b_win"]].sum(axis=1) - 1).abs().max()
        ),
        "max_template_probability_sum_error": float(
            (pred[["template_p_a_win", "template_p_draw", "template_p_b_win"]].sum(axis=1) - 1).abs().max()
        ),
        "missing_odds": int(pred[["rate_a", "rate_draw", "rate_b"]].isna().sum().sum()),
        "duplicate_fixtures": int(fixtures.duplicated().sum()),
    }
    checks["probability_sums_ok"] = (
        checks["max_model_probability_sum_error"] <= 0.001
        and checks["max_template_probability_sum_error"] <= 0.001
    )
    return checks


def write_prediction_audit(pred: pd.DataFrame, template: pd.DataFrame) -> None:
    checks = build_prediction_audit(pred, template)
    SUBMISSION_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Submission Prediction Audit",
        "",
        f"- Prediction file audited: `{rel(PREDICTIONS)}`",
        f"- Rows: **{checks['row_count']}**; expected 72: **{checks['row_count_ok']}**",
        f"- Groups A-L each have 6 matches: **{checks['all_groups_have_6_matches']}**",
        f"- Match numbers unique: **{checks['unique_match_numbers']}**",
        f"- Team A / Team B orientation matches FIF8A template: **{checks['template_orientation_matches']}**",
        "- Score orientation: every score is interpreted as `team_a_goals-team_b_goals`.",
        f"- Safe score columns parseable: **{checks['safe_scores_parseable']}**",
        f"- EV score columns parseable: **{checks['ev_scores_parseable']}**",
        f"- Missing model probabilities: **{checks['missing_model_probabilities']}**",
        f"- Max model probability sum error: **{checks['max_model_probability_sum_error']:.6f}**",
        f"- Max template probability sum error: **{checks['max_template_probability_sum_error']:.6f}**",
        f"- Probability sums materially valid: **{checks['probability_sums_ok']}**",
        f"- Missing odds: **{checks['missing_odds']}**",
        f"- Duplicate fixtures ignoring orientation: **{checks['duplicate_fixtures']}**",
        "",
        "## Group Match Counts",
        "",
        "| Group | Matches |",
        "|---|--:|",
    ]
    for group in list("ABCDEFGHIJKL"):
        lines.append(f"| {group} | {checks['group_match_counts'].get(group, 0)} |")
    SUBMISSION_AUDIT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_decision_table(pred: pd.DataFrame, suspicious: set[int]) -> pd.DataFrame:
    rows = []
    for _, row in pred.iterrows():
        probs = model_probs(row)
        ordered_probs = sorted(probs.values(), reverse=True)
        model_top = top_outcome(probs)
        safe_outcome = score_outcome(row["recommended_score_safe"])
        ev_outcome = score_outcome(row["recommended_score_ev"])
        max_probability = ordered_probs[0]
        top_gap = ordered_probs[0] - ordered_probs[1]
        ev_uplift = float(row["expected_points_ev"] - row["expected_points_safe"])
        edge_values = [float(row["value_edge_a"]), float(row["value_edge_draw"]), float(row["value_edge_b"])]
        value_edge_max = max(abs(value) for value in edge_values)
        contrarian = "CONTRARIAN" in str(row.get("notes", "")).upper() or ev_outcome != model_top
        high_variance = bool(max_probability < 0.45 or top_gap < 0.12 or ev_outcome != model_top)
        suspicious_flag = int(row["match_number"]) in suspicious

        suggested = row["recommended_score_safe"]
        reason = "keep_safe_default"
        if row["recommended_score_safe"] != row["recommended_score_ev"]:
            if (
                ev_uplift >= 0.20
                and value_edge_max >= 0.08
                and not high_variance
                and not suspicious_flag
                and safe_outcome == ev_outcome
            ):
                suggested = row["recommended_score_ev"]
                reason = "consider_ev_same_outcome_clear_edge"
            elif ev_uplift >= 0.20 and not suspicious_flag and safe_outcome == ev_outcome:
                suggested = "manual_review"
                reason = "ev_uplift_exists_but_variance_or_edge_is_unclear"
            elif high_variance or suspicious_flag or ev_outcome != safe_outcome or ev_uplift >= 0.10:
                suggested = "manual_review"
                reason = "safe_ev_disagreement_requires_manual_review"
            else:
                reason = "keep_safe_ev_is_contrarian_or_low_uplift"

        rows.append(
            {
                "match_number": row["match_number"],
                "group": row["group"],
                "date": row["date"],
                "team_a": row["team_a"],
                "team_b": row["team_b"],
                "rate_a": row["rate_a"],
                "rate_draw": row["rate_draw"],
                "rate_b": row["rate_b"],
                "model_p_a_win": row["model_p_a_win"],
                "model_p_draw": row["model_p_draw"],
                "model_p_b_win": row["model_p_b_win"],
                "recommended_score_safe": row["recommended_score_safe"],
                "recommended_score_ev": row["recommended_score_ev"],
                "expected_points_safe": row["expected_points_safe"],
                "expected_points_ev": row["expected_points_ev"],
                "ev_uplift": round(ev_uplift, 3),
                "value_edge_max": round(value_edge_max, 4),
                "high_variance_flag": high_variance,
                "contrarian_flag": bool(contrarian),
                "suspicious_flag": suspicious_flag,
                "suggested_submission_score": suggested,
                "suggestion_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def write_decision_outputs(decisions: pd.DataFrame) -> None:
    DECISION_CSV.parent.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(DECISION_CSV, index=False)

    rows = [
        "# Safe vs EV Decision Table",
        "",
        f"- CSV output: `{rel(DECISION_CSV)}`",
        "- Rule: default to safe score; only suggest EV on same-outcome, meaningful-uplift, clear-edge, non-suspicious cases.",
        f"- Safe vs EV disagreements: **{int((decisions['recommended_score_safe'] != decisions['recommended_score_ev']).sum())}**",
        f"- Suggested EV scores: **{int(((decisions['recommended_score_safe'] != decisions['recommended_score_ev']) & (decisions['suggested_submission_score'] == decisions['recommended_score_ev'])).sum())}**",
        f"- Manual-review suggestions: **{int(decisions['suggested_submission_score'].eq('manual_review').sum())}**",
        "",
        "| # | Group | Match | Safe | EV | EV uplift | Edge max | High variance | Suspicious | Suggested | Reason |",
        "|---:|---|---|---|---|--:|--:|---:|---:|---|---|",
    ]
    for _, row in decisions.iterrows():
        rows.append(
            f"| {row['match_number']} | {row['group']} | {row['team_a']} vs {row['team_b']} | "
            f"{row['recommended_score_safe']} | {row['recommended_score_ev']} | {row['ev_uplift']:.3f} | "
            f"{row['value_edge_max']:.3f} | {row['high_variance_flag']} | {row['suspicious_flag']} | "
            f"{row['suggested_submission_score']} | {row['suggestion_reason']} |"
        )
    DECISION_MD.write_text("\n".join(rows) + "\n", encoding="utf-8")


def flag_reasons(decision_row: pd.Series, pred_row: pd.Series) -> list[str]:
    reasons = []
    if decision_row["recommended_score_safe"] != decision_row["recommended_score_ev"]:
        reasons.append("safe_vs_ev_disagreement")
    if max(model_probs(pred_row).values()) > 0.80:
        reasons.append("high_confidence_outlier")
    if decision_row["high_variance_flag"]:
        reasons.append("high_variance")
    if decision_row["value_edge_max"] > 0.20:
        reasons.append("model_template_edge_gt_20pp")
    if pred_row["team_a"] in SPECIAL_TEAMS or pred_row["team_b"] in SPECIAL_TEAMS:
        reasons.append("special_watchlist_team")
    if decision_row["suggested_submission_score"] != decision_row["recommended_score_safe"]:
        reasons.append("suggestion_differs_from_safe")
    if bool(decision_row["suspicious_flag"]):
        reasons.append("top_prediction_flags_report")
    return reasons


def write_manual_review(pred: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    pred_by_match = pred.set_index("match_number")
    review_rows = []
    for _, decision in decisions.iterrows():
        pred_row = pred_by_match.loc[decision["match_number"]]
        reasons = flag_reasons(decision, pred_row)
        if not reasons:
            continue
        probs = model_probs(pred_row)
        t_probs = template_probs(pred_row)
        recommendation = "manual-review" if "top_prediction_flags_report" in reasons or "high_variance" in reasons else "keep safe"
        if decision["suggested_submission_score"] == decision["recommended_score_ev"] and decision["recommended_score_safe"] != decision["recommended_score_ev"]:
            recommendation = "consider EV"
        elif decision["suggested_submission_score"] == "manual_review":
            recommendation = "manual-review"
        review_rows.append(
            {
                **decision.to_dict(),
                "flag_reasons": ", ".join(reasons),
                "review_recommendation": recommendation,
                "model_prob_summary": f"A {probs['a_win']:.3f} / D {probs['draw']:.3f} / B {probs['b_win']:.3f}",
                "template_prob_summary": f"A {t_probs['a_win']:.3f} / D {t_probs['draw']:.3f} / B {t_probs['b_win']:.3f}",
            }
        )

    review = pd.DataFrame(review_rows).sort_values(
        by=["suspicious_flag", "value_edge_max", "ev_uplift"],
        ascending=[False, False, False],
    )
    lines = [
        "# Manual Review Shortlist",
        "",
        f"- Matches flagged: **{len(review)}**",
        "- Includes safe-vs-EV disagreements, high-confidence outliers, high-variance matches, >20pp model/template edges, watchlist teams, and any suggested non-safe score.",
        "",
        "| # | Match | Flags | Safe | EV | Model probabilities | Template probabilities | Recommendation |",
        "|---:|---|---|---|---|---|---|---|",
    ]
    for _, row in review.iterrows():
        lines.append(
            f"| {row['match_number']} | {row['team_a']} vs {row['team_b']} | {row['flag_reasons']} | "
            f"{row['recommended_score_safe']} | {row['recommended_score_ev']} | {row['model_prob_summary']} | "
            f"{row['template_prob_summary']} | {row['review_recommendation']} |"
        )
    MANUAL_REVIEW.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return review


def group_confidence(group: pd.DataFrame) -> tuple[str, list[str]]:
    ordered = group.sort_values(["p_top2", "expected_points"], ascending=[False, False]).reset_index(drop=True)
    second_third_gap = float(ordered.loc[1, "p_top2"] - ordered.loc[2, "p_top2"])
    first_second_order_gap = float(ordered.loc[0, "p_finish_1st"] - ordered.loc[1, "p_finish_1st"])
    flags = []
    if second_third_gap < 0.10:
        flags.append("close_2nd_3rd_race")
    if first_second_order_gap < 0.15 or second_third_gap < 0.15:
        flags.append("high_exact_standing_uncertainty")
    if second_third_gap >= 0.20 and first_second_order_gap >= 0.20:
        confidence = "high"
    elif second_third_gap >= 0.10:
        confidence = "medium"
    else:
        confidence = "low"
    return confidence, flags


def build_group_view(sim: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_name, group in sim.groupby("group", sort=True):
        ordered = group.sort_values(["p_top2", "expected_points"], ascending=[False, False]).reset_index(drop=True)
        confidence, flags = group_confidence(group)
        for standing, (_, row) in enumerate(ordered.iterrows(), start=1):
            best_third_signal = float(row["p_advance_with_best_thirds"] - row["p_top2"])
            rows.append(
                {
                    "group": group_name,
                    "team": row["team"],
                    "expected_rank_order": " > ".join(ordered["team"].tolist()),
                    "p_finish_1st": row["p_finish_1st"],
                    "p_finish_2nd": row["p_finish_2nd"],
                    "p_top2": row["p_top2"],
                    "p_top3": row["p_top3"],
                    "expected_points": row["expected_points"],
                    "expected_goal_difference": row["expected_goal_difference"],
                    "suggested_group_standing": standing,
                    "confidence_level": confidence,
                    "group_flags": ", ".join(flags) if flags else "none",
                    "likely_best_third_signal": best_third_signal if standing == 3 else 0.0,
                    "likely_best_third_team": bool(standing == 3 and best_third_signal >= 0.20),
                }
            )
    return pd.DataFrame(rows)


def write_group_view(group_view: pd.DataFrame) -> None:
    GROUP_CSV.parent.mkdir(parents=True, exist_ok=True)
    group_view.to_csv(GROUP_CSV, index=False)
    summary = group_view.drop_duplicates("group")["confidence_level"].value_counts().to_dict()
    lines = [
        "# Group Submission View",
        "",
        f"- CSV output: `{rel(GROUP_CSV)}`",
        f"- Confidence summary: `{summary}`",
        "",
        "## Suggested Standings",
        "",
    ]
    for group_name, group in group_view.groupby("group", sort=True):
        flags = group["group_flags"].iloc[0]
        confidence = group["confidence_level"].iloc[0]
        lines.extend(
            [
                f"### Group {group_name}",
                "",
                f"- Confidence: **{confidence}**",
                f"- Flags: `{flags}`",
                "",
                "| Standing | Team | Exp pts | Exp GD | P 1st | P 2nd | P top2 | P top3 | Best-third signal |",
                "|---:|---|--:|--:|--:|--:|--:|--:|--:|",
            ]
        )
        for _, row in group.sort_values("suggested_group_standing").iterrows():
            lines.append(
                f"| {row['suggested_group_standing']} | {row['team']} | {row['expected_points']:.3f} | "
                f"{row['expected_goal_difference']:.3f} | {row['p_finish_1st']:.3f} | {row['p_finish_2nd']:.3f} | "
                f"{row['p_top2']:.3f} | {row['p_top3']:.3f} | {row['likely_best_third_signal']:.3f} |"
            )
        lines.append("")

    likely_thirds = group_view.loc[group_view["likely_best_third_team"]].sort_values(
        "likely_best_third_signal", ascending=False
    )
    lines.extend(["## Likely Best-Third Candidates", ""])
    if likely_thirds.empty:
        lines.append("- Simulation output does not identify strong best-third-only candidates under this threshold.")
    else:
        lines.extend(["| Group | Team | Best-third signal | P advance with best thirds |", "|---|---|--:|--:|"])
        sim = pd.read_csv(SIMULATION)
        adv = sim.set_index(["group", "team"])["p_advance_with_best_thirds"].to_dict()
        for _, row in likely_thirds.iterrows():
            lines.append(
                f"| {row['group']} | {row['team']} | {row['likely_best_third_signal']:.3f} | "
                f"{adv[(row['group'], row['team'])]:.3f} |"
            )
    GROUP_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_last8_readiness(group_view: pd.DataFrame) -> None:
    knockout_files = sorted(
        str(path.relative_to(ROOT))
        for path in (list((ROOT / "scripts").glob("*knockout*")) + list((ROOT / "outputs").glob("**/*knockout*")))
    )
    top_group_teams = group_view.sort_values(["p_top2", "expected_points"], ascending=False).head(16)
    lines = [
        "# Last-8 / Progression Readiness",
        "",
        f"- Knockout simulation files found: `{knockout_files or 'none'}`",
        "- Current repo status: group-stage simulation exists; no full bracket/knockout simulation output is present.",
        "- Phase 4.5 outputs are sufficient to fill group-match scores and a probability-informed group-standing view.",
        "- Phase 4.5 outputs are not sufficient to fill QF/SF/final/winner as a final modelled block without a bracket simulation or manual selection.",
        "",
        "## Missing For Final Last-8 Block",
        "",
        "- Round-of-32 bracket construction from group winners/runners-up/best-thirds.",
        "- Match probability application through knockout rounds.",
        "- Extra-time/penalty handling or a deterministic qualification probability abstraction.",
        "- Path-aware QF/SF/final/winner probabilities.",
        "",
        "## Recommended Next Modelling Step",
        "",
        "- Build a bracket simulation using group advancement probabilities plus the existing match probability model.",
        "- If time is short, use a manual top-8/top-4/top-2/winner selection informed by team strength and group-path favourability, clearly marked as manual.",
        "",
        "## Provisional Shortlist Only",
        "",
        "| Team | Group | Suggested standing | P top2 | Expected points | Expected GD |",
        "|---|---|--:|--:|--:|--:|",
    ]
    for _, row in top_group_teams.iterrows():
        lines.append(
            f"| {row['team']} | {row['group']} | {row['suggested_group_standing']} | "
            f"{row['p_top2']:.3f} | {row['expected_points']:.3f} | {row['expected_goal_difference']:.3f} |"
        )
    LAST8.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_checklist(decisions: pd.DataFrame, review: pd.DataFrame) -> None:
    final_status = FINAL_SOURCE_STATUS.read_text(encoding="utf-8").strip() if FINAL_SOURCE_STATUS.exists() else "No final source status report present."
    lines = [
        "# Final Submission Checklist",
        "",
        "## Use Today",
        "",
        f"- Group-match prediction file: `{rel(PREDICTIONS)}`",
        f"- Decision helper CSV: `{rel(DECISION_CSV)}`",
        f"- Group standing helper CSV: `{rel(GROUP_CSV)}`",
        "- Squad overlay is context-only and did not change probabilities.",
        "",
        "## Phase 5C Status",
        "",
        "- Separate Phase 5C historical squad/player/coach work is not included in this submission pack.",
        "- If Phase 5C changes final prediction recommendations, rerun this submission pack afterward.",
        "",
        "## Last Friendlies / Data Refresh",
        "",
        "- Last friendlies and any late FIFA rankings still need to be imported before final submission if they are available.",
        "- Re-run the full prediction pipeline only in the main modelling worktree, not from this audit-only task.",
        "",
        "## Commands To Rerun Before Submission",
        "",
        "- `python scripts/generate_group_stage_predictions.py`",
        "- `python scripts/audit_group_stage_predictions.py`",
        "- `python scripts/simulate_group_stage.py`",
        "- `python scripts/build_submission_pack.py`",
        "- `pytest`",
        "",
        "## Files To Inspect Manually",
        "",
        f"- `{rel(DECISION_MD)}`",
        f"- `{rel(MANUAL_REVIEW)}`",
        f"- `{rel(GROUP_MD)}`",
        f"- `{rel(LAST8)}`",
        "- `outputs/reports/prediction_sanity_audit.md`",
        "- `outputs/reports/top_prediction_flags.md`",
        "",
        "## Known Risks",
        "",
        f"- Manual-review matches: **{len(review)}**",
        f"- Safe-vs-EV disagreements: **{int((decisions['recommended_score_safe'] != decisions['recommended_score_ev']).sum())}**",
        "- Some team/match flags reflect missing FIFA snapshot data in existing reports.",
        "- Group-standing and progression bonuses are flat; avoid over-weighting high-odd contrarian match picks.",
        "- Last-8 block is not yet modelled path-aware.",
        "",
        "## Manual Decisions Still Required",
        "",
        "- Decide whether to accept any EV score suggestions or keep all safe picks.",
        "- Resolve manual-review shortlist before filling the group-stage score sheet.",
        "- Fill group standings from `group_submission_view.csv` after manual review.",
        "- Create Last-8/QF/SF/final/winner entries using a future bracket model or documented manual shortlist.",
        "",
        "## Current Final Source Status",
        "",
        final_status,
        "",
    ]
    CHECKLIST.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    pred = pd.read_csv(PREDICTIONS)
    template = pd.read_csv(TEMPLATE)
    sim = pd.read_csv(SIMULATION)
    suspicious = read_suspicious_match_numbers()

    write_prediction_audit(pred, template)
    decisions = build_decision_table(pred, suspicious)
    write_decision_outputs(decisions)
    review = write_manual_review(pred, decisions)
    group_view = build_group_view(sim)
    write_group_view(group_view)
    write_last8_readiness(group_view)
    write_checklist(decisions, review)

    print(f"Wrote {rel(SUBMISSION_AUDIT)}")
    print(f"Wrote {rel(DECISION_CSV)}")
    print(f"Wrote {rel(DECISION_MD)}")
    print(f"Wrote {rel(MANUAL_REVIEW)}")
    print(f"Wrote {rel(GROUP_CSV)}")
    print(f"Wrote {rel(GROUP_MD)}")
    print(f"Wrote {rel(LAST8)}")
    print(f"Wrote {rel(CHECKLIST)}")


if __name__ == "__main__":
    main()
