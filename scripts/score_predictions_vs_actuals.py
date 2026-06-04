#!/usr/bin/env python3
"""Score the active candidate's predictions vs actual results (Travel Mode, Task E).

Compares the active candidate's frozen ``final_recommended_score`` for every
*played* group match against the manually entered scoreline, applying the
group-stage scoring of RULES_AND_SCORING.md. Does NOT retrain, fetch APIs, or
change any prediction.

Inputs:
    data/live/active_candidate.yml -> active candidate score predictions
    data/live/scores_override.csv
    data/reference/fif8a_group_stage_template.csv  (template odds)
    data/reference/scoring_rules.yml

Outputs:
    outputs/live/prediction_vs_actual.csv
    outputs/live/prediction_vs_actual.json
    outputs/live/scoring_summary.csv
    outputs/live/scoring_summary.json
    outputs/reports/prediction_vs_actual_report.md
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.live.active_candidate import load_active_candidate
from src.live.prediction_scoring import (
    load_scoring_rules,
    load_template_odds,
    score_predictions_vs_actuals,
    summarise,
)
from src.live.scores_override import OVERRIDE_PATH, load_override, utc_now_iso

ROOT = Path(__file__).resolve().parents[1]
LIVE_DIR = ROOT / "outputs" / "live"
REPORTS_DIR = ROOT / "outputs" / "reports"


def _tick(flag: bool) -> str:
    return "✅" if flag else "❌"


def write_report(detail: pd.DataFrame, summary: dict, candidate: dict, path: Path) -> None:
    lines = [
        "# Prediction vs Actual Scoring Report",
        "",
        f"_Generated: {utc_now_iso()}_",
        "",
        f"- Active candidate: **{candidate['name']}** "
        f"(`{candidate['active_candidate_dir']}`)",
        f"- Score predictions file: `{candidate['score_predictions_file']}`",
        "",
        "## Totals",
        "",
        f"- Played matches scored: **{summary['played_matches']}**",
        f"- Outcomes correct: **{summary['outcomes_correct']}**",
        f"- Exact goal differences correct: **{summary['goal_differences_correct']}**",
        f"- Exact scores correct: **{summary['exact_scores_correct']}**",
        f"- **Total points earned: {summary['total_points']:g}**",
        f"- Possible points (played matches): {summary['possible_points_for_played_matches']:g}",
        f"- Points missed: {summary['points_missed']:g}",
        f"- Average points / played match: {summary['average_points_per_played_match']:g}",
        "",
        "## Scoring rule",
        "",
        "Correct outcome = `6 × template odd` of the predicted outcome. Exact goal "
        "difference adds a flat `+2`, exact score adds a flat `+3`; both bonuses "
        "apply only when the outcome is correct. A wrong outcome scores zero.",
        "",
    ]

    if summary["played_matches"]:
        lines += [
            "## Per-match detail",
            "",
            "| # | Match | Pred | Actual | Out | GD | Exact | Odd | Pts | Max | Missed |",
            "|---|-------|------|--------|-----|----|----|-----|-----|-----|--------|",
        ]
        for _, r in detail.iterrows():
            lines.append(
                f"| {r['match_number']} | {r['team_a']} v {r['team_b']} | "
                f"{r['predicted_score']} | {r['actual_score']} | "
                f"{_tick(r['outcome_correct'])} | {_tick(r['goal_difference_correct'])} | "
                f"{_tick(r['exact_score_correct'])} | {r['applicable_odd']:g} | "
                f"{r['total_points']:g} | {r['max_possible_points_for_match']:g} | "
                f"{r['points_missed']:g} |"
            )
        lines.append("")

        lines += ["## Points by group", "", "| Group | Played | Points | Possible | Missed |",
                  "|-------|--------|--------|----------|--------|"]
        for group, g in sorted(summary["total_by_group"].items()):
            lines.append(
                f"| {group} | {g['played_matches']} | {g['total_points']:g} | "
                f"{g['possible_points']:g} | {g['points_missed']:g} |"
            )
        lines.append("")
    else:
        lines += ["_No played matches yet — enter scores to see prediction-vs-actual points._", ""]

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    candidate = load_active_candidate()
    predictions = candidate.load_score_predictions()
    scores = load_override(OVERRIDE_PATH)
    odds = load_template_odds()
    rules = load_scoring_rules()

    detail = score_predictions_vs_actuals(predictions, scores, odds, rules)
    summary = summarise(detail)
    candidate_info = candidate.as_dict()

    detail.to_csv(LIVE_DIR / "prediction_vs_actual.csv", index=False)
    (LIVE_DIR / "prediction_vs_actual.json").write_text(
        json.dumps(
            {
                "generated_at": utc_now_iso(),
                "active_candidate": candidate_info,
                "matches": detail.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Flat one-row summary CSV plus per-group rows for convenience.
    summary_flat = {k: v for k, v in summary.items() if k != "total_by_group"}
    pd.DataFrame([summary_flat]).to_csv(LIVE_DIR / "scoring_summary.csv", index=False)
    (LIVE_DIR / "scoring_summary.json").write_text(
        json.dumps(
            {"generated_at": utc_now_iso(), "active_candidate": candidate_info, **summary},
            indent=2,
        ),
        encoding="utf-8",
    )

    write_report(detail, summary, candidate_info, REPORTS_DIR / "prediction_vs_actual_report.md")
    print(
        f"Scored {summary['played_matches']} played match(es) for "
        f"{candidate_info['name']}: {summary['total_points']:g} points "
        f"(missed {summary['points_missed']:g}). Wrote prediction_vs_actual + scoring_summary."
    )


if __name__ == "__main__":
    main()
