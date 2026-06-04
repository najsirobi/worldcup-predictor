#!/usr/bin/env python3
"""Generate Last-8 recommendations from full tournament simulation output."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
SUMMARY = ROOT / "outputs" / "predictions" / "full_tournament_simulation_summary.csv"
OUT = ROOT / "outputs" / "predictions" / "last8_recommendations.csv"
FINAL_LAST8 = ROOT / "outputs" / "predictions" / "final_last8_predictions.csv"
REPORT = ROOT / "outputs" / "reports" / "last8_recommendation_report.md"


def build_last8_recommendations(summary: pd.DataFrame) -> dict[str, list[str] | str | float]:
    required = {"team", "p_reach_qf", "p_reach_sf", "p_reach_final", "p_win_world_cup"}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"Full tournament summary missing columns: {sorted(missing)}")
    return {
        "quarter_finalists": summary.sort_values("p_reach_qf", ascending=False).head(8)["team"].tolist(),
        "semi_finalists": summary.sort_values("p_reach_sf", ascending=False).head(4)["team"].tolist(),
        "finalists": summary.sort_values("p_reach_final", ascending=False).head(2)["team"].tolist(),
        "winner": str(summary.sort_values("p_win_world_cup", ascending=False).iloc[0]["team"]),
        "expected_points_estimate": float(
            summary.sort_values("p_reach_qf", ascending=False).head(8)["p_reach_qf"].sum() * 20
            + summary.sort_values("p_reach_sf", ascending=False).head(4)["p_reach_sf"].sum() * 40
            + summary.sort_values("p_reach_final", ascending=False).head(2)["p_reach_final"].sum() * 60
            + summary["p_win_world_cup"].max() * 100
        ),
    }


def _stage_probability_column(stage: str) -> str:
    return {
        "quarter_finalist": "p_reach_qf",
        "semi_finalist": "p_reach_sf",
        "finalist": "p_reach_final",
        "winner": "p_win_world_cup",
    }[stage]


def _stage_points(stage: str) -> int:
    return {
        "quarter_finalist": 20,
        "semi_finalist": 40,
        "finalist": 60,
        "winner": 100,
    }[stage]


def write_unavailable_report() -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        "\n".join(
            [
                "# Last-8 Recommendation Report",
                "",
                "- Last-8 recommendations created: **False**",
                f"- Missing simulation summary: `{SUMMARY.relative_to(ROOT)}`",
                "- Path-aware recommendations require `data/reference/knockout_bracket_mapping.csv` and successful `scripts/simulate_full_tournament.py`.",
                "- Do not fill Last-8 as final model output from this report.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    if not SUMMARY.exists():
        write_unavailable_report()
        raise SystemExit(2)
    summary = pd.read_csv(SUMMARY)
    rec = build_last8_recommendations(summary)
    summary_by_team = summary.drop_duplicates("team").set_index("team")
    rows = []
    for stage, teams in [
        ("quarter_finalist", rec["quarter_finalists"]),
        ("semi_finalist", rec["semi_finalists"]),
        ("finalist", rec["finalists"]),
    ]:
        for rank, team in enumerate(teams, start=1):
            probability = float(summary_by_team.loc[team, _stage_probability_column(stage)])
            rows.append(
                {
                    "stage": stage,
                    "rank": rank,
                    "team": team,
                    "probability": probability,
                    "stage_points": _stage_points(stage),
                    "expected_points": probability * _stage_points(stage),
                    "selection_type": "safe_highest_probability",
                    "alternatives": "See last8_recommendation_report.md",
                }
            )
    winner_probability = float(summary_by_team.loc[rec["winner"], "p_win_world_cup"])
    rows.append(
        {
            "stage": "winner",
            "rank": 1,
            "team": rec["winner"],
            "probability": winner_probability,
            "stage_points": 100,
            "expected_points": winner_probability * 100,
            "selection_type": "safe_highest_probability",
            "alternatives": "See last8_recommendation_report.md",
        }
    )
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    out.assign(expected_points_estimate=rec["expected_points_estimate"]).to_csv(FINAL_LAST8, index=False)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Last-8 Recommendation Report",
        "",
        "- Last-8 recommendations created: **True**",
        f"- Source: `{SUMMARY.relative_to(ROOT)}`",
        f"- Estimated expected progression points: **{rec['expected_points_estimate']:.2f}**",
        "- Default rule: flat progression points mean highest stage probability is selected.",
        "- Contrarian alternatives are review-only, not defaults.",
        "",
        "## Recommendations",
        "",
        f"- Quarter-finalists: {', '.join(rec['quarter_finalists'])}",
        f"- Semi-finalists: {', '.join(rec['semi_finalists'])}",
        f"- Finalists: {', '.join(rec['finalists'])}",
        f"- Winner: {rec['winner']}",
        "",
        "## High-Upside Alternatives",
        "",
        "| Stage | Team | Probability |",
        "|---|---|--:|",
    ]
    for stage, column, selected_count in [
        ("QF", "p_reach_qf", 8),
        ("SF", "p_reach_sf", 4),
        ("Final", "p_reach_final", 2),
        ("Winner", "p_win_world_cup", 1),
    ]:
        alternatives = summary.sort_values(column, ascending=False).iloc[selected_count : selected_count + 4]
        for _, row in alternatives.iterrows():
            lines.append(f"| {stage} | {row['team']} | {row[column]:.3f} |")
    lines.extend(
        [
            "",
            "## Borderline Teams",
            "",
            "| Stage | Team | Probability | Note |",
            "|---|---|--:|---|",
        ]
    )
    selected_by_stage = {
        "QF": set(rec["quarter_finalists"]),
        "SF": set(rec["semi_finalists"]),
        "Final": set(rec["finalists"]),
        "Winner": {rec["winner"]},
    }
    for stage, column, selected_count in [
        ("QF", "p_reach_qf", 8),
        ("SF", "p_reach_sf", 4),
        ("Final", "p_reach_final", 2),
        ("Winner", "p_win_world_cup", 1),
    ]:
        ranked = summary.sort_values(column, ascending=False).iloc[max(0, selected_count - 2) : selected_count + 3]
        for _, row in ranked.iterrows():
            note = "selected" if row["team"] in selected_by_stage[stage] else "alternative"
            lines.append(f"| {stage} | {row['team']} | {row[column]:.3f} | {note} |")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)}")
    print(f"Wrote {FINAL_LAST8.relative_to(ROOT)}")
    print(f"Wrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
