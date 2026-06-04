#!/usr/bin/env python3
"""Build final submission pack from existing validated outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
DECISIONS = ROOT / "outputs" / "predictions" / "submission_decision_table.csv"
GROUP_VIEW = ROOT / "outputs" / "predictions" / "group_submission_view.csv"
LAST8 = ROOT / "outputs" / "predictions" / "last8_recommendations.csv"
LAST8_REPORT = ROOT / "outputs" / "reports" / "last8_recommendation_report.md"
BRACKET_AUDIT = ROOT / "outputs" / "reports" / "bracket_structure_audit.md"

FINAL_PACK_MD = ROOT / "outputs" / "reports" / "final_submission_pack.md"
FINAL_PACK_CSV = ROOT / "outputs" / "predictions" / "final_submission_pack.csv"
FINAL_SCORES = ROOT / "outputs" / "predictions" / "final_group_score_predictions.csv"
FINAL_STANDINGS = ROOT / "outputs" / "predictions" / "final_group_standing_predictions.csv"
FINAL_LAST8 = ROOT / "outputs" / "predictions" / "final_last8_predictions.csv"
SAFE_EV_POLICY = ROOT / "outputs" / "reports" / "final_safe_vs_ev_policy.md"


def final_score_predictions(decisions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in decisions.iterrows():
        manual = row["suggested_submission_score"] == "manual_review"
        final_score = row["recommended_score_safe"] if manual else row["suggested_submission_score"]
        rows.append(
            {
                "match_number": row["match_number"],
                "group": row["group"],
                "team_a": row["team_a"],
                "team_b": row["team_b"],
                "final_recommended_score": final_score,
                "safe_score": row["recommended_score_safe"],
                "ev_score": row["recommended_score_ev"],
                "manual_review_flag": bool(manual),
                "reason": row["suggestion_reason"],
            }
        )
    return pd.DataFrame(rows)


def final_group_standings(group_view: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_name, group in group_view.groupby("group", sort=True):
        ordered = group.sort_values("suggested_group_standing")
        ranks = ordered.set_index("suggested_group_standing")["team"].to_dict()
        rows.append(
            {
                "group": group_name,
                "rank_1": ranks.get(1, ""),
                "rank_2": ranks.get(2, ""),
                "rank_3": ranks.get(3, ""),
                "rank_4": ranks.get(4, ""),
                "confidence": ordered["confidence_level"].iloc[0],
                "notes": ordered["group_flags"].iloc[0],
            }
        )
    return pd.DataFrame(rows)


def final_last8_rows() -> pd.DataFrame:
    if not LAST8.exists():
        return pd.DataFrame(
            [
                {
                    "stage": "unavailable",
                    "rank": "",
                    "team": "",
                    "expected_points_estimate": "",
                    "alternatives": "Path-aware Last-8 not modelled because knockout bracket mapping is missing.",
                }
            ]
        )
    last8 = pd.read_csv(LAST8)
    if "expected_points_estimate" not in last8.columns:
        last8["expected_points_estimate"] = ""
    if "alternatives" not in last8.columns:
        last8["alternatives"] = "See last8_recommendation_report.md"
    return last8


def write_safe_ev_policy(decisions: pd.DataFrame) -> None:
    disagreements = int((decisions["recommended_score_safe"] != decisions["recommended_score_ev"]).sum())
    manual = int(decisions["suggested_submission_score"].eq("manual_review").sum())
    lines = [
        "# Final Safe vs EV Policy",
        "",
        "- Safe score is the default final score.",
        "- EV score is used only if the existing decision table marks it as a non-suspicious, meaningful-uplift choice.",
        "- High-variance contrarian picks remain optional/manual-review, not default.",
        "- Manual-review flags are preserved in `final_group_score_predictions.csv`.",
        f"- Safe-vs-EV disagreements: **{disagreements}**",
        f"- Manual-review group matches: **{manual}**",
        "- In this run, manual-review rows retain the safe score as the machine default until a human overrides them.",
    ]
    SAFE_EV_POLICY.parent.mkdir(parents=True, exist_ok=True)
    SAFE_EV_POLICY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    decisions = pd.read_csv(DECISIONS)
    group_view = pd.read_csv(GROUP_VIEW)
    scores = final_score_predictions(decisions)
    standings = final_group_standings(group_view)
    last8 = final_last8_rows()

    FINAL_SCORES.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(FINAL_SCORES, index=False)
    standings.to_csv(FINAL_STANDINGS, index=False)
    last8.to_csv(FINAL_LAST8, index=False)
    write_safe_ev_policy(decisions)

    pack_rows = []
    for _, row in scores.iterrows():
        pack_rows.append({"section": "group_score", **row.to_dict()})
    for _, row in standings.iterrows():
        pack_rows.append({"section": "group_standing", **row.to_dict()})
    for _, row in last8.iterrows():
        pack_rows.append({"section": "last8", **row.to_dict()})
    pd.DataFrame(pack_rows).to_csv(FINAL_PACK_CSV, index=False)

    low_groups = standings.loc[standings["confidence"].eq("low"), "group"].tolist()
    manual_matches = scores[scores["manual_review_flag"]]
    last8_ready = not last8["stage"].astype(str).eq("unavailable").any()
    last8_status = (
        "Last-8 path-aware recommendations are ready from official FIFA bracket mapping and Annexe C."
        if last8_ready
        else "Last-8 path-aware recommendations are **not ready** because exact Round-of-32 bracket mapping is missing."
    )
    lines = [
        "# Final Submission Pack",
        "",
        f"- Group score predictions: `{FINAL_SCORES.relative_to(ROOT)}` ({len(scores)} matches)",
        f"- Group standing predictions: `{FINAL_STANDINGS.relative_to(ROOT)}` ({len(standings)} groups)",
        f"- Last-8 predictions: `{FINAL_LAST8.relative_to(ROOT)}`",
        f"- Combined CSV: `{FINAL_PACK_CSV.relative_to(ROOT)}`",
        f"- Safe-vs-EV policy: `{SAFE_EV_POLICY.relative_to(ROOT)}`",
        "",
        "## Status",
        "",
        "- Group-stage score predictions are ready, with manual-review flags preserved.",
        "- Group standings are ready from current group simulation summary.",
        f"- {last8_status}",
        f"- Bracket audit: `{BRACKET_AUDIT.relative_to(ROOT)}`",
        "",
        "## Manual Review",
        "",
        f"- Group matches requiring human decision: **{len(manual_matches)}**",
        f"- Low-confidence groups: `{low_groups or 'none'}`",
        "- Last-8 borderline teams are listed in `last8_recommendation_report.md`." if last8_ready else "- Last-8 borderline teams cannot be computed path-aware until the bracket mapping is supplied.",
        "",
        "## First Manual-Review Matches",
        "",
        "| # | Match | Safe | EV | Default | Reason |",
        "|---:|---|---|---|---|---|",
    ]
    for _, row in manual_matches.head(20).iterrows():
        lines.append(
            f"| {row['match_number']} | {row['team_a']} vs {row['team_b']} | {row['safe_score']} | "
            f"{row['ev_score']} | {row['final_recommended_score']} | {row['reason']} |"
        )
    if LAST8_REPORT.exists():
        lines.extend(["", "## Last-8 Report", "", f"- `{LAST8_REPORT.relative_to(ROOT)}`"])
    FINAL_PACK_MD.parent.mkdir(parents=True, exist_ok=True)
    FINAL_PACK_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {FINAL_SCORES.relative_to(ROOT)}")
    print(f"Wrote {FINAL_STANDINGS.relative_to(ROOT)}")
    print(f"Wrote {FINAL_LAST8.relative_to(ROOT)}")
    print(f"Wrote {FINAL_PACK_CSV.relative_to(ROOT)}")
    print(f"Wrote {FINAL_PACK_MD.relative_to(ROOT)}")
    print(f"Wrote {SAFE_EV_POLICY.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
