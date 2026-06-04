#!/usr/bin/env python3
"""Build final group standings from deterministic auto-selected scores."""

from __future__ import annotations

from pathlib import Path
import shutil

import pandas as pd

from src.evaluation.auto_consensus import parse_score, validate_final_scores


ROOT = Path(__file__).parent.parent
TEMPLATE = ROOT / "data" / "reference" / "fif8a_group_stage_template.csv"
AUTO_SCORES = ROOT / "outputs" / "predictions" / "final_group_score_predictions_auto.csv"
AUTO_STANDINGS = ROOT / "outputs" / "predictions" / "final_group_standing_predictions_auto.csv"
AUTO_LAST8 = ROOT / "outputs" / "predictions" / "final_last8_predictions_auto.csv"
AUTO_PACK = ROOT / "outputs" / "predictions" / "final_submission_pack_auto.csv"
AUTO_CANDIDATES = ROOT / "outputs" / "predictions" / "auto_score_candidates.csv"

V1_SCORES = ROOT / "outputs" / "final_candidate_v1" / "final_group_score_predictions.csv"
V1_STANDINGS = ROOT / "outputs" / "final_candidate_v1" / "final_group_standing_predictions.csv"
V1_LAST8 = ROOT / "outputs" / "final_candidate_v1" / "final_last8_predictions.csv"

STANDING_REPORT = ROOT / "outputs" / "reports" / "final_group_standing_auto_report.md"
DIFF_REPORT = ROOT / "outputs" / "reports" / "final_candidate_v1_vs_v2_auto_diff.md"
DIFF_CSV = ROOT / "outputs" / "predictions" / "final_candidate_v1_vs_v2_auto_diff.csv"
FINAL_CANDIDATE_REPORT = ROOT / "outputs" / "reports" / "final_candidate_v2_auto_science_report.md"

POLICY_REPORT = ROOT / "outputs" / "reports" / "final_group_score_auto_policy.md"
SEED_REPORT = ROOT / "outputs" / "reports" / "auto_consensus_seed_stability_report.md"
CANDIDATE_DIR = ROOT / "outputs" / "final_candidate_v2_auto_science"

REQUIRED_CANDIDATE_FILES = [
    "final_group_score_predictions_auto.csv",
    "final_group_standing_predictions_auto.csv",
    "final_last8_predictions_auto.csv",
    "final_submission_pack_auto.csv",
    "auto_score_candidates.csv",
    "final_group_score_auto_policy.md",
    "final_group_standing_auto_report.md",
    "auto_consensus_seed_stability_report.md",
    "README.md",
]


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def score_sort_key(team: str) -> str:
    return team


def compute_group_standings(scores: pd.DataFrame, template: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_final_scores(scores, template)
    rows: list[dict[str, object]] = []
    table_rows: list[dict[str, object]] = []
    template_teams = {
        group_name: sorted(set(group["team_a"]) | set(group["team_b"]))
        for group_name, group in template.groupby("group", sort=True)
    }

    for group_name, group_scores in scores.groupby("group", sort=True):
        teams = template_teams[group_name]
        stats = {
            team: {"points": 0, "goals_for": 0, "goals_against": 0, "goal_difference": 0}
            for team in teams
        }
        for _, match in group_scores.iterrows():
            goals_a, goals_b = parse_score(match["final_recommended_score"])
            team_a = match["team_a"]
            team_b = match["team_b"]
            stats[team_a]["goals_for"] += goals_a
            stats[team_a]["goals_against"] += goals_b
            stats[team_b]["goals_for"] += goals_b
            stats[team_b]["goals_against"] += goals_a
            if goals_a > goals_b:
                stats[team_a]["points"] += 3
            elif goals_a < goals_b:
                stats[team_b]["points"] += 3
            else:
                stats[team_a]["points"] += 1
                stats[team_b]["points"] += 1

        for team_stats in stats.values():
            team_stats["goal_difference"] = team_stats["goals_for"] - team_stats["goals_against"]

        ordered = sorted(
            stats.items(),
            key=lambda item: (
                -item[1]["points"],
                -item[1]["goal_difference"],
                -item[1]["goals_for"],
                item[1]["goals_against"],
                score_sort_key(item[0]),
            ),
        )
        unresolved = []
        tie_cols = ["points", "goal_difference", "goals_for", "goals_against"]
        seen: dict[tuple[int, int, int, int], list[str]] = {}
        for team, team_stats in ordered:
            key = tuple(int(team_stats[col]) for col in tie_cols)
            seen.setdefault(key, []).append(team)
        for tied_teams in seen.values():
            if len(tied_teams) > 1:
                unresolved.append("/".join(sorted(tied_teams)))

        ranks = [team for team, _ in ordered]
        notes = (
            "deterministic_fallback_alphabetical_after_points_gd_gf_ga: " + "; ".join(unresolved)
            if unresolved
            else "none"
        )
        rows.append(
            {
                "group": group_name,
                "rank_1": ranks[0],
                "rank_2": ranks[1],
                "rank_3": ranks[2],
                "rank_4": ranks[3],
                "confidence_if_available": "not_available",
                "notes": notes,
            }
        )
        for rank, (team, team_stats) in enumerate(ordered, start=1):
            table_rows.append({"group": group_name, "rank": rank, "team": team, **team_stats})

    standings = pd.DataFrame(rows)
    details = pd.DataFrame(table_rows)
    return standings, details


def validate_standings(standings: pd.DataFrame, scores: pd.DataFrame, template: pd.DataFrame) -> None:
    if len(standings) != 12:
        raise ValueError(f"Auto standings must have exactly 12 groups; observed {len(standings)}.")
    if set(standings["group"]) != set("ABCDEFGHIJKL"):
        raise ValueError("Auto standings must include groups A-L exactly.")
    template_teams = {
        group_name: sorted(set(group["team_a"]) | set(group["team_b"]))
        for group_name, group in template.groupby("group", sort=True)
    }
    for _, row in standings.iterrows():
        ranked = [row["rank_1"], row["rank_2"], row["rank_3"], row["rank_4"]]
        expected = template_teams[row["group"]]
        if sorted(ranked) != expected:
            raise ValueError(f"Group {row['group']} standings do not contain template teams exactly once.")
    score_teams = {
        group_name: sorted(set(group["team_a"]) | set(group["team_b"]))
        for group_name, group in scores.groupby("group", sort=True)
    }
    for group_name, teams in template_teams.items():
        if score_teams[group_name] != teams:
            raise ValueError(f"Group {group_name} score teams do not match template teams.")


def write_standing_report(standings: pd.DataFrame, details: pd.DataFrame) -> None:
    fallback_count = int(standings["notes"].ne("none").sum())
    lines = [
        "# Final Group Standing Auto Report",
        "",
        "- Source: `outputs/predictions/final_group_score_predictions_auto.csv`.",
        "- Standings are computed directly from the 72 selected score predictions.",
        "- Rules: 3 points win, 1 point draw, 0 loss, then goal difference, goals scored, goals against.",
        "- Remaining exact ties use deterministic alphabetical fallback and are reported in `notes`.",
        f"- Groups with fallback tie-breaks: **{fallback_count}**",
        "",
        "## Final Standings",
        "",
        "| Group | 1 | 2 | 3 | 4 | Notes |",
        "|---|---|---|---|---|---|",
    ]
    for _, row in standings.iterrows():
        lines.append(
            f"| {row['group']} | {row['rank_1']} | {row['rank_2']} | {row['rank_3']} | {row['rank_4']} | {row['notes']} |"
        )
    lines.extend(["", "## Table Details", "", "| Group | Rank | Team | Pts | GD | GF | GA |", "|---|--:|---|--:|--:|--:|--:|"])
    for _, row in details.iterrows():
        lines.append(
            f"| {row['group']} | {row['rank']} | {row['team']} | {row['points']} | "
            f"{row['goal_difference']} | {row['goals_for']} | {row['goals_against']} |"
        )
    STANDING_REPORT.parent.mkdir(parents=True, exist_ok=True)
    STANDING_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_pack(scores: pd.DataFrame, standings: pd.DataFrame, last8: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in scores.iterrows():
        rows.append({"section": "group_score", **row.to_dict()})
    for _, row in standings.iterrows():
        rows.append({"section": "group_standing", **row.to_dict()})
    for _, row in last8.iterrows():
        rows.append({"section": "last8", **row.to_dict()})
    return pd.DataFrame(rows)


def write_diff(scores: pd.DataFrame, standings: pd.DataFrame, last8: pd.DataFrame) -> dict[str, int | bool]:
    v1_scores = pd.read_csv(V1_SCORES)
    v1_standings = pd.read_csv(V1_STANDINGS)
    v1_last8 = pd.read_csv(V1_LAST8)

    merged_scores = v1_scores.merge(
        scores,
        on=["match_number", "group", "team_a", "team_b"],
        suffixes=("_v1", "_auto"),
    )
    score_changes = merged_scores["final_recommended_score_v1"].ne(merged_scores["final_recommended_score_auto"])
    manual_resolved = int(merged_scores["manual_review_flag"].map(str).str.lower().eq("true").sum())
    ev_accepted = int(scores["auto_policy_decision"].eq("ev_override_accepted").sum())
    ev_rejected = int(
        (
            scores["safe_score"].ne(scores["ev_score"])
            & scores["auto_policy_decision"].ne("ev_override_accepted")
        ).sum()
    )
    safe_kept = int(scores["final_recommended_score"].eq(scores["safe_score"]).sum())

    v1_order = v1_standings.set_index("group")[["rank_1", "rank_2", "rank_3", "rank_4"]]
    auto_order = standings.set_index("group")[["rank_1", "rank_2", "rank_3", "rank_4"]]
    changed_groups = [group for group in v1_order.index if not v1_order.loc[group].equals(auto_order.loc[group])]
    last8_changed = not v1_last8.equals(last8)

    diff_rows = []
    for _, row in merged_scores.iterrows():
        diff_rows.append(
            {
                "change_type": "score",
                "match_number": row["match_number"],
                "group": row["group"],
                "team_a": row["team_a"],
                "team_b": row["team_b"],
                "v1": row["final_recommended_score_v1"],
                "auto": row["final_recommended_score_auto"],
                "changed": row["final_recommended_score_v1"] != row["final_recommended_score_auto"],
                "notes": row["reason_auto"],
            }
        )
    for group in v1_order.index:
        diff_rows.append(
            {
                "change_type": "group_standing",
                "match_number": "",
                "group": group,
                "team_a": "",
                "team_b": "",
                "v1": " > ".join(v1_order.loc[group].astype(str).tolist()),
                "auto": " > ".join(auto_order.loc[group].astype(str).tolist()),
                "changed": group in changed_groups,
                "notes": "",
            }
        )
    diff_rows.append(
        {
            "change_type": "last8",
            "match_number": "",
            "group": "",
            "team_a": "",
            "team_b": "",
            "v1": rel(V1_LAST8),
            "auto": rel(AUTO_LAST8),
            "changed": last8_changed,
            "notes": "copied unchanged from v1" if not last8_changed else "last8 changed",
        }
    )
    diff = pd.DataFrame(diff_rows)
    DIFF_CSV.parent.mkdir(parents=True, exist_ok=True)
    diff.to_csv(DIFF_CSV, index=False)

    lines = [
        "# Final Candidate v1 vs v2 Auto Diff",
        "",
        f"- Score changes: **{int(score_changes.sum())}**",
        f"- Group standing changes: **{len(changed_groups)}** groups (`{changed_groups or 'none'}`)",
        f"- Last-8 changed: **{last8_changed}**",
        f"- Former manual-review rows auto-resolved: **{manual_resolved}**",
        f"- EV overrides accepted: **{ev_accepted}**",
        f"- EV overrides rejected: **{ev_rejected}**",
        f"- Safe selections kept: **{safe_kept}**",
        f"- Diff CSV: `{rel(DIFF_CSV)}`",
        "",
        "## Changed Scores",
        "",
        "| # | Group | Match | v1 | auto | Reason |",
        "|---:|---|---|---|---|---|",
    ]
    changed_score_rows = merged_scores[score_changes]
    if changed_score_rows.empty:
        lines.append("| - | - | none | - | - | - |")
    else:
        for _, row in changed_score_rows.iterrows():
            lines.append(
                f"| {row['match_number']} | {row['group']} | {row['team_a']} vs {row['team_b']} | "
                f"{row['final_recommended_score_v1']} | {row['final_recommended_score_auto']} | {row['reason_auto']} |"
            )
    DIFF_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DIFF_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "score_changes": int(score_changes.sum()),
        "group_standing_changes": len(changed_groups),
        "last8_changed": last8_changed,
        "manual_resolved": manual_resolved,
        "ev_accepted": ev_accepted,
        "ev_rejected": ev_rejected,
        "safe_kept": safe_kept,
    }


def write_candidate_report(metrics: dict[str, int | bool]) -> None:
    lines = [
        "# Final Candidate v2 Auto Science Report",
        "",
        "- Candidate created: **yes**",
        "- Manual overrides used: **no**",
        f"- Manual-review rows auto-resolved: **{metrics['manual_resolved']}**",
        f"- EV overrides accepted: **{metrics['ev_accepted']}**",
        f"- EV overrides rejected: **{metrics['ev_rejected']}**",
        f"- Score changes vs v1: **{metrics['score_changes']}**",
        f"- Group standings changed vs v1: **{metrics['group_standing_changes']}**",
        f"- Last-8 changed: **{metrics['last8_changed']}**",
        "",
        "## Final Files",
        "",
    ]
    for name in REQUIRED_CANDIDATE_FILES:
        lines.append(f"- `{rel(CANDIDATE_DIR / name)}`")
    FINAL_CANDIDATE_REPORT.parent.mkdir(parents=True, exist_ok=True)
    FINAL_CANDIDATE_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(metrics: dict[str, int | bool]) -> None:
    lines = [
        "# Final Candidate v2 Auto Science",
        "",
        "This frozen candidate uses a fully automatic science-only score policy.",
        "",
        "- No manual overrides used.",
        f"- Manual-review flags were auto-resolved: **{metrics['manual_resolved']}** rows.",
        "- Score policy: source agreement, modal consensus, expected-points tie-break, safe-score final fallback, and gated EV override.",
        "- Thresholds: `min_ev_uplift_to_override_safe=0.25`, `max_allowed_variance_flag_for_ev=false`, `contrarian_ev_allowed_by_default=false`.",
        "- Sources used: safe score, EV score, most probable score, ensemble score, expected-points-max score.",
        "- Last-8 source: copied unchanged from `outputs/final_candidate_v1/final_last8_predictions.csv`.",
        "- Tests passed: pending verification after candidate generation.",
        "",
        "## Final Files To Submit",
        "",
        "- `final_group_score_predictions_auto.csv`",
        "- `final_group_standing_predictions_auto.csv`",
        "- `final_last8_predictions_auto.csv`",
        "- `final_submission_pack_auto.csv`",
    ]
    (CANDIDATE_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def freeze_candidate(metrics: dict[str, int | bool]) -> None:
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    copies = {
        AUTO_SCORES: CANDIDATE_DIR / "final_group_score_predictions_auto.csv",
        AUTO_STANDINGS: CANDIDATE_DIR / "final_group_standing_predictions_auto.csv",
        AUTO_LAST8: CANDIDATE_DIR / "final_last8_predictions_auto.csv",
        AUTO_PACK: CANDIDATE_DIR / "final_submission_pack_auto.csv",
        AUTO_CANDIDATES: CANDIDATE_DIR / "auto_score_candidates.csv",
        POLICY_REPORT: CANDIDATE_DIR / "final_group_score_auto_policy.md",
        STANDING_REPORT: CANDIDATE_DIR / "final_group_standing_auto_report.md",
        SEED_REPORT: CANDIDATE_DIR / "auto_consensus_seed_stability_report.md",
    }
    for src, dst in copies.items():
        shutil.copyfile(src, dst)
    write_readme(metrics)

    missing = [name for name in REQUIRED_CANDIDATE_FILES if not (CANDIDATE_DIR / name).exists()]
    if missing:
        raise ValueError(f"Frozen candidate is missing required files: {missing}")


def main() -> None:
    scores = pd.read_csv(AUTO_SCORES)
    template = pd.read_csv(TEMPLATE)
    last8 = pd.read_csv(AUTO_LAST8)
    standings, details = compute_group_standings(scores, template)
    validate_standings(standings, scores, template)

    AUTO_STANDINGS.parent.mkdir(parents=True, exist_ok=True)
    standings.to_csv(AUTO_STANDINGS, index=False)
    write_standing_report(standings, details)
    pack = build_pack(scores, standings, last8)
    pack.to_csv(AUTO_PACK, index=False)

    metrics = write_diff(scores, standings, last8)
    write_candidate_report(metrics)
    freeze_candidate(metrics)

    print(f"Wrote {rel(AUTO_STANDINGS)}")
    print(f"Wrote {rel(STANDING_REPORT)}")
    print(f"Wrote {rel(AUTO_PACK)}")
    print(f"Wrote {rel(DIFF_CSV)}")
    print(f"Wrote {rel(DIFF_REPORT)}")
    print(f"Wrote {rel(FINAL_CANDIDATE_REPORT)}")
    print(f"Wrote {rel(CANDIDATE_DIR)}")


if __name__ == "__main__":
    main()
