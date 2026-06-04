#!/usr/bin/env python3
"""Generate deterministic auto-consensus group score predictions."""

from __future__ import annotations

from pathlib import Path
import shutil

import pandas as pd

from src.evaluation.auto_consensus import (
    AutoPolicyConfig,
    collect_candidate_scores,
    seed_stability_rows,
    select_final_scores,
    validate_final_scores,
)


ROOT = Path(__file__).parent.parent
PREDICTIONS = ROOT / "outputs" / "predictions" / "fif8a_group_stage_predictions.csv"
ENSEMBLE = ROOT / "outputs" / "predictions" / "fif8a_group_stage_predictions_ensemble.csv"
DECISIONS = ROOT / "outputs" / "predictions" / "submission_decision_table.csv"
TEMPLATE = ROOT / "data" / "reference" / "fif8a_group_stage_template.csv"
FINAL_V1 = ROOT / "outputs" / "final_candidate_v1" / "final_group_score_predictions.csv"
LAST8_V1 = ROOT / "outputs" / "final_candidate_v1" / "final_last8_predictions.csv"

AUTO_CANDIDATES = ROOT / "outputs" / "predictions" / "auto_score_candidates.csv"
AUTO_SCORES = ROOT / "outputs" / "predictions" / "final_group_score_predictions_auto.csv"
AUTO_LAST8 = ROOT / "outputs" / "predictions" / "final_last8_predictions_auto.csv"
POLICY_REPORT = ROOT / "outputs" / "reports" / "final_group_score_auto_policy.md"
SEED_REPORT = ROOT / "outputs" / "reports" / "auto_consensus_seed_stability_report.md"

SEEDS = [101, 202, 303, 404, 505]
CONFIG = AutoPolicyConfig(
    min_ev_uplift_to_override_safe=0.25,
    max_allowed_variance_flag_for_ev=False,
    contrarian_ev_allowed_by_default=False,
)


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def write_seed_report(stability: pd.DataFrame) -> None:
    stable_count = int(stability["stable"].sum())
    lines = [
        "# Auto Consensus Seed Stability Report",
        "",
        f"- Fixed seeds requested: `{SEEDS}`",
        "- Existing group-score generation is deterministic from saved prediction CSVs; no stochastic score selector needed rerun.",
        "- Each seed therefore receives the same deterministic auto-policy score for every match.",
        f"- Stable matches: **{stable_count} / {len(stability)}**",
        f"- Maximum score entropy: **{stability['score_entropy'].max():.6f}**",
        "",
        "| # | Group | Match | Seed 101 | Seed 202 | Seed 303 | Seed 404 | Seed 505 | Modal | Support | Entropy | Stable |",
        "|---:|---|---|---|---|---|---|---|---|--:|--:|---|",
    ]
    for _, row in stability.iterrows():
        lines.append(
            f"| {row['match_number']} | {row['group']} | {row['team_a']} vs {row['team_b']} | "
            f"{row['seed_101_score']} | {row['seed_202_score']} | {row['seed_303_score']} | "
            f"{row['seed_404_score']} | {row['seed_505_score']} | {row['modal_score']} | "
            f"{row['modal_support_count']} | {row['score_entropy']:.6f} | {row['stable']} |"
        )
    SEED_REPORT.parent.mkdir(parents=True, exist_ok=True)
    SEED_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_policy_report(final_scores: pd.DataFrame, candidates: pd.DataFrame, skipped: list[str]) -> None:
    original_manual = int(final_scores["manual_review_flag_original"].sum())
    ev_accepted = int(final_scores["auto_policy_decision"].eq("ev_override_accepted").sum())
    safe_ev_disagreements = int(final_scores["safe_score"].ne(final_scores["ev_score"]).sum())
    ev_rejected = int(
        (
            final_scores["safe_score"].ne(final_scores["ev_score"])
            & final_scores["auto_policy_decision"].ne("ev_override_accepted")
        ).sum()
    )
    source_counts = candidates["candidate_source"].value_counts().sort_index()
    lines = [
        "# Final Group Score Auto Policy",
        "",
        "This file documents the deterministic science-only score selector. No manual review override is used.",
        "",
        "## Thresholds",
        "",
        f"- `min_ev_uplift_to_override_safe`: **{CONFIG.min_ev_uplift_to_override_safe:.2f} expected points**",
        f"- `max_allowed_variance_flag_for_ev`: **{CONFIG.max_allowed_variance_flag_for_ev}**",
        f"- `contrarian_ev_allowed_by_default`: **{CONFIG.contrarian_ev_allowed_by_default}**",
        "",
        "## Selection Rule",
        "",
        "1. If all scientific sources agree, choose that score.",
        "2. Otherwise choose the modal score across candidate sources and deterministic seed views.",
        "3. If tied, choose the score with highest average expected FIF8A points.",
        "4. If still tied, choose `safe_score`.",
        "5. If the selected score is the EV score while safe and EV disagree, keep EV only when uplift exceeds the threshold and the row is neither high-variance nor contrarian.",
        "6. Otherwise choose `safe_score`; no row requires manual input.",
        "",
        "## Sources Used",
        "",
    ]
    for source, count in source_counts.items():
        lines.append(f"- `{source}`: **{int(count)}** candidate rows")
    lines.extend(
        [
            "",
            "## Source Availability",
            "",
            f"- Skipped source entries: **{len(skipped)}**",
        ]
    )
    if skipped:
        for item in skipped[:25]:
            lines.append(f"- {item}")
        if len(skipped) > 25:
            lines.append(f"- ... {len(skipped) - 25} more")
    else:
        lines.append("- All configured sources were available for all matches.")
    lines.extend(
        [
            "",
            "## Outcomes",
            "",
            f"- Final score rows: **{len(final_scores)}**",
            f"- Original manual-review rows auto-resolved: **{original_manual}**",
            f"- Safe-vs-EV disagreements: **{safe_ev_disagreements}**",
            f"- EV overrides accepted: **{ev_accepted}**",
            f"- EV overrides rejected: **{ev_rejected}**",
            f"- Manual review still required: **0**",
            "",
            f"- Candidate CSV: `{rel(AUTO_CANDIDATES)}`",
            f"- Auto score CSV: `{rel(AUTO_SCORES)}`",
            f"- Seed stability report: `{rel(SEED_REPORT)}`",
        ]
    )
    POLICY_REPORT.parent.mkdir(parents=True, exist_ok=True)
    POLICY_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    predictions = pd.read_csv(PREDICTIONS)
    decisions = pd.read_csv(DECISIONS)
    final_v1 = pd.read_csv(FINAL_V1)
    template = pd.read_csv(TEMPLATE)
    ensemble = pd.read_csv(ENSEMBLE) if ENSEMBLE.exists() else None

    candidates, skipped = collect_candidate_scores(predictions, decisions, ensemble)
    final_scores = select_final_scores(predictions, final_v1, decisions, candidates, config=CONFIG)
    validate_final_scores(final_scores, template)
    stability = seed_stability_rows(final_scores, SEEDS)

    AUTO_CANDIDATES.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(AUTO_CANDIDATES, index=False)
    final_scores.to_csv(AUTO_SCORES, index=False)
    shutil.copyfile(LAST8_V1, AUTO_LAST8)
    write_seed_report(stability)
    write_policy_report(final_scores, candidates, skipped)

    print(f"Wrote {rel(AUTO_CANDIDATES)}")
    print(f"Wrote {rel(AUTO_SCORES)}")
    print(f"Wrote {rel(AUTO_LAST8)} (copied unchanged from {rel(LAST8_V1)})")
    print(f"Wrote {rel(POLICY_REPORT)}")
    print(f"Wrote {rel(SEED_REPORT)}")


if __name__ == "__main__":
    main()
