#!/usr/bin/env python3
"""Apply a conservative current-only WC2026 squad overlay to baseline predictions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
BASELINE = ROOT / "outputs" / "predictions" / "fif8a_group_stage_predictions.csv"
SQUAD_FEATURES = ROOT / "data" / "interim" / "wc2026_squad_features.parquet"
TM_FEATURES = ROOT / "data" / "interim" / "wc2026_transfermarkt_team_features.parquet"
OUT = ROOT / "outputs" / "predictions" / "fif8a_group_stage_predictions_squad_overlay.csv"
REPORT = ROOT / "outputs" / "reports" / "wc2026_squad_overlay_report.md"
STATUS = ROOT / "outputs" / "reports" / "final_prediction_source_status.md"


CONTEXT_COLUMNS = [
    "squad_player_count",
    "squad_avg_age",
    "squad_avg_height_cm",
    "squad_fw_count",
    "squad_domestic_club_share",
    "squad_top5_europe_club_share",
    "squad_has_official_pdf_data",
]


def attach_team_features(predictions: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    feature_subset = features[["team", *CONTEXT_COLUMNS]].copy()
    out = predictions.merge(
        feature_subset.add_prefix("team_a_"),
        left_on="team_a",
        right_on="team_a_team",
        how="left",
    ).drop(columns=["team_a_team"])
    out = out.merge(
        feature_subset.add_prefix("team_b_"),
        left_on="team_b",
        right_on="team_b_team",
        how="left",
    ).drop(columns=["team_b_team"])
    return out


def main() -> None:
    predictions = pd.read_csv(BASELINE)
    features = pd.read_parquet(SQUAD_FEATURES)
    overlay = attach_team_features(predictions, features)

    for column in ["model_p_a_win", "model_p_draw", "model_p_b_win"]:
        overlay[f"{column}_baseline"] = overlay[column]
        overlay[f"{column}_squad_overlay"] = overlay[column]
    overlay["squad_overlay_applied"] = False
    overlay["squad_overlay_reason"] = (
        "No probability shift: official squad features are current-only, not historically backtestable; "
        "Transfermarkt exact-match market-value coverage is too low/uneven for production adjustments."
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    overlay.to_csv(OUT, index=False)

    tm_summary = ""
    if TM_FEATURES.exists():
        tm = pd.read_parquet(TM_FEATURES)
        tm_summary = (
            f"- Transfermarkt teams meeting 18-player threshold: **{int(tm['has_transfermarkt_enrichment'].sum())} / {tm['team'].nunique()}**\n"
            f"- Transfermarkt median team match coverage: **{tm['transfermarkt_match_coverage'].median():.3f}**\n"
        )
    else:
        tm_summary = "- Transfermarkt enrichment was not available.\n"

    max_probability_delta = 0.0
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("w", encoding="utf-8") as handle:
        handle.write("# WC2026 Squad Overlay Report\n\n")
        handle.write(f"- Baseline prediction file: `{BASELINE.relative_to(ROOT)}`\n")
        handle.write(f"- Overlay prediction file: `{OUT.relative_to(ROOT)}`\n")
        handle.write(f"- Rows: **{len(overlay)}**\n")
        handle.write(f"- Official squad feature teams: **{features['team'].nunique()} / 48**\n")
        handle.write(tm_summary)
        handle.write(f"- Probability adjustments applied: **0**\n")
        handle.write(f"- Maximum probability delta: **{max_probability_delta:.3f}**\n")
        handle.write("- Final Phase 4.5 recommendation file is unchanged.\n")
        handle.write("- Overlay file is for squad-context review, not a promoted prediction replacement.\n\n")
        handle.write("## Rationale\n\n")
        handle.write("- Official FIFA squad features are reliable for current WC2026 coverage but cannot be backtested against WC2018/WC2022 without comparable historical official squad features.\n")
        handle.write("- Market-value enrichment uses strict exact normalized name plus DOB matching only; coverage is too uneven to support robust star-attacker adjustments.\n")
        handle.write("- The overlay therefore attaches context columns and keeps all W/D/L probabilities and recommendations unchanged.\n")

    with STATUS.open("w", encoding="utf-8") as handle:
        handle.write("# Final Prediction Source Status\n\n")
        handle.write(f"- Official FIFA squad coverage: **{features['team'].nunique()} / 48 teams**\n")
        handle.write(tm_summary)
        handle.write(f"- Squad overlay created: **True** (`{OUT.relative_to(ROOT)}`)\n")
        handle.write("- Final recommendation file changed: **False**\n")
        handle.write(f"- Final recommended prediction file: `{BASELINE.relative_to(ROOT)}`\n")
        handle.write("- Reason: current-only squad data is useful context but not a validated replacement for Phase 4.5 predictions.\n")

    if len(overlay) != len(predictions):
        raise SystemExit("Overlay row count changed; expected to preserve baseline row count.")


if __name__ == "__main__":
    main()
