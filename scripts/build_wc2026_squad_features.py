#!/usr/bin/env python3
"""Build current-only WC2026 official squad features."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.wc2026_squad_features import (
    aggregate_wc2026_squad_features,
    apply_explicit_team_mapping,
    build_template_join_table,
)

ROOT = Path(__file__).parent.parent
SQUADS = ROOT / "data" / "interim" / "wc2026_official_squads.parquet"
TEMPLATE = ROOT / "data" / "reference" / "fif8a_group_stage_template.csv"
FEATURES_OUT = ROOT / "data" / "interim" / "wc2026_squad_features.parquet"
JOIN_REPORT = ROOT / "outputs" / "reports" / "wc2026_squad_template_join_report.md"
FEATURE_REPORT = ROOT / "outputs" / "reports" / "wc2026_squad_features_report.md"
MAPPING_CANDIDATES = ROOT / "data" / "reference" / "wc2026_team_name_map_candidates.csv"


SPECIAL_TEAMS = [
    "Qatar",
    "Switzerland",
    "Canada",
    "Bosnia and Herzegovina",
    "Iraq",
    "Norway",
    "Spain",
    "Saudi Arabia",
]


def write_join_report(join: pd.DataFrame, mapping: pd.DataFrame) -> None:
    JOIN_REPORT.parent.mkdir(parents=True, exist_ok=True)
    missing = join.loc[~join["has_official_squad_coverage"], "template_team_name"].tolist()
    with JOIN_REPORT.open("w", encoding="utf-8") as handle:
        handle.write("# WC2026 Squad Template Join Report\n\n")
        handle.write(f"- FIF8A teams checked: **{len(join)}**\n")
        handle.write(f"- Teams with official squad rows: **{int(join['has_official_squad_coverage'].sum())}**\n")
        handle.write(f"- Missing teams: `{missing or 'none'}`\n")
        handle.write("- Production join uses exact names plus explicit casefold-only mappings written to `data/reference/wc2026_team_name_map_candidates.csv`.\n")
        handle.write("- No fuzzy team mapping is used.\n\n")
        handle.write("## Explicit Mapping Evidence\n\n")
        if mapping.empty:
            handle.write("- No mapping candidates were needed.\n\n")
        else:
            handle.write("| Template team | FIFA PDF team | FIFA code | Method | Needs review | Evidence |\n")
            handle.write("|---|---|---|---|---:|---|\n")
            for _, row in mapping.iterrows():
                handle.write(
                    f"| {row['template_team_name']} | {row['fifa_pdf_team_name']} | {row['fifa_code']} | "
                    f"{row['match_method']} | {row['needs_review']} | {row['evidence']} |\n"
                )
            handle.write("\n")
        handle.write("## Coverage Table\n\n")
        handle.write("| Group | Team | Squad rows | FIFA code | Covered |\n|---|---|--:|---|---:|\n")
        for _, row in join.sort_values(["template_group", "template_team_name"]).iterrows():
            handle.write(
                f"| {row['template_group']} | {row['template_team_name']} | {row['squad_rows_found']} | "
                f"{row['fifa_codes']} | {row['has_official_squad_coverage']} |\n"
            )


def write_feature_report(features: pd.DataFrame, squads: pd.DataFrame) -> None:
    FEATURE_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with FEATURE_REPORT.open("w", encoding="utf-8") as handle:
        handle.write("# WC2026 Squad Features Report\n\n")
        handle.write(f"- Source squads: `{SQUADS.relative_to(ROOT)}`\n")
        handle.write(f"- Output: `{FEATURES_OUT.relative_to(ROOT)}`\n")
        handle.write(f"- Feature rows: **{len(features)}** teams\n")
        handle.write(f"- Player rows used: **{len(squads)}**\n")
        handle.write("- These features are current-only and are not historical backtest features.\n")
        handle.write("- The official PDF does not contain market values; no market-value or star-attacker value features are created here.\n")
        handle.write("- Club country codes are parsed only from explicit club suffixes such as `(ENG)` in the PDF.\n\n")
        handle.write("## Feature Columns\n\n")
        for column in features.columns:
            handle.write(f"- `{column}`\n")
        handle.write("\n## Special Inspection Teams\n\n")
        handle.write("| Team | Players | Avg age | Avg height | FW count | Domestic club share | Top5 Europe club share |\n")
        handle.write("|---|--:|--:|--:|--:|--:|--:|\n")
        for team in SPECIAL_TEAMS:
            subset = features.loc[features["team"] == team]
            if subset.empty:
                handle.write(f"| {team} | 0 |  |  |  |  |  |\n")
                continue
            row = subset.iloc[0]
            handle.write(
                f"| {team} | {row['squad_player_count']} | {row['squad_avg_age']:.2f} | "
                f"{row['squad_avg_height_cm']:.1f} | {row['squad_fw_count']} | "
                f"{row['squad_domestic_club_share']:.3f} | {row['squad_top5_europe_club_share']:.3f} |\n"
            )


def main() -> None:
    if not SQUADS.exists():
        raise FileNotFoundError(f"Missing parsed official squads: {SQUADS}")
    squads = pd.read_parquet(SQUADS)
    template = pd.read_csv(TEMPLATE)
    join, mapping = build_template_join_table(template, squads)
    MAPPING_CANDIDATES.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(MAPPING_CANDIDATES, index=False)
    mapped_squads = apply_explicit_team_mapping(squads, mapping)
    features = aggregate_wc2026_squad_features(mapped_squads)
    FEATURES_OUT.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(FEATURES_OUT, index=False)
    write_join_report(join, mapping)
    write_feature_report(features, mapped_squads)

    if not bool(join["has_official_squad_coverage"].all()):
        raise SystemExit("Not every FIF8A team has official squad coverage; see join report.")


if __name__ == "__main__":
    main()
