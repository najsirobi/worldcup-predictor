#!/usr/bin/env python3
"""Build historical squad features (Phase 5C, Task C).

Aggregates the historical squad table into one row per (tournament_year, team)
using only WC2026-comparable features (age + position mix). Height / club-country
features stay null because no historical source provides them.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.historical_squad_features import (
    COMPARABLE_FEATURE_COLUMNS,
    UNAVAILABLE_FEATURE_COLUMNS,
    aggregate_historical_squad_features,
)

ROOT = Path(__file__).parent.parent
SQUADS = ROOT / "data" / "interim" / "historical_world_cup_squads.parquet"
OUT = ROOT / "data" / "interim" / "historical_squad_features.parquet"
REPORT = ROOT / "outputs" / "reports" / "historical_squad_features_report.md"


def main() -> None:
    if not SQUADS.exists():
        raise FileNotFoundError("Run build_historical_world_cup_squads.py first.")
    squads = pd.read_parquet(SQUADS)
    features = aggregate_historical_squad_features(squads)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUT, index=False)

    usable_age = features["squad_avg_age"].notna()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT, "w") as fh:
        fh.write("# Historical Squad Features Report\n\n")
        fh.write(f"- Output: `data/interim/historical_squad_features.parquet`\n")
        fh.write(f"- Rows (tournament-team): **{len(features)}**\n")
        fh.write(f"- Tournaments: {int(features['tournament_year'].min())}-{int(features['tournament_year'].max())}\n\n")
        fh.write("## Comparable (usable) features\n\n")
        fh.write("These are computable from BOTH historical sources and the WC2026 official squad PDF:\n\n")
        for c in COMPARABLE_FEATURE_COLUMNS:
            fh.write(f"- `{c}` (non-null {features[c].notna().mean():.3f})\n")
        fh.write("\n## Emitted-but-unavailable features (null historically)\n\n")
        fh.write("Present for WC2026 only; null here, so excluded from historical training:\n\n")
        for c in UNAVAILABLE_FEATURE_COLUMNS:
            fh.write(f"- `{c}` (non-null {features[c].notna().mean():.3f})\n")
        fh.write("\n## Age coverage by year\n\n")
        fh.write("| Year | Teams | avg squad age (mean over teams) | teams w/ age |\n|---|--:|--:|--:|\n")
        for yr, g in features.groupby("tournament_year"):
            fh.write(f"| {int(yr)} | {len(g)} | "
                     f"{g['squad_avg_age'].mean():.2f} | {int(g['squad_avg_age'].notna().sum())} |\n")
        fh.write("\n## Notes\n\n")
        fh.write("- 2022 (world_cup_2022_player_data) carries age and position; world_cup_database carries "
                 "age (from DOB) and position for 1930-2018.\n")
        fh.write("- Missing values remain null; no fake zeros.\n")
        fh.write(f"- Teams with usable squad age features: **{int(usable_age.sum())}** / {len(features)}.\n")

    print(f"Wrote {OUT} ({len(features)} rows) and {REPORT}")


if __name__ == "__main__":
    main()
