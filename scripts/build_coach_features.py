#!/usr/bin/env python3
"""Build historical coach features for controlled Phase 5 experiments."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.coach_features import (
    build_coach_match_features,
    world_cup_database_coach_appearances,
    world_cup_history_2022_coach_appearances,
)

ROOT = Path(__file__).parent.parent
WCD = ROOT / "data" / "raw" / "kaggle" / "world_cup_database"
WCH = ROOT / "data" / "raw" / "kaggle" / "world_cup_history"
OUT = ROOT / "data" / "interim" / "coach_features.parquet"
REPORT = ROOT / "outputs" / "reports" / "coach_features_report.md"


def main() -> None:
    frames = []
    if (WCD / "manager_appearances.csv").exists():
        frames.append(world_cup_database_coach_appearances(str(WCD)))
    if (WCH / "matches_1930_2022.csv").exists():
        frames.append(world_cup_history_2022_coach_appearances(str(WCH)))
    if not frames:
        raise FileNotFoundError("No coach sources found.")

    appearances = pd.concat(frames, ignore_index=True)
    features = build_coach_match_features(appearances)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUT, index=False)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT, "w") as handle:
        handle.write("# Coach Features Report\n\n")
        handle.write(f"- Output: `data/interim/coach_features.parquet`\n")
        handle.write(f"- Rows: **{len(features)}** team-match coach rows\n")
        handle.write("- Tenure and performance are observed within available World Cup match data only.\n")
        handle.write("- Every performance field uses matches strictly before the current match date.\n\n")
        handle.write("| Tournament | Team-match rows | Coaches |\n|---|--:|--:|\n")
        for tournament, sub in features.groupby("tournament_name"):
            handle.write(f"| {tournament} | {len(sub)} | {sub['coach_name'].nunique()} |\n")


if __name__ == "__main__":
    main()
