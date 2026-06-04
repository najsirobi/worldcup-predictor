#!/usr/bin/env python3
"""Build historical squad features for controlled Phase 5 experiments."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.squad_features import (
    aggregate_squad_features,
    world_cup_2022_players,
    world_cup_database_players,
)

ROOT = Path(__file__).parent.parent
WCD = ROOT / "data" / "raw" / "kaggle" / "world_cup_database"
WC2022 = ROOT / "data" / "raw" / "kaggle" / "world_cup_2022_player_data"
OUT = ROOT / "data" / "interim" / "squad_features.parquet"
REPORT = ROOT / "outputs" / "reports" / "squad_features_report.md"


def main() -> None:
    frames = []
    if (WCD / "squads.csv").exists():
        frames.append(world_cup_database_players(str(WCD)))
    if (WC2022 / "player_playingtime.csv").exists():
        frames.append(world_cup_2022_players(str(WC2022)))
    if not frames:
        raise FileNotFoundError("No squad sources found.")

    players = pd.concat(frames, ignore_index=True)
    features = aggregate_squad_features(players)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUT, index=False)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT, "w") as handle:
        handle.write("# Squad Features Report\n\n")
        handle.write(f"- Output: `data/interim/squad_features.parquet`\n")
        handle.write(f"- Rows: **{len(features)}** tournament-team squads\n")
        handle.write(f"- Sources: {', '.join(sorted(players['source'].dropna().unique()))}\n")
        handle.write("- Market-value fields are missing for these sources and remain null; no fake zeros are inserted.\n")
        handle.write("- Star-attacker market-value features are not production-usable from these sources.\n\n")
        by_source = players.groupby("source").agg(players=("player", "count"), teams=("team", "nunique")).reset_index()
        handle.write("| Source | Player rows | Teams |\n|---|--:|--:|\n")
        for _, row in by_source.iterrows():
            handle.write(f"| {row['source']} | {row['players']} | {row['teams']} |\n")


if __name__ == "__main__":
    main()
