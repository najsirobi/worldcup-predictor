#!/usr/bin/env python3
"""Build the historical World Cup squad table (Phase 5C, Task B).

Combines world_cup_database (1930-2018) and world_cup_2022_player_data (2022)
into a single standard-schema parquet. No invented values; missing fields stay
null. Validates plausible squad sizes and writes a coverage report.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ingest.historical_squads import build_historical_squads

ROOT = Path(__file__).parent.parent
WCD = ROOT / "data" / "raw" / "kaggle" / "world_cup_database"
WC2022 = ROOT / "data" / "raw" / "kaggle" / "world_cup_2022_player_data"
OUT = ROOT / "data" / "interim" / "historical_world_cup_squads.parquet"
REPORT = ROOT / "outputs" / "reports" / "historical_world_cup_squads_report.md"

PRIORITY_YEARS = [2010, 2014, 2018, 2022]


def main() -> None:
    squads = build_historical_squads(str(WCD), str(WC2022))

    # Validation: plausible squad sizes (no team < 11 or > 30 players).
    sizes = squads.groupby(["tournament_year", "team"])["player_name"].nunique()
    implausible = sizes[(sizes < 11) | (sizes > 30)]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    squads.to_parquet(OUT, index=False)

    by_year = (
        squads.groupby("tournament_year")
        .agg(
            teams=("team", "nunique"),
            players=("player_name", "count"),
            source=("source", lambda s: ", ".join(sorted(s.unique()))),
        )
        .reset_index()
    )
    per_team = squads.groupby(["tournament_year", "team"])["player_name"].nunique()
    size_stats = per_team.groupby("tournament_year").agg(["min", "median", "max"]).reset_index()
    by_year = by_year.merge(size_stats, on="tournament_year")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT, "w") as fh:
        fh.write("# Historical World Cup Squads Report\n\n")
        fh.write(f"- Output: `data/interim/historical_world_cup_squads.parquet`\n")
        fh.write(f"- Rows (players): **{len(squads)}**\n")
        fh.write(f"- Tournaments: **{squads['tournament_year'].nunique()}** ({int(squads['tournament_year'].min())}-{int(squads['tournament_year'].max())})\n")
        fh.write(f"- Sources: {', '.join(sorted(squads['source'].unique()))}\n")
        fh.write(f"- Priority years {PRIORITY_YEARS} all present: "
                 f"**{all(y in set(squads['tournament_year'].dropna().astype(int)) for y in PRIORITY_YEARS)}**\n\n")
        fh.write("## Field availability\n\n")
        fh.write("| Field | Non-null share |\n|---|--:|\n")
        for col in ["position", "date_of_birth", "age_at_tournament_start", "country_code",
                    "club", "club_country", "height_cm", "coach_name"]:
            fh.write(f"| {col} | {squads[col].notna().mean():.3f} |\n")
        fh.write("\n> club / club_country / height are unavailable in every historical source "
                 "and remain null (not zero). They are therefore **not comparable** to the WC2026 "
                 "official-PDF features that include them.\n\n")
        fh.write("## Coverage by tournament\n\n")
        fh.write("| Year | Teams | Players | Squad min | median | max | Source(s) |\n|---|--:|--:|--:|--:|--:|---|\n")
        for _, r in by_year.iterrows():
            fh.write(f"| {int(r['tournament_year'])} | {r['teams']} | {r['players']} | "
                     f"{int(r['min'])} | {int(r['median'])} | {int(r['max'])} | {r['source']} |\n")
        fh.write("\n## Validation\n\n")
        fh.write(f"- Implausible squad sizes (<11 or >30 players): **{len(implausible)}**\n")
        if len(implausible):
            for (yr, team), n in implausible.items():
                fh.write(f"  - {int(yr)} {team}: {n} players\n")
        fh.write("- Team mappings are explicit (raw == canonical; national-team names align with the match backbone).\n")
        fh.write("- No invented players, ages, clubs, coaches or market values; missing values are null.\n")

    print(f"Wrote {OUT} ({len(squads)} rows) and {REPORT}")


if __name__ == "__main__":
    main()
