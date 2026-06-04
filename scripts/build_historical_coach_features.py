#!/usr/bin/env python3
"""Build historical coach features (Phase 5C, Task E).

Reuses the strictly leak-free coach match-feature builder (every performance
field uses only matches *before* the current match date) and reshapes it into
the Phase 5C schema, including a tournament-start tenure proxy. Writes the
compatibility report.
"""

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
OUT = ROOT / "data" / "interim" / "historical_coach_features.parquet"
REPORT = ROOT / "outputs" / "reports" / "historical_coach_compatibility_report.md"

REQUIRED_COLUMNS = [
    "tournament_name",
    "team",
    "match_date",
    "coach_name",
    "coach_tenure_days_at_tournament_start",
    "coach_matches_before_match",
    "coach_winrate_before_match",
    "prior_world_cup_experience",
    "has_historical_coach_features",
]


def main() -> None:
    frames = []
    if (WCD / "manager_appearances.csv").exists():
        frames.append(world_cup_database_coach_appearances(str(WCD)))
    if (WCH / "matches_1930_2022.csv").exists():
        frames.append(world_cup_history_2022_coach_appearances(str(WCH)))
    if not frames:
        raise FileNotFoundError("No coach sources found.")

    appearances = pd.concat(frames, ignore_index=True)
    feats = build_coach_match_features(appearances)

    # Tournament-start tenure proxy: tenure at the team's first match of the tournament.
    feats = feats.copy()
    feats["match_date"] = pd.to_datetime(feats["match_date"])
    first_match = feats.groupby(["tournament_name", "team"])["match_date"].transform("min")
    start_tenure = (
        feats[feats["match_date"].eq(first_match)]
        .groupby(["tournament_name", "team"])["coach_tenure_days"].max()
        .rename("coach_tenure_days_at_tournament_start")
        .reset_index()
    )
    out = feats.merge(start_tenure, on=["tournament_name", "team"], how="left")
    out = out.rename(columns={"has_coach_features": "has_historical_coach_features"})
    out = out[REQUIRED_COLUMNS]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT, "w") as fh:
        fh.write("# Historical Coach Compatibility Report\n\n")
        fh.write(f"- Output: `data/interim/historical_coach_features.parquet`\n")
        fh.write(f"- Rows (team-match): **{len(out)}**\n")
        fh.write(f"- Coverage: 1930-2022 World Cup matches (managers from world_cup_database + 2022 from world_cup_history).\n\n")
        fh.write("## Leakage controls\n\n")
        fh.write("- `coach_matches_before_match`, `coach_winrate_before_match` and the tenure proxy use **only matches strictly before** the current match date.\n")
        fh.write("- `prior_world_cup_experience` flags whether the coach led the team in any earlier observed WC match.\n\n")
        fh.write("## WC2026 comparability\n\n")
        fh.write("- The WC2026 official PDF gives only the **coach name** (no historical match log), so for WC2026 we can derive `prior_world_cup_experience` and (with the historical manager log) a tenure/record-to-date, but the *richer* within-tournament performance fields have no forward-looking WC2026 equivalent before the tournament starts.\n")
        fh.write("- Coach performance history within WC matches is thin (a handful of prior matches per coach), so signal is weak. It is leak-free but not a strong predictor.\n\n")
        fh.write("## Field availability\n\n")
        fh.write("| Field | Non-null share |\n|---|--:|\n")
        for col in ["coach_name", "coach_tenure_days_at_tournament_start",
                    "coach_matches_before_match", "coach_winrate_before_match",
                    "prior_world_cup_experience"]:
            fh.write(f"| {col} | {out[col].notna().mean():.3f} |\n")
        recent = out[out["tournament_name"].isin([f"{y} FIFA World Cup" for y in (2010, 2014, 2018, 2022)])]
        fh.write("\n## Recent tournaments (priority years)\n\n")
        fh.write("| Tournament | Team-match rows | Coaches |\n|---|--:|--:|\n")
        for t, sub in recent.groupby("tournament_name"):
            fh.write(f"| {t} | {len(sub)} | {sub['coach_name'].nunique()} |\n")

    print(f"Wrote {OUT} ({len(out)} rows) and {REPORT}")


if __name__ == "__main__":
    main()
