#!/usr/bin/env python3
"""Recompute live group tables from manual scores (Travel Mode, Task B).

Reads the manual score override plus the frozen final-candidate predictions and
writes the live group tables, played/remaining match lists and a markdown
report. Does NOT retrain, fetch APIs, or modify the baseline predictions.

Outputs:
    outputs/live/live_group_tables.csv
    outputs/live/live_group_tables.json
    outputs/live/played_matches.csv
    outputs/live/remaining_matches.csv
    outputs/reports/live_tournament_state_report.md
"""

from __future__ import annotations

import json
from pathlib import Path

from src.live.active_candidate import load_active_candidate
from src.live.scores_override import OVERRIDE_PATH, load_override
from src.live.tournament_state import (
    TIE_BREAK_NOTE,
    compute_group_tables,
    group_tables_to_records,
    split_played_remaining,
)
from src.live.scores_override import utc_now_iso

ROOT = Path(__file__).resolve().parents[1]
LIVE_DIR = ROOT / "outputs" / "live"
REPORTS_DIR = ROOT / "outputs" / "reports"


def write_report(played, remaining, tables, candidate, path: Path) -> None:
    n_played = len(played)
    n_remaining = len(remaining)
    lines = [
        "# Live Tournament State Report",
        "",
        f"_Generated: {utc_now_iso()}_",
        "",
        f"- Active candidate: **{candidate['name']}** (`{candidate['active_candidate_dir']}`)",
        f"- Matches played: **{n_played} / 72**",
        f"- Matches remaining: **{n_remaining}**",
        "",
        "## Tie-break limitation",
        "",
        TIE_BREAK_NOTE,
        "",
        "## Live group tables",
        "",
    ]
    for group, sub in tables.groupby("group"):
        lines.append(f"### Group {group}")
        lines.append("")
        lines.append("| # | Team | P | W | D | L | GF | GA | GD | Pts |")
        lines.append("|---|------|---|---|---|---|----|----|----|-----|")
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['rank']} | {r['team']} | {r['played']} | {r['won']} | "
                f"{r['drawn']} | {r['lost']} | {r['goals_for']} | {r['goals_against']} | "
                f"{r['goal_difference']:+d} | {r['points']} |"
            )
        lines.append("")

    if n_played:
        lines.append("## Played matches")
        lines.append("")
        lines.append("| # | Group | Result |")
        lines.append("|---|-------|--------|")
        for _, r in played.iterrows():
            lines.append(
                f"| {r['match_number']} | {r['group']} | {r['team_a']} "
                f"{int(r['team_a_goals'])}-{int(r['team_b_goals'])} {r['team_b']} |"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Validate the active candidate's prediction files exist (fail clearly if not).
    candidate = load_active_candidate().as_dict()

    scores = load_override(OVERRIDE_PATH)
    played, remaining = split_played_remaining(scores)
    tables = compute_group_tables(scores)

    tables.to_csv(LIVE_DIR / "live_group_tables.csv", index=False)
    played.to_csv(LIVE_DIR / "played_matches.csv", index=False)
    remaining.to_csv(LIVE_DIR / "remaining_matches.csv", index=False)

    payload = {
        "generated_at": utc_now_iso(),
        "active_candidate": candidate,
        "matches_played": int(len(played)),
        "matches_remaining": int(len(remaining)),
        "tie_break_note": TIE_BREAK_NOTE,
        "groups": group_tables_to_records(tables),
    }
    (LIVE_DIR / "live_group_tables.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    write_report(played, remaining, tables, candidate, REPORTS_DIR / "live_tournament_state_report.md")
    print(
        f"Live tournament state updated: {len(played)} played, {len(remaining)} remaining. "
        f"Wrote outputs/live/ tables + report."
    )


if __name__ == "__main__":
    main()
