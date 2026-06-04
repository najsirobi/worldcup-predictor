#!/usr/bin/env python3
"""Parse the cached FIFA World Cup 2026 squad-list PDF."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ingest.fifa_squad_pdf import VALID_POSITIONS, parse_pdf

ROOT = Path(__file__).parent.parent
PDF = ROOT / "data" / "raw" / "fifa_official" / "squads" / "SquadLists-English.pdf"
CSV_OUT = ROOT / "data" / "interim" / "wc2026_official_squads.csv"
PARQUET_OUT = ROOT / "data" / "interim" / "wc2026_official_squads.parquet"
EXCEPTIONS_OUT = ROOT / "data" / "interim" / "wc2026_official_squad_parse_exceptions.csv"
REPORT = ROOT / "outputs" / "reports" / "wc2026_official_squad_parse_report.md"


def main() -> None:
    if not PDF.exists():
        raise FileNotFoundError(f"Missing official squad PDF: {PDF}")

    parsed = parse_pdf(PDF)
    players = parsed.players
    exceptions = parsed.exceptions

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    players.to_csv(CSV_OUT, index=False)
    players.to_parquet(PARQUET_OUT, index=False)
    exceptions.to_csv(EXCEPTIONS_OUT, index=False)

    required = ["team", "fifa_code", "position", "player_name"]
    missing_required_counts = {column: int(players[column].isna().sum() + players[column].eq("").sum()) for column in required}
    invalid_positions = sorted(set(players["position"].dropna()) - VALID_POSITIONS) if not players.empty else []
    per_team = players.groupby(["team", "fifa_code"], dropna=False).size().reset_index(name="players")
    dob_parsed = int(pd.to_datetime(players["date_of_birth"], errors="coerce").notna().sum()) if not players.empty else 0
    height_parsed = int(pd.to_numeric(players["height_cm"], errors="coerce").notna().sum()) if not players.empty else 0
    clubs_present = int(players["club"].notna().sum() - players["club"].eq("").sum()) if not players.empty else 0

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("w", encoding="utf-8") as handle:
        handle.write("# WC2026 Official Squad Parse Report\n\n")
        handle.write(f"- Source PDF: `{PDF.relative_to(ROOT)}`\n")
        handle.write(f"- CSV output: `{CSV_OUT.relative_to(ROOT)}`\n")
        handle.write(f"- Parquet output: `{PARQUET_OUT.relative_to(ROOT)}`\n")
        handle.write(f"- Parse exceptions: `{EXCEPTIONS_OUT.relative_to(ROOT)}`\n")
        handle.write(f"- Teams parsed: **{players['team'].nunique() if not players.empty else 0}**\n")
        handle.write(f"- Players parsed: **{len(players)}**\n")
        if not per_team.empty:
            handle.write(f"- Players per team min/median/max: **{per_team['players'].min()} / {per_team['players'].median():.0f} / {per_team['players'].max()}**\n")
        handle.write(f"- Rows with parsed DOB: **{dob_parsed}**\n")
        handle.write(f"- Rows with numeric height: **{height_parsed}**\n")
        handle.write(f"- Rows with club text: **{clubs_present}**\n")
        handle.write(f"- Invalid positions: `{invalid_positions or 'none'}`\n")
        handle.write(f"- Parse exception rows: **{len(exceptions)}**\n")
        handle.write(f"- Required field missing counts: `{missing_required_counts}`\n\n")
        handle.write("## Validation Notes\n\n")
        handle.write("- Expected teams: 48.\n")
        handle.write("- Expected total players: near 1,248.\n")
        handle.write("- Expected players per team: usually 23-26.\n")
        handle.write("- The parser uses `pypdf` layout text and records unparsed player-like rows as exceptions.\n")
        handle.write("- Shirt numbers come from the official table extraction; no squad membership is inferred.\n\n")
        handle.write("## Teams\n\n")
        handle.write("| Team | FIFA code | Players |\n|---|---:|--:|\n")
        for _, row in per_team.sort_values("team").iterrows():
            handle.write(f"| {row['team']} | {row['fifa_code']} | {row['players']} |\n")

    if len(players) == 0 or len(exceptions) > 0:
        raise SystemExit("Official FIFA squad PDF parse produced no rows or exceptions; see parse report.")


if __name__ == "__main__":
    main()
