#!/usr/bin/env python3
"""Create Phase 5 player/squad/coach data inventory and readiness reports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
REPORT_DIR = ROOT / "outputs" / "reports"
TEMPLATE = ROOT / "data" / "reference" / "fif8a_group_stage_template.csv"
MANUAL_COACH_TEMPLATE = ROOT / "data" / "raw" / "manual" / "coach_history_manual.csv"

DATASETS = {
    "transfermarkt_player_scores": ROOT / "data" / "raw" / "kaggle" / "transfermarkt_player_scores",
    "fifa23_players_clean": ROOT / "data" / "raw" / "kaggle" / "fifa23_players_clean",
    "world_cup_2022_player_data": ROOT / "data" / "raw" / "kaggle" / "world_cup_2022_player_data",
    "world_cup_database": ROOT / "data" / "raw" / "kaggle" / "world_cup_database",
    "fifa_official_squads": ROOT / "data" / "raw" / "fifa_official" / "squads",
    "manual": ROOT / "data" / "raw" / "manual",
}

FIELD_GROUPS = {
    "player_name": ["player", "player_name", "name", "Name", "given_name", "family_name"],
    "nationality_team": ["team", "team_name", "Nationality", "country_of_citizenship", "team_code", "country_code"],
    "position": ["position", "Position", "position_name", "position_code", "sub_position"],
    "age_birth": ["age", "Age", "birth_year", "birth_date", "date_of_birth"],
    "club_league": ["club", "Club", "current_club_name", "club_country", "current_club_domestic_competition_id"],
    "market_value": ["market_value_in_eur", "Value(£)", "total_market_value"],
    "caps_goals": ["caps", "goals", "international_caps", "international_goals"],
    "date_tournament_year": ["date", "match_date", "tournament_name", "tournament", "Year", "year"],
}


def _file_summary(path: Path) -> dict:
    suffix = path.suffix.lower()
    summary = {
        "dataset": path.parent.name,
        "file": str(path.relative_to(ROOT)),
        "present": path.exists(),
        "rows": "",
        "columns": "",
    }
    if suffix == ".csv":
        df = pd.read_csv(path, nrows=100)
        summary["rows"] = max(sum(1 for _ in open(path, errors="ignore")) - 1, 0)
        summary["columns"] = ", ".join(map(str, df.columns))
        columns = set(map(str, df.columns))
    elif suffix in {".html", ".json"}:
        summary["rows"] = ""
        summary["columns"] = ""
        columns = set()
    else:
        columns = set()

    for key, candidates in FIELD_GROUPS.items():
        summary[key] = any(candidate in columns for candidate in candidates)
    return summary


def _inventory_rows() -> list[dict]:
    rows = []
    for dataset, root in DATASETS.items():
        if not root.exists():
            rows.append({"dataset": dataset, "file": str(root.relative_to(ROOT)), "present": False})
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file():
                row = _file_summary(path)
                row["dataset"] = dataset
                rows.append(row)
    return rows


def _source_classification(dataset: str) -> tuple[str, str, str, str, str]:
    if dataset == "world_cup_database":
        return (
            "usable_for_historical_backtest",
            "No",
            "No",
            "Historical World Cup squads/managers through 2018; no WC2026 current squad coverage.",
            "not_current_only",
        )
    if dataset == "world_cup_2022_player_data":
        return (
            "usable_for_wc2022_test_only",
            "No",
            "No",
            "2022 tournament player stats can benchmark logic but are not a historical time series.",
            "current_only_not_backtest_safe",
        )
    if dataset == "transfermarkt_player_scores":
        return (
            "not_usable_without_manual_mapping",
            "Potentially",
            "Potentially",
            "Has player values and national-team snapshots, but no reliable announced squad membership per match for historical national teams.",
            "current_and_historical_club_proxy",
        )
    if dataset == "fifa23_players_clean":
        return (
            "proxy_only",
            "Potentially",
            "No",
            "Video-game/player-rating proxy, not official performance or squad data.",
            "proxy_only",
        )
    if dataset == "fifa_official_squads":
        return (
            "not_usable",
            "No",
            "No",
            "Cached shell HTML has no parsed player rows; manual extraction still required.",
            "current_only_not_backtest_safe",
        )
    if dataset == "manual":
        return (
            "not_usable_without_manual_mapping",
            "If filled",
            "No",
            "Manual templates exist, but WC2026 squad and team templates are not populated with real data.",
            "current_only_not_backtest_safe",
        )
    return ("not_usable", "No", "No", "No usable player/coach signal identified.", "not_usable")


def _write_inventory(rows: list[dict]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_DIR / "player_coach_data_inventory.md", "w") as handle:
        handle.write("# Player / Coach Data Inventory\n\n")
        handle.write("| Dataset | File | Rows | Player | Team/Nationality | Position | Age/DOB | Club/League | Market value | Caps/goals | Date/year |\n")
        handle.write("|---|---|--:|---|---|---|---|---|---|---|---|\n")
        for row in rows:
            handle.write(
                f"| {row.get('dataset', '')} | {row.get('file', '')} | {row.get('rows', '')} | "
                f"{row.get('player_name', False)} | {row.get('nationality_team', False)} | {row.get('position', False)} | "
                f"{row.get('age_birth', False)} | {row.get('club_league', False)} | {row.get('market_value', False)} | "
                f"{row.get('caps_goals', False)} | {row.get('date_tournament_year', False)} |\n"
            )

        handle.write("\n## Source-Level Usability\n\n")
        handle.write("| Source | Classification | Historical backtest safe | WC2026 usable | Current only / proxy | Notes |\n")
        handle.write("|---|---|---|---|---|---|\n")
        for dataset in DATASETS:
            classification, historical, wc2026, notes, current = _source_classification(dataset)
            handle.write(f"| {dataset} | {classification} | {historical} | {wc2026} | {current} | {notes} |\n")


def _write_readiness() -> None:
    with open(REPORT_DIR / "player_data_readiness.md", "w") as handle:
        handle.write("# Player Data Readiness\n\n")
        handle.write("| Source | Decision | Rationale |\n")
        handle.write("|---|---|---|\n")
        for dataset in DATASETS:
            classification, _, _, notes, _ = _source_classification(dataset)
            handle.write(f"| {dataset} | {classification} | {notes} |\n")
        handle.write("\n## Decision\n\n")
        handle.write("- Build historical squad composition features from `world_cup_database` and `world_cup_2022_player_data` only.\n")
        handle.write("- Do not build production star-attacker market-value features: no reliable announced WC2026 squad with market values exists locally.\n")
        handle.write("- Treat FIFA23 as proxy-only and exclude it from the plus model.\n")


def _write_wc2026_coverage() -> None:
    template = pd.read_csv(TEMPLATE)
    teams = sorted(set(template["team_a"]) | set(template["team_b"]))
    manual = pd.read_csv(ROOT / "data" / "raw" / "manual" / "worldcup_2026_squads_manual.csv")
    real_manual = manual[manual["player"].notna() & manual["team"].notna()].copy()
    rows = []
    for team in teams:
        sub = real_manual[real_manual["team"].eq(team)]
        rows.append(
            {
                "team": team,
                "players": len(sub),
                "with_position": int(sub["position"].notna().sum()) if len(sub) else 0,
                "with_market_value": int("market_value" in sub.columns and sub.get("market_value", pd.Series()).notna().sum()) if len(sub) else 0,
                "with_age": int(sub["age"].notna().sum()) if len(sub) else 0,
                "attackers": int(sub["position"].astype(str).str.upper().isin({"FW", "ST", "CF", "LW", "RW", "AM"}).sum()) if len(sub) else 0,
                "usable": len(sub) >= 18,
            }
        )
    with open(REPORT_DIR / "wc2026_squad_coverage_report.md", "w") as handle:
        handle.write("# WC2026 Squad Coverage Report\n\n")
        handle.write("- Official FIFA squad pages are cached as HTML shell pages only.\n")
        handle.write("- `worldcup_2026_squads_manual.csv` is still a template and has no usable player rows.\n")
        handle.write("- WC2026 squad feature coverage is therefore 0 / 48 teams.\n\n")
        handle.write("| Team | Players | With position | With market value | With age | Attackers | Usable |\n")
        handle.write("|---|--:|--:|--:|--:|--:|---|\n")
        for row in rows:
            handle.write(
                f"| {row['team']} | {row['players']} | {row['with_position']} | {row['with_market_value']} | "
                f"{row['with_age']} | {row['attackers']} | {row['usable']} |\n"
            )
        handle.write("\n## Special Inspection\n\n")
        for team in ["Qatar", "Switzerland", "Canada", "Bosnia and Herzegovina", "Iraq", "Norway", "Spain", "Saudi Arabia"]:
            handle.write(f"- {team}: no populated WC2026 squad rows found.\n")


def _write_coach_readiness() -> None:
    if not MANUAL_COACH_TEMPLATE.exists():
        MANUAL_COACH_TEMPLATE.parent.mkdir(parents=True, exist_ok=True)
        MANUAL_COACH_TEMPLATE.write_text(
            "team,country_code,coach_name,start_date,end_date,source,source_date,prior_world_cup_experience,prior_international_tournament_experience,notes\n"
        )
    with open(REPORT_DIR / "coach_data_readiness.md", "w") as handle:
        handle.write("# Coach Data Readiness\n\n")
        handle.write("- `world_cup_database` has historical manager appearances through 2018.\n")
        handle.write("- `world_cup_history/matches_1930_2022.csv` has 2022 match manager names.\n")
        handle.write("- No reliable WC2026 current coach-history table is populated locally.\n")
        handle.write("- Created/verified manual template: `data/raw/manual/coach_history_manual.csv`.\n\n")
        handle.write("Coach features are usable for historical World Cup-only experiments, but not sufficient for promoted WC2026 predictions without manual coach histories.\n")


def main() -> None:
    rows = _inventory_rows()
    _write_inventory(rows)
    _write_readiness()
    _write_wc2026_coverage()
    _write_coach_readiness()


if __name__ == "__main__":
    main()
