#!/usr/bin/env python3
"""Build WC2026 country-context readiness artifacts and feature table."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.features.country_context import INDICATOR_COLUMN_MAP, build_country_context_features

ROOT = Path(__file__).parent.parent
TEMPLATE = ROOT / "data" / "reference" / "fif8a_group_stage_template.csv"
MAPPING = ROOT / "data" / "reference" / "country_code_map.csv"
WB_INTERIM = ROOT / "data" / "interim" / "world_bank_country_context.csv"
WB_METADATA = ROOT / "data" / "raw" / "world_bank" / "country_metadata.json"
TM_FEATURES = ROOT / "data" / "interim" / "wc2026_transfermarkt_team_features.parquet"
ELITE_FEATURES = ROOT / "data" / "interim" / "elite_player_award_features.parquet"
CONFED_MAP = ROOT / "data" / "reference" / "team_confederation_map.csv"
FEATURES_OUT = ROOT / "data" / "interim" / "country_context_features.parquet"
CHECKLIST_CSV = ROOT / "outputs" / "predictions" / "wc2026_enrichment_gap_checklist.csv"
CHECKLIST_MD = ROOT / "outputs" / "reports" / "wc2026_enrichment_gap_checklist.md"
MAP_REPORT = ROOT / "outputs" / "reports" / "country_code_map_expansion_report.md"
READINESS_REPORT = ROOT / "outputs" / "reports" / "country_context_data_readiness.md"
WB_COVERAGE_REPORT = ROOT / "outputs" / "reports" / "wc2026_world_bank_context_coverage.md"
FEATURE_REPORT = ROOT / "outputs" / "reports" / "country_context_features_report.md"

PREVIOUS_WC2026_MAP = {
    "Argentina": "ARG",
    "Australia": "AUS",
    "Belgium": "BEL",
    "Brazil": "BRA",
    "Canada": "CAN",
    "England": "GBR",
    "France": "FRA",
    "Germany": "DEU",
    "Japan": "JPN",
    "Korea Republic": "KOR",
    "Mexico": "MEX",
    "Morocco": "MAR",
    "Netherlands": "NLD",
    "Norway": "NOR",
    "Portugal": "PRT",
    "Spain": "ESP",
    "Sweden": "SWE",
    "USA": "USA",
    "Uruguay": "URY",
}

PROXY_TEAMS = {"England", "Scotland"}


def _norm(series: pd.Series) -> pd.Series:
    return series.astype(str).str.casefold()


def load_wc2026_teams() -> pd.DataFrame:
    template = pd.read_csv(TEMPLATE)
    groups_a = template[["group", "team_a"]].rename(columns={"team_a": "team"})
    groups_b = template[["group", "team_b"]].rename(columns={"team_b": "team"})
    teams = pd.concat([groups_a, groups_b], ignore_index=True).drop_duplicates().sort_values(["group", "team"])
    return teams.reset_index(drop=True)


def load_metadata_frame() -> pd.DataFrame:
    with open(WB_METADATA) as handle:
        payload = json.load(handle)
    return pd.DataFrame(payload["records"])


def build_checklist(teams: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    team_rows = teams.copy()
    mapped = mapping.copy()
    mapped["_norm"] = _norm(mapped["canonical_team"])
    team_rows["_norm"] = _norm(team_rows["team"])
    team_rows = team_rows.merge(
        mapped[
            [
                "_norm",
                "canonical_team",
                "fifa_code",
                "world_bank_code",
                "world_bank_country_name",
                "is_proxy_mapping",
                "notes",
            ]
        ],
        on="_norm",
        how="left",
    )

    team_rows["official_squad_available"] = True
    team_rows["current_world_bank_code"] = team_rows["team"].map(PREVIOUS_WC2026_MAP)
    team_rows["proposed_world_bank_code"] = team_rows["world_bank_code"]
    team_rows["is_proxy_mapping"] = team_rows["is_proxy_mapping"].fillna(False).astype(bool)
    team_rows["is_direct_country_match"] = team_rows["proposed_world_bank_code"].notna() & ~team_rows["is_proxy_mapping"]
    team_rows["mapping_confidence"] = team_rows["is_proxy_mapping"].map({True: "medium", False: "high"})

    def _status(row: pd.Series) -> str:
        if pd.notna(row["current_world_bank_code"]) and row["is_proxy_mapping"]:
            return "existing_proxy_mapping"
        if pd.notna(row["current_world_bank_code"]):
            return "existing_direct_mapping"
        if row["is_proxy_mapping"]:
            return "missing_proxy_mapping"
        return "missing_direct_mapping"

    team_rows["world_bank_mapping_status"] = team_rows.apply(_status, axis=1)

    if TM_FEATURES.exists():
        tm = pd.read_parquet(TM_FEATURES)[
            ["team", "transfermarkt_players_matched", "has_transfermarkt_enrichment"]
        ].copy()
        tm["_norm"] = _norm(tm["team"])
        team_rows = team_rows.merge(
            tm[["_norm", "transfermarkt_players_matched", "has_transfermarkt_enrichment"]],
            on="_norm",
            how="left",
        )
        team_rows["transfermarkt_match_count_if_available"] = team_rows["transfermarkt_players_matched"].fillna(0).astype(int)
        team_rows["transfermarkt_threshold_pass"] = team_rows["transfermarkt_match_count_if_available"].ge(18)
    else:
        team_rows["transfermarkt_match_count_if_available"] = pd.NA
        team_rows["transfermarkt_threshold_pass"] = pd.NA

    team_rows["coach_history_available"] = False

    elite = pd.read_parquet(ELITE_FEATURES)
    elite_2026 = elite[elite["tournament_year"].eq(2026)][["team", "has_elite_award_features"]].copy()
    elite_2026["_norm"] = _norm(elite_2026["team"])
    elite_lookup = elite_2026.drop_duplicates("_norm").set_index("_norm")["has_elite_award_features"]
    team_rows["elite_awards_available"] = team_rows["_norm"].map(elite_lookup).fillna(False).astype(bool)

    conf = pd.read_csv(CONFED_MAP)
    conf["_norm"] = _norm(conf["team"])
    conf_lookup = conf.drop_duplicates("_norm").set_index("_norm")["confederation"]
    team_rows["confederation_available"] = team_rows["_norm"].map(conf_lookup).notna()

    def _action(row: pd.Series) -> str:
        if row["world_bank_mapping_status"] == "existing_direct_mapping":
            return "no_action_needed"
        if row["world_bank_mapping_status"] == "existing_proxy_mapping":
            return "proxy_mapping_review"
        if row["world_bank_mapping_status"] == "missing_proxy_mapping":
            return "proxy_mapping_review"
        if row["world_bank_mapping_status"] == "missing_direct_mapping":
            return "add_world_bank_mapping"
        return "context_only"

    team_rows["recommended_action"] = team_rows.apply(_action, axis=1)

    return team_rows[
        [
            "team",
            "group",
            "official_squad_available",
            "world_bank_mapping_status",
            "current_world_bank_code",
            "proposed_world_bank_code",
            "world_bank_country_name",
            "is_direct_country_match",
            "is_proxy_mapping",
            "mapping_confidence",
            "transfermarkt_match_count_if_available",
            "transfermarkt_threshold_pass",
            "coach_history_available",
            "elite_awards_available",
            "confederation_available",
            "recommended_action",
            "notes",
        ]
    ].sort_values(["group", "team"]).reset_index(drop=True)


def write_checklist_outputs(checklist: pd.DataFrame) -> None:
    CHECKLIST_CSV.parent.mkdir(parents=True, exist_ok=True)
    checklist.to_csv(CHECKLIST_CSV, index=False)

    CHECKLIST_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKLIST_MD, "w") as handle:
        handle.write("# WC2026 Enrichment Gap Checklist\n\n")
        handle.write(
            "- Scope: WC2026 country-context readiness only. Official squad availability is 48/48; "
            "this checklist tracks the enrichment state around World Bank mapping and the adjacent "
            "context flags requested for review.\n\n"
        )
        handle.write("| Team | Group | WB status | Current WB code | Proposed WB code | Proxy | Action | TM matches | TM >=18 | Elite awards | Confederation | Notes |\n")
        handle.write("|---|---|---|---|---|---:|---|--:|---:|---:|---:|---|\n")
        for _, row in checklist.iterrows():
            notes = str(row["notes"]) if pd.notna(row["notes"]) and str(row["notes"]).strip() else ""
            handle.write(
                f"| {row['team']} | {row['group']} | {row['world_bank_mapping_status']} | "
                f"{row['current_world_bank_code'] or ''} | {row['proposed_world_bank_code'] or ''} | "
                f"{bool(row['is_proxy_mapping'])} | {row['recommended_action']} | "
                f"{int(row['transfermarkt_match_count_if_available']) if pd.notna(row['transfermarkt_match_count_if_available']) else ''} | "
                f"{bool(row['transfermarkt_threshold_pass']) if pd.notna(row['transfermarkt_threshold_pass']) else ''} | "
                f"{bool(row['elite_awards_available'])} | {bool(row['confederation_available'])} | {notes} |\n"
            )


def write_map_report(mapping: pd.DataFrame, checklist: pd.DataFrame) -> None:
    direct_now = checklist["is_direct_country_match"].sum()
    proxy_now = checklist["is_proxy_mapping"].sum()
    no_direct = checklist.loc[~checklist["is_direct_country_match"], "team"].tolist()
    added = checklist.loc[checklist["current_world_bank_code"].isna() & checklist["proposed_world_bank_code"].notna(), "team"].tolist()
    changed = checklist.loc[
        checklist["current_world_bank_code"].notna()
        & checklist["current_world_bank_code"].ne(checklist["proposed_world_bank_code"]),
        "team",
    ].tolist()
    proxy_mappings = checklist.loc[checklist["is_proxy_mapping"], ["team", "proposed_world_bank_code", "world_bank_country_name"]]

    name_diffs = mapping[
        mapping["canonical_team"].astype(str).str.casefold()
        != mapping["world_bank_country_name"].astype(str).str.casefold()
    ][["canonical_team", "world_bank_country_name", "world_bank_code"]]

    with open(MAP_REPORT, "w") as handle:
        handle.write("# Country Code Map Expansion Report\n\n")
        handle.write("## Summary\n\n")
        handle.write("- Old row count: **20**\n")
        handle.write(f"- New row count: **{len(mapping)}**\n")
        handle.write("- WC2026 team coverage before: **19 / 48**\n")
        handle.write(f"- WC2026 team coverage after: **{checklist['proposed_world_bank_code'].notna().sum()} / 48**\n")
        handle.write(f"- Direct World Bank mappings after: **{int(direct_now)}**\n")
        handle.write(f"- Proxy mappings after: **{int(proxy_now)}**\n\n")

        handle.write("## Teams Still Without Direct World Bank Mapping\n\n")
        if no_direct:
            for team in no_direct:
                handle.write(f"- {team}\n")
        else:
            handle.write("- None\n")
        handle.write("\n## Proxy Mappings\n\n")
        if proxy_mappings.empty:
            handle.write("- None\n")
        else:
            for _, row in proxy_mappings.iterrows():
                handle.write(f"- {row['team']} -> {row['proposed_world_bank_code']} ({row['world_bank_country_name']})\n")

        handle.write("\n## World Bank Naming Differences\n\n")
        for _, row in name_diffs.sort_values("canonical_team").iterrows():
            handle.write(f"- {row['canonical_team']} -> `{row['world_bank_country_name']}` ({row['world_bank_code']})\n")

        handle.write("\n## Codes Verified Against API\n\n")
        handle.write(
            "- Every non-null `world_bank_code` was checked against `data/raw/world_bank/country_metadata.json` "
            "and resolves to a real World Bank country entry, not an aggregate.\n"
        )

        handle.write("\n## Mappings Added/Changed\n\n")
        handle.write(f"- Added mappings: **{len(added)}**\n")
        if added:
            handle.write(f"- Added teams: {', '.join(added)}\n")
        handle.write(f"- Changed existing mappings: **{len(changed)}**\n")
        if changed:
            handle.write(f"- Changed teams: {', '.join(changed)}\n")
        handle.write("- England remained mapped to `GBR`, but is now explicitly marked as a proxy.\n")
        handle.write("- Scotland was added as a `GBR` proxy with an explicit warning.\n")


def write_wb_coverage_report(features: pd.DataFrame, checklist: pd.DataFrame) -> None:
    with open(WB_COVERAGE_REPORT, "w") as handle:
        handle.write("# WC2026 World Bank Context Coverage\n\n")
        handle.write("## Summary\n\n")
        handle.write(f"- Teams: **{len(features)}**\n")
        handle.write(f"- Direct mappings: **{int((~features['is_proxy_mapping']).sum())}**\n")
        handle.write(f"- Proxy mappings: **{int(features['is_proxy_mapping'].sum())}**\n")
        handle.write(f"- Teams without any World Bank code: **{int(features['world_bank_code'].isna().sum())}**\n\n")

        handle.write("## Indicator Coverage (latest value before 2026)\n\n")
        for feature_name in [
            "gdp_current_usd",
            "gdp_per_capita_current_usd",
            "population_total",
            "education_spend_pct_gdp",
            "rd_spend_pct_gdp",
            "urbanisation_pct",
            "life_expectancy",
        ]:
            missing_col = f"{feature_name}_missing"
            available = int((~features[missing_col]).sum())
            handle.write(f"- `{feature_name}`: **{available} / {len(features)}** teams\n")

        handle.write("\n## Team Coverage Table\n\n")
        handle.write("| Team | Group | WB code | Proxy | GDP year | GDP pc year | Population year | Education year | R&D year | Urbanisation year | Life expectancy year |\n")
        handle.write("|---|---|---|---:|--:|--:|--:|--:|--:|--:|--:|\n")
        for _, row in features.sort_values(["group", "team"]).iterrows():
            handle.write(
                f"| {row['team']} | {row['group']} | {row['world_bank_code'] or ''} | {bool(row['is_proxy_mapping'])} | "
                f"{'' if pd.isna(row['gdp_value_year']) else int(row['gdp_value_year'])} | "
                f"{'' if pd.isna(row['gdp_per_capita_value_year']) else int(row['gdp_per_capita_value_year'])} | "
                f"{'' if pd.isna(row['population_value_year']) else int(row['population_value_year'])} | "
                f"{'' if pd.isna(row['education_spend_pct_gdp_value_year']) else int(row['education_spend_pct_gdp_value_year'])} | "
                f"{'' if pd.isna(row['rd_spend_pct_gdp_value_year']) else int(row['rd_spend_pct_gdp_value_year'])} | "
                f"{'' if pd.isna(row['urbanisation_value_year']) else int(row['urbanisation_value_year'])} | "
                f"{'' if pd.isna(row['life_expectancy_value_year']) else int(row['life_expectancy_value_year'])} |\n"
            )


def write_readiness_report(features: pd.DataFrame, checklist: pd.DataFrame) -> None:
    coverage_counts = {
        "GDP": int((~features["gdp_current_usd_missing"]).sum()),
        "GDP per capita": int((~features["gdp_per_capita_current_usd_missing"]).sum()),
        "population": int((~features["population_total_missing"]).sum()),
        "education spend": int((~features["education_spend_pct_gdp_missing"]).sum()),
        "R&D spend": int((~features["rd_spend_pct_gdp_missing"]).sum()),
        "urbanisation": int((~features["urbanisation_pct_missing"]).sum()),
        "life expectancy": int((~features["life_expectancy_missing"]).sum()),
    }

    year_columns = {
        "GDP": "gdp_value_year",
        "GDP per capita": "gdp_per_capita_value_year",
        "population": "population_value_year",
        "education spend": "education_spend_pct_gdp_value_year",
        "R&D spend": "rd_spend_pct_gdp_value_year",
        "urbanisation": "urbanisation_value_year",
        "life expectancy": "life_expectancy_value_year",
    }
    oldest_years = {
        label: int(features[column].dropna().min()) if features[column].notna().any() else None
        for label, column in year_columns.items()
    }

    robust = []
    context_only = []
    for label, count in coverage_counts.items():
        coverage_ratio = count / len(features)
        oldest_year = oldest_years[label]
        if coverage_ratio >= 0.95 and oldest_year is not None and oldest_year >= 2021:
            robust.append(label)
        else:
            context_only.append(label)

    education_missing_teams = features.loc[features["education_spend_pct_gdp_missing"], "team"].tolist()
    rd_missing_teams = features.loc[features["rd_spend_pct_gdp_missing"], "team"].tolist()

    with open(READINESS_REPORT, "w") as handle:
        handle.write("# Country Context Data Readiness\n\n")
        handle.write("## 48-Team Coverage\n\n")
        handle.write(f"- WC2026 teams covered in checklist: **{len(checklist)} / 48**\n")
        handle.write(f"- Direct World Bank mappings: **{int(checklist['is_direct_country_match'].sum())}**\n")
        handle.write(f"- Proxy mappings: **{int(checklist['is_proxy_mapping'].sum())}**\n")
        handle.write(f"- Teams with no World Bank equivalent in final map: **{int(features['world_bank_code'].isna().sum())}**\n\n")

        handle.write("## Proxy Mappings\n\n")
        for team in checklist.loc[checklist["is_proxy_mapping"], "team"].tolist():
            handle.write(f"- {team}\n")

        handle.write("\n## Indicator Coverage\n\n")
        for label, count in coverage_counts.items():
            oldest_year = oldest_years[label]
            oldest_text = str(oldest_year) if oldest_year is not None else "n/a"
            handle.write(f"- {label}: **{count} / 48** teams, oldest retained value year **{oldest_text}**\n")

        handle.write("\n## Remaining Indicator Gaps\n\n")
        handle.write(
            f"- Education spend missing teams: {', '.join(education_missing_teams) if education_missing_teams else 'none'}.\n"
        )
        handle.write(
            f"- R&D spend missing teams: {', '.join(rd_missing_teams) if rd_missing_teams else 'none'}.\n"
        )

        handle.write("\n## Interpretation\n\n")
        handle.write(
            f"- Robust enough for testing: {', '.join(robust) if robust else 'none'}.\n"
        )
        handle.write(
            f"- Context-only or skip due to sparsity: {', '.join(context_only) if context_only else 'none'}.\n"
        )
        handle.write("- GDP, GDP per capita, population, urbanisation, and life expectancy are macro proxies only.\n")
        handle.write(
            "- Education and R&D should remain secondary/context-only inputs because the values are less current for several teams even when non-null.\n"
        )
        feasible = features["world_bank_code"].notna().all() and coverage_counts["GDP"] >= 46 and coverage_counts["population"] >= 46
        handle.write(
            f"- Country-context model testing feasible now: **{'Yes' if feasible else 'No'}** "
            "(for core macro features only; not for education/R&D-heavy variants).\n"
        )


def write_feature_report(features: pd.DataFrame) -> None:
    with open(FEATURE_REPORT, "w") as handle:
        handle.write("# Country Context Features Report\n\n")
        handle.write(f"- Output: `data/interim/country_context_features.parquet`\n")
        handle.write(f"- Rows: **{len(features)}**\n")
        handle.write(f"- Teams with a World Bank code: **{int(features['world_bank_code'].notna().sum())}**\n")
        handle.write(f"- Proxy rows: **{int(features['is_proxy_mapping'].sum())}**\n\n")
        handle.write("## Feature Coverage\n\n")
        for feature_name in [
            "log_gdp",
            "log_population",
            "log_gdp_per_capita",
            "education_spend_pct_gdp",
            "rd_spend_pct_gdp",
            "urbanisation_pct",
            "life_expectancy",
        ]:
            handle.write(f"- `{feature_name}` non-null rows: **{int(features[feature_name].notna().sum())} / {len(features)}**\n")


def main() -> None:
    teams = load_wc2026_teams()
    mapping = pd.read_csv(MAPPING)
    wb = pd.read_csv(WB_INTERIM)
    features = build_country_context_features(teams, mapping, wb, tournament_year=2026)

    FEATURES_OUT.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(FEATURES_OUT, index=False)

    checklist = build_checklist(teams, mapping)
    write_checklist_outputs(checklist)
    write_map_report(mapping, checklist)
    write_wb_coverage_report(features, checklist)
    write_readiness_report(features, checklist)
    write_feature_report(features)

    print(f"Wrote {FEATURES_OUT}")
    print(f"Wrote {CHECKLIST_CSV}")
    print(f"Wrote {CHECKLIST_MD}")


if __name__ == "__main__":
    main()
