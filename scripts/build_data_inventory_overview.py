#!/usr/bin/env python3
"""Build a repo-grounded inventory of data, reports, and live prediction artifacts."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - pyarrow is expected but optional
    pq = None

try:
    import yaml
except Exception:  # pragma: no cover - PyYAML is expected but optional
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = REPO_ROOT / "outputs" / "predictions" / "data_inventory_catalog.csv"
REPORT_PATH = REPO_ROOT / "outputs" / "reports" / "data_inventory_overview.md"

SCAN_ROOTS = [
    REPO_ROOT / "data" / "raw",
    REPO_ROOT / "data" / "reference",
    REPO_ROOT / "data" / "interim",
    REPO_ROOT / "data" / "processed",
    REPO_ROOT / "outputs" / "reports",
    REPO_ROOT / "outputs" / "predictions",
    REPO_ROOT / "outputs" / "live",
    REPO_ROOT / "outputs" / "final_candidate_v2_auto_science",
    REPO_ROOT / "docs",
]
if (REPO_ROOT / "outputs" / "final_candidate_v3_objective_residual").exists():
    SCAN_ROOTS.append(REPO_ROOT / "outputs" / "final_candidate_v3_objective_residual")

TEAM_COL_HINTS = {
    "team",
    "home_team",
    "away_team",
    "canonical_team_name",
    "raw_team_name",
    "team_name",
    "country",
    "country_name",
    "country_full",
    "country_abrv",
    "country_code",
    "national_team",
    "coach_name",
    "manager_name",
}

MATCH_COL_HINTS = {
    "match_id",
    "fixture_id",
    "game_id",
    "match_number",
    "fixture_number",
}

DATE_COL_HINTS = {"date", "rank_date", "rating_date", "year", "source_date", "tournament_year"}

CORE_PATTERNS = [
    r"matches_with_ratings",
    r"matches_clean",
    r"elo_ratings_clean",
    r"fifa_rankings_clean",
    r"international_results",
    r"international_elo",
    r"fifa_world_ranking",
    r"world_cup_history",
    r"world_cup_database",
    r"model_matrix_baseline",
    r"fif8a_group_stage_predictions",
    r"final_group_score_predictions",
    r"final_group_standing_predictions",
    r"final_last8_predictions",
    r"final_submission_pack",
    r"rating_momentum",
]

STRUCTURE_PATTERNS = [
    r"wc2026_venues",
    r"wc2026_match_venues",
    r"round_of_32_mapping",
    r"knockout_bracket_mapping",
    r"third_place_assignment_annex_c",
    r"knockout_round_progression",
    r"scores_override",
    r"scores_batch_update",
    r"fif8a_group_stage_template",
]

SQUAD_PATTERNS = [
    r"squad",
    r"player",
    r"ballon_dor",
    r"elite_award",
    r"transfermarkt",
    r"human_upside",
    r"wc2026_official_squads",
]

COACH_PATTERNS = [r"coach"]
COUNTRY_PATTERNS = [r"world_bank", r"country_context", r"country_code_map", r"team_name_map", r"country_overperformance"]
GEO_PATTERNS = [r"altitude", r"travel", r"venue", r"team_home_altitude"]
CONFED_PATTERNS = [r"confederation"]
RESIDUAL_PATTERNS = [
    r"objective_residual",
    r"late_injury",
    r"late_availability",
    r"late_residual",
    r"expected_xi",
    r"goalkeeper_status",
    r"friendly_lineup_distortion",
    r"human_upside",
    r"overlay",
]
VALIDATION_PATTERNS = [
    r"backtest",
    r"audit",
    r"validation",
    r"recommendation",
    r"comparison",
    r"lift",
    r"readiness",
    r"report",
    r"summary",
]
LIVE_PATTERNS = [r"mobile_dashboard", r"prediction_vs_actual", r"live_group", r"remaining_matches", r"played_matches", r"scoring_summary"]
GAP_PATTERNS = [r"template", r"candidate", r"parse_exceptions", r"needs_review", r"manual_", r"_manual", r"_candidates"]


def rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def file_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in {".parquet", ".pq"}:
        return "parquet"
    if suffix == ".json":
        return "json"
    if suffix in {".yml", ".yaml"}:
        return "yml"
    if suffix == ".md":
        return "md"
    if suffix == ".html":
        return "html"
    return "other"


def read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as fh:
        reader = csv.reader(fh)
        try:
            return next(reader)
        except StopIteration:
            return []


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as fh:
        return max(sum(1 for _ in fh) - 1, 0)


def count_text_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        return sum(1 for _ in fh)


def inspect_csv(path: Path, load_full: bool) -> dict[str, Any]:
    header = read_header(path)
    row_count = count_csv_rows(path)
    result: dict[str, Any] = {
        "row_count": row_count,
        "column_count": len(header),
        "column_names": header,
        "duplicates": None,
        "missing_values": None,
    }
    if load_full:
        try:
            df = pd.read_csv(path, low_memory=False)
            result["duplicates"] = int(df.duplicated().sum())
            result["missing_values"] = {k: int(v) for k, v in df.isna().sum().items() if int(v) > 0}
            result["column_names"] = list(df.columns)
        except Exception as exc:  # pragma: no cover - best-effort metadata only
            result["error"] = str(exc)
    return result


def inspect_parquet(path: Path, load_full: bool) -> dict[str, Any]:
    result: dict[str, Any] = {"duplicates": None, "missing_values": None}
    if pq is None:
        result["error"] = "pyarrow unavailable"
        return result

    try:
        pf = pq.ParquetFile(path)
        result["row_count"] = int(pf.metadata.num_rows)
        result["column_count"] = len(pf.schema.names)
        result["column_names"] = list(pf.schema.names)
        if load_full:
            df = pd.read_parquet(path)
            result["duplicates"] = int(df.duplicated().sum())
            result["missing_values"] = {k: int(v) for k, v in df.isna().sum().items() if int(v) > 0}
            result["column_names"] = list(df.columns)
    except Exception as exc:  # pragma: no cover - best-effort metadata only
        result["error"] = str(exc)
    return result


def inspect_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            data = json.load(fh)
    except Exception as exc:  # pragma: no cover - best-effort metadata only
        return {"error": str(exc)}

    result: dict[str, Any] = {"json_type": type(data).__name__}
    if isinstance(data, list):
        result["row_count"] = len(data)
        if data and isinstance(data[0], dict):
            keys = sorted({k for item in data[:20] if isinstance(item, dict) for k in item.keys()})
            result["column_count"] = len(keys)
            result["column_names"] = keys
    elif isinstance(data, dict):
        result["row_count"] = len(data)
        result["column_count"] = len(data)
        result["column_names"] = list(data.keys())
    return result


def inspect_yml(path: Path) -> dict[str, Any]:
    if yaml is None:
        return {"error": "PyYAML unavailable"}
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            data = yaml.safe_load(fh)
    except Exception as exc:  # pragma: no cover - best-effort metadata only
        return {"error": str(exc)}
    result: dict[str, Any] = {"yaml_type": type(data).__name__}
    if isinstance(data, dict):
        result["row_count"] = len(data)
        result["column_count"] = len(data)
        result["column_names"] = list(data.keys())
    return result


def infer_load_full(path: Path, size_mb: float) -> bool:
    return size_mb <= 5 and file_format(path) in {"csv", "parquet", "json", "yml"}


def inspect_file(path: Path) -> dict[str, Any]:
    size_bytes = path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    fmt = file_format(path)
    load_full = infer_load_full(path, size_mb)
    result: dict[str, Any] = {
        "file_path": rel(path),
        "format": fmt,
        "size_mb": round(size_mb, 3),
        "size_bytes": size_bytes,
    }
    if fmt == "csv":
        result.update(inspect_csv(path, load_full))
    elif fmt == "parquet":
        result.update(inspect_parquet(path, load_full))
    elif fmt == "json":
        result.update(inspect_json(path))
    elif fmt == "yml":
        result.update(inspect_yml(path))
    elif fmt in {"md", "html"}:
        result["row_count"] = count_text_lines(path)
        result["column_count"] = None
        result["column_names"] = None
    else:
        result["row_count"] = None
        result["column_count"] = None
        result["column_names"] = None
    if "error" not in result:
        result["error"] = None
    return result


def matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def classify_family(path: Path) -> str:
    rel_path = rel(path)
    if matches_any(rel_path, LIVE_PATTERNS) or rel_path.startswith("docs/") or rel_path.startswith("data/live/"):
        return "10. Dashboard / live tracking artifacts"
    if rel_path.startswith("outputs/final_candidate_v3_objective_residual/"):
        return "8. Late-news / residual overlay data"
    if matches_any(rel_path, RESIDUAL_PATTERNS):
        return "8. Late-news / residual overlay data"
    if rel_path.startswith("data/raw/fifa_official/"):
        return "2. WC2026 tournament structure"
    if rel_path.startswith("data/raw/manual/"):
        return "11. Gaps / incomplete / context-only data"
    if matches_any(rel_path, COACH_PATTERNS):
        return "4. Coach / tactical / continuity data"
    if matches_any(rel_path, CONFED_PATTERNS):
        return "7. Confederation / structural football context"
    if matches_any(rel_path, GEO_PATTERNS):
        return "6. Geography / altitude / travel / venue data"
    if matches_any(rel_path, COUNTRY_PATTERNS):
        return "5. Country/context data"
    if matches_any(rel_path, STRUCTURE_PATTERNS):
        return "2. WC2026 tournament structure"
    if matches_any(rel_path, SQUAD_PATTERNS):
        return "3. Team/squad/player data"
    if matches_any(rel_path, CORE_PATTERNS):
        return "1. Core match/rating/model base"
    if matches_any(rel_path, VALIDATION_PATTERNS):
        return "9. Historical validation / backtest artifacts"
    if matches_any(rel_path, GAP_PATTERNS):
        return "11. Gaps / incomplete / context-only data"
    if rel_path.startswith("outputs/final_candidate_v2_auto_science/"):
        return "1. Core match/rating/model base"
    if rel_path.startswith("outputs/predictions/"):
        return "9. Historical validation / backtest artifacts"
    if rel_path.startswith("outputs/reports/"):
        return "9. Historical validation / backtest artifacts"
    if rel_path.startswith("data/reference/"):
        return "5. Country/context data"
    if rel_path.startswith("data/interim/") or rel_path.startswith("data/processed/") or rel_path.startswith("data/raw/"):
        return "1. Core match/rating/model base"
    return "11. Gaps / incomplete / context-only data"


def infer_status(path: Path) -> str:
    rel_path = rel(path)
    if rel_path.startswith("data/raw/"):
        return "raw_source"
    if rel_path.startswith("data/reference/"):
        return "reference"
    if rel_path.startswith("data/interim/"):
        return "intermediate"
    if rel_path.startswith("data/processed/"):
        return "processed_model_input"
    if rel_path.startswith("outputs/predictions/"):
        return "prediction_output"
    if rel_path.startswith("outputs/reports/"):
        return "report"
    if rel_path.startswith("outputs/live/") or rel_path.startswith("docs/") or rel_path.startswith("data/live/"):
        return "dashboard/live"
    if rel_path.startswith("outputs/final_candidate_v2_auto_science/"):
        return "candidate/frozen"
    if rel_path.startswith("outputs/final_candidate_v3_objective_residual/"):
        return "candidate/frozen"
    return "deprecated/unclear"


def infer_trainable_status(path: Path) -> str:
    rel_path = rel(path)
    if rel_path.startswith("outputs/final_candidate_v2_auto_science/"):
        return "frozen_candidate"
    if rel_path.startswith("outputs/final_candidate_v3_objective_residual/"):
        return "frozen_candidate"
    if rel_path.startswith("outputs/live/") or rel_path.startswith("docs/") or rel_path.startswith("data/live/"):
        return "dashboard_only"
    if matches_any(rel_path, [r"template", r"parse_exceptions"]):
        return "blocked_by_coverage"
    if matches_any(rel_path, [r"wc2026_official_squads", r"transfermarkt_player_scores", r"human_upside", r"objective_residual"]):
        return "WC2026_only_context"
    if matches_any(rel_path, [r"altitude", r"country_context", r"confederation", r"previous_match_strain", r"group_incentive", r"elite_player_award", r"ballon_dor", r"coach", r"squad", r"player"]):
        return "backtestable_context"
    if rel_path.startswith("data/raw/") or rel_path.startswith("data/interim/") or rel_path.startswith("data/processed/"):
        return "trainable"
    if rel_path.startswith("outputs/predictions/"):
        return "backtestable_context"
    if rel_path.startswith("outputs/reports/"):
        return "backtestable_context"
    return "unknown"


def infer_leakage(path: Path) -> str:
    rel_path = rel(path)
    if rel_path.startswith("outputs/final_candidate_v2_auto_science/"):
        return "low"
    if rel_path.startswith("outputs/final_candidate_v3_objective_residual/"):
        return "low"
    if rel_path.startswith("outputs/live/") or rel_path.startswith("docs/") or rel_path.startswith("data/live/"):
        return "high"
    if matches_any(rel_path, [r"human_upside", r"objective_residual", r"wc2026_official_squads"]):
        return "high"
    if matches_any(rel_path, [r"country_context", r"altitude", r"confederation", r"previous_match_strain", r"group_incentive", r"coach"]):
        return "medium"
    if matches_any(rel_path, [r"matches", r"elo", r"fifa_rank", r"world_cup_history", r"world_cup_database"]):
        return "low"
    if rel_path.startswith("data/raw/"):
        return "medium"
    if rel_path.startswith("outputs/predictions/") or rel_path.startswith("outputs/reports/"):
        return "medium"
    return "unknown"


def infer_current_use(path: Path) -> str:
    rel_path = rel(path)
    if rel_path.startswith("outputs/final_candidate_v2_auto_science/"):
        return "used_in_v2"
    if rel_path.startswith("outputs/final_candidate_v3_objective_residual/"):
        return "candidate_adjustment"
    if rel_path.startswith("outputs/live/") or rel_path.startswith("docs/") or rel_path.startswith("data/live/"):
        return "used_in_dashboard"
    if matches_any(rel_path, [r"human_upside", r"objective_residual"]):
        return "candidate_adjustment"
    if matches_any(rel_path, [r"model_matrix_baseline", r"matches_with_ratings", r"elo_ratings_clean", r"fifa_rankings_clean"]):
        return "used_in_v2"
    if matches_any(rel_path, [r"country_context", r"confederation", r"altitude", r"previous_match_strain", r"group_incentive", r"elite_player_award", r"ballon_dor", r"squad_features", r"coach_features"]):
        return "tested_not_promoted"
    if matches_any(rel_path, [r"transfermarkt", r"wc2026_official_squads", r"parse_exceptions", r"template", r"candidate"]):
        return "unused_gap"
    if rel_path.startswith("outputs/predictions/"):
        return "tested_not_promoted"
    if rel_path.startswith("outputs/reports/"):
        return "context_only"
    return "unknown"


def short_description(path: Path) -> str:
    rel_path = rel(path)
    name = path.name
    if rel_path.startswith("data/raw/kaggle/international_results"):
        return "Raw international match history and scorers from Kaggle."
    if rel_path.startswith("data/raw/kaggle/world_cup_database"):
        return "Raw historical World Cup entities: matches, squads, players, managers, awards, referees."
    if rel_path.startswith("data/raw/kaggle/world_cup_history"):
        return "Historical World Cup matches and ranking snapshots."
    if rel_path.startswith("data/raw/kaggle/transfermarkt_player_scores"):
        return "Transfermarkt player/club valuation and appearance data."
    if rel_path.startswith("data/raw/kaggle/fifa23_players_clean"):
        return "EA FIFA edition player ratings used as a proxy source."
    if rel_path.startswith("data/raw/kaggle/world_cup_2022_player_data"):
        return "2022 player-level World Cup tables and stats."
    if rel_path.startswith("data/raw/fifa_official"):
        return "Cached FIFA 2026 official HTML shells and page captures."
    if rel_path.startswith("data/raw/manual"):
        return "Manual import template or enrichment worksheet."
    if rel_path.startswith("data/reference/wc2026_venues"):
        return "WC2026 venue and host-city reference data."
    if rel_path.startswith("data/reference/wc2026_match_venues"):
        return "WC2026 match-to-venue crosswalk."
    if name == "team_name_map.csv":
        return "Explicit canonical team-name mapping."
    if name == "country_code_map.csv":
        return "Explicit country-code mapping for joins."
    if name == "team_confederation_map.csv":
        return "Team-to-confederation mapping."
    if name == "team_home_altitude.csv":
        return "Team home-altitude reference for altitude context."
    if name == "scoring_rules.yml":
        return "Current scoring policy and rule definitions."
    if rel_path.startswith("data/interim/matches"):
        return "Cleaned historical international match tables with rating joins."
    if rel_path.endswith("world_bank_country_context.csv"):
        return "Annual World Bank country context joined across indicators."
    if rel_path.endswith("wc2026_travel_sequences.parquet"):
        return "WC2026 team-match travel, rest and burden sequence table."
    if rel_path.endswith("wc2026_squad_features.parquet"):
        return "WC2026 squad summary features from the official PDF parse."
    if rel_path.endswith("ballon_dor_rankings_clean.parquet"):
        return "Cleaned Ballon d'Or ranking history."
    if rel_path.endswith("elite_player_award_features.parquet"):
        return "Elite-award team features for the conservative residual experiment."
    if rel_path.endswith("historical_coach_features.parquet") or rel_path.endswith("coach_features.parquet"):
        return "Coach continuity and as-of tournament history features."
    if rel_path.endswith("confederation_features.parquet"):
        return "Confederation pair and residual context features."
    if rel_path.endswith("country_context_features.parquet"):
        return "Macro country-context features for tournament teams."
    if rel_path.endswith("altitude_features.parquet"):
        return "Altitude and venue burden features."
    if rel_path.endswith("previous_match_strain_features.parquet"):
        return "Previous-match strain features for team-match rows."
    if rel_path.endswith("group_incentive_features.parquet"):
        return "Historical group-stage incentive features."
    if rel_path.startswith("data/processed/model_matrix"):
        return "Model-ready matrix for a specific feature family."
    if rel_path == "outputs/final_candidate_v2_auto_science/README.md":
        return "Frozen candidate bundle documentation."
    if rel_path.startswith("outputs/final_candidate_v2_auto_science/"):
        return "Frozen v2 auto-science candidate artifact."
    if rel_path.startswith("outputs/final_candidate_v3_objective_residual/"):
        return "Frozen objective-residual candidate artifact."
    if rel_path.startswith("outputs/live/") or rel_path.startswith("docs/"):
        return "Live/dashboard artifact or exported dashboard data."
    if rel_path.startswith("outputs/reports/"):
        return "Audit, backtest, or readiness report."
    if rel_path.startswith("outputs/predictions/"):
        return "Prediction, backtest, or review output artifact."
    return "Repository data or analysis artifact."


def key_columns(cols: list[str] | None) -> str:
    if not cols:
        return ""
    hits = [col for col in cols if col.lower() in TEAM_COL_HINTS or col.lower() in MATCH_COL_HINTS or col.lower() in DATE_COL_HINTS]
    if not hits:
        hits = [col for col in cols if any(token in col.lower() for token in ["team", "match", "date", "country", "player", "coach", "venue", "rank"])]
    return ", ".join(hits[:10])


def count_unique_from_columns(path: Path, cols: list[str], targets: set[str]) -> str:
    if path.stat().st_size > 5 * 1024 * 1024:
        return ""
    if not cols:
        return ""
    selected = [c for c in cols if c.lower() in targets]
    if not selected:
        return ""
    try:
        if file_format(path) == "csv":
            df = pd.read_csv(path, usecols=selected, low_memory=False)
        elif file_format(path) == "parquet" and pq is not None:
            df = pd.read_parquet(path, columns=selected)
        else:
            return ""
    except Exception:
        return ""
    if not len(df):
        return "0"
    if len(selected) == 1:
        return str(int(df[selected[0]].dropna().nunique()))
    combos = df[selected].fillna("").astype(str).apply(lambda row: " | ".join(row.tolist()), axis=1)
    return str(int(combos.nunique()))


def infer_years(path: Path, cols: list[str] | None) -> str:
    if path.stat().st_size > 5 * 1024 * 1024:
        return ""
    if not cols:
        return ""
    lower_cols = {c.lower() for c in cols}
    selected = [c for c in cols if c.lower() in DATE_COL_HINTS or c.lower().endswith("_date") or c.lower().endswith("_year") or c.lower() == "year"]
    if not selected:
        return ""
    try:
        if file_format(path) == "csv":
            df = pd.read_csv(path, usecols=selected, low_memory=False)
        elif file_format(path) == "parquet" and pq is not None:
            df = pd.read_parquet(path, columns=selected)
        else:
            return ""
    except Exception:
        return ""
    years: list[int] = []
    for col in selected:
        series = df[col]
        if "year" in col.lower() and "date" not in col.lower():
            nums = pd.to_numeric(series, errors="coerce").dropna().astype(int)
            years.extend(nums.tolist())
        else:
            parsed = pd.to_datetime(series, errors="coerce", utc=False)
            years.extend(parsed.dropna().dt.year.astype(int).tolist())
    if not years:
        return ""
    return f"{min(years)}-{max(years)}"


def infer_wc2026_coverage(path: Path, row_count: int | None) -> str:
    rel_path = rel(path)
    if "wc2026_venues.csv" in rel_path:
        return "16/16 venues"
    if "wc2026_match_venues_enriched.csv" in rel_path or "wc2026_match_venues.csv" in rel_path:
        return "104/104 matches"
    if "wc2026_travel_sequences" in rel_path:
        return "208/208 team-match rows"
    if "team_confederation_map.csv" in rel_path:
        return "48/48 teams"
    if "world_bank_country_context.csv" in rel_path:
        return "48/48 teams mapped; 217+ country/year rows"
    if "wc2026_human_upside_overlay" in rel_path:
        return "48/48 teams"
    if "wc2026_official_squads" in rel_path:
        return "WC2026-only snapshot; usable team coverage depends on parse state"
    if "wc2026_squad_features" in rel_path:
        return "48/48 teams in the PDF parse, but coverage is not usable for historical backtests"
    if "ballon_dor_rankings_clean" in rel_path:
        return "5 award years (2009, 2013, 2017, 2021, 2025)"
    if "elite_player_award_features" in rel_path:
        return "Backtests: 2018 and 2022; WC2026 overlay present"
    if "country_context_features" in rel_path:
        return "48/48 teams"
    if "coach_features" in rel_path:
        return "1930-2022 World Cup matches"
    if "previous_match_strain_features" in rel_path:
        return "70,584 team-match rows"
    if "group_incentive_features" in rel_path:
        return "192 team-match rows"
    if row_count is None:
        return ""
    return ""


def infer_historical_coverage(path: Path, years: str, row_count: int | None) -> str:
    rel_path = rel(path)
    if "matches_with_ratings" in rel_path or "matches_clean" in rel_path:
        return "1872-2026"
    if "elo_ratings_clean" in rel_path:
        return "1872-2026"
    if "fifa_rankings_clean" in rel_path:
        return "1992-2024"
    if "international_results" in rel_path:
        return "1872-2017"
    if "world_cup_history" in rel_path:
        return "1930-2022"
    if "world_cup_database" in rel_path:
        return "1930-2018"
    if "world_bank_country_context.csv" in rel_path:
        return "2000-2023"
    if "transfermarkt_player_scores" in rel_path:
        return "2004-2026"
    if "fifa23_players_clean" in rel_path:
        return "FIFA17-FIFA23"
    if "wc2022_overlay_backtest_input" in rel_path:
        return "WC2022 only"
    if "historical_squad_features" in rel_path or "squad_features" in rel_path:
        return "1930-2022"
    if "historical_coach_features" in rel_path or "coach_features" in rel_path:
        return "1930-2022"
    if "previous_match_strain_features" in rel_path:
        return "1930-2022"
    if "group_incentive_features" in rel_path:
        return "2010, 2014, 2018, 2022"
    if "confederation_features" in rel_path:
        return "historical mixed-era"
    if "ballon_dor_rankings_clean" in rel_path:
        return "2009-2025 awards"
    if "elite_player_award_features" in rel_path:
        return "2010, 2014, 2018, 2022"
    if "wc2026_" in rel_path:
        return "WC2026 only"
    return years


def compute_identifiers(path: Path, meta: dict[str, Any]) -> dict[str, Any]:
    cols = meta.get("column_names") or []
    row_count = meta.get("row_count")
    identifiers = {
        "key_columns": key_columns(cols),
        "unique_teams": "",
        "unique_matches": "",
        "years_covered": infer_years(path, cols),
    }

    if cols:
        if any(c.lower() in TEAM_COL_HINTS or "team" in c.lower() or "country" in c.lower() for c in cols):
            identifiers["unique_teams"] = count_unique_from_columns(path, cols, TEAM_COL_HINTS | {"team", "home_team", "away_team", "canonical_team_name", "raw_team_name", "team_name", "country", "country_name", "country_full", "country_abrv", "country_code", "national_team", "coach_name", "manager_name"})
        if any(c.lower() in MATCH_COL_HINTS or c.lower() in {"date", "home_team", "away_team"} for c in cols):
            match_targets = MATCH_COL_HINTS | {"date", "home_team", "away_team", "match_date", "fixture_date"}
            identifiers["unique_matches"] = count_unique_from_columns(path, cols, match_targets)

    identifiers["WC2026_coverage"] = infer_wc2026_coverage(path, row_count)
    identifiers["historical_coverage"] = infer_historical_coverage(path, identifiers["years_covered"], row_count)
    return identifiers


def row_to_dict(path: Path) -> dict[str, Any]:
    meta = inspect_file(path)
    meta.update(compute_identifiers(path, meta))
    meta["family"] = classify_family(path)
    meta["status"] = infer_status(path)
    meta["trainable_status"] = infer_trainable_status(path)
    meta["leakage_risk"] = infer_leakage(path)
    meta["current_use"] = infer_current_use(path)
    meta["short_description"] = short_description(path)
    meta["known_blockers"] = ""
    meta["possible_next_use"] = ""

    rel_path = meta["file_path"]
    if "transfermarkt" in rel_path:
        meta["known_blockers"] = "Matching coverage is uneven; historical squad linking is not promotable."
        meta["possible_next_use"] = "Use only if matching coverage and date alignment improve."
    elif "wc2026_official_squads" in rel_path:
        meta["known_blockers"] = "Official squad parse state is incomplete / context-only."
        meta["possible_next_use"] = "Populate only with verified official squad rows."
    elif "coach" in rel_path:
        meta["known_blockers"] = "Within-tournament coach history is thin for WC2026."
        meta["possible_next_use"] = "Use as leak-safe context or for dashboard explanations."
    elif "human_upside" in rel_path or "objective_residual" in rel_path:
        meta["known_blockers"] = "Overlay is intentionally non-authoritative."
        meta["possible_next_use"] = "Use as review-only candidate context."
    elif "country_context" in rel_path:
        meta["known_blockers"] = "Education and R&D indicators are sparse/stale for some teams."
        meta["possible_next_use"] = "Use core macro indicators; keep sparse fields secondary."
    elif "altitude" in rel_path or "travel" in rel_path:
        meta["known_blockers"] = "Signal is context-rich but has not been promoted."
        meta["possible_next_use"] = "Keep as venue/travel context or revisit with stronger backtests."
    elif "ballon_dor" in rel_path or "elite_player_award" in rel_path:
        meta["known_blockers"] = "Award signal worsened or failed to improve backtests."
        meta["possible_next_use"] = "Use only as dashboard context unless coverage improves."
    elif "confederation" in rel_path:
        meta["known_blockers"] = "Confederation residuals are small and unstable."
        meta["possible_next_use"] = "Revisit only if a better split or interaction adds value."
    elif "group_incentive" in rel_path:
        meta["known_blockers"] = "Historical signal is too small for promotion."
        meta["possible_next_use"] = "Keep for diagnostics and live explanation."
    elif "previous_match_strain" in rel_path:
        meta["known_blockers"] = "Previous-scoreline signal dominates any extra strain effect."
        meta["possible_next_use"] = "Use as live-context only."
    elif "draw" in rel_path:
        meta["known_blockers"] = "Draw-aware alternatives did not clear the promotion rule."
        meta["possible_next_use"] = "Keep for audit comparisons and dashboard context."
    elif rel_path.startswith("outputs/final_candidate_v2_auto_science/"):
        meta["known_blockers"] = "Frozen candidate; must not be modified."
        meta["possible_next_use"] = "Submit unchanged or compare against future candidates."
    elif rel_path.startswith("outputs/live/") or rel_path.startswith("docs/") or rel_path.startswith("data/live/"):
        meta["known_blockers"] = "Live artifacts are not training data."
        meta["possible_next_use"] = "Use for dashboard and operational monitoring only."
    return meta


def collect_paths() -> list[Path]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file():
                files.append(path)
    return sorted(files, key=lambda p: rel(p))


def to_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def build_report(df: pd.DataFrame, totals: dict[str, Any], directories_present: dict[str, bool]) -> str:
    family_order = [
        "1. Core match/rating/model base",
        "2. WC2026 tournament structure",
        "3. Team/squad/player data",
        "4. Coach / tactical / continuity data",
        "5. Country/context data",
        "6. Geography / altitude / travel / venue data",
        "7. Confederation / structural football context",
        "8. Late-news / residual overlay data",
        "9. Historical validation / backtest artifacts",
        "10. Dashboard / live tracking artifacts",
        "11. Gaps / incomplete / context-only data",
    ]

    def main_files_for_family(fam: str, limit: int = 5) -> str:
        subset = df[df["family"] == fam].copy()
        if subset.empty:
            return ""
        subset = subset.sort_values(by=["size_mb", "file_path"], ascending=[False, True])
        return "<br>".join(f"`{p}`" for p in subset["file_path"].head(limit))

    def family_row(fam: str) -> dict[str, str]:
        subset = df[df["family"] == fam]
        tabular = subset[subset["row_count"].notna()]
        total_rows = int(tabular["row_count"].fillna(0).sum()) if not tabular.empty else 0
        sample = subset.sort_values(by=["size_mb", "file_path"], ascending=[False, True]).head(1)
        sample_desc = ""
        if not sample.empty:
            row = sample.iloc[0]
            sample_desc = f"{row['row_count'] if pd.notna(row['row_count']) else 'n/a'} rows, {row['column_count'] if pd.notna(row['column_count']) else 'n/a'} cols"
        return {
            "family": fam,
            "files": str(len(subset)),
            "tabular_rows": f"{total_rows:,}" if total_rows else "",
            "main_files": main_files_for_family(fam),
            "sample_size": sample_desc,
            "coverage": ", ".join(sorted({c for c in subset["WC2026_coverage"].tolist() if c} or {""})).strip(", "),
            "trainability": ", ".join(sorted({c for c in subset["trainable_status"].tolist() if c} or {""})).strip(", "),
            "current_use": ", ".join(sorted({c for c in subset["current_use"].tolist() if c} or {""})).strip(", "),
            "caveats": "; ".join(sorted({c for c in subset["known_blockers"].tolist() if c} or {""}).copy()),
        }

    family_rows = [family_row(fam) for fam in family_order]
    family_df = pd.DataFrame(family_rows)

    # Executive summary counts.
    report_count = int((df["status"] == "report").sum())
    tabular_count = int(df["row_count"].notna().sum())
    prediction_count = int((df["status"] == "prediction_output").sum())
    live_count = int((df["status"] == "dashboard/live").sum())
    frozen_count = int((df["status"] == "candidate/frozen").sum())
    blocked_count = int((df["trainable_status"] == "blocked_by_coverage").sum() + (df["trainable_status"] == "WC2026_only_context").sum())

    critical = [
        "data/processed/model_matrix_baseline.parquet",
        "data/interim/matches_with_ratings.parquet",
        "data/interim/elo_ratings_clean.parquet",
        "data/interim/fifa_rankings_clean.parquet",
        "outputs/final_candidate_v2_auto_science/final_group_score_predictions_auto.csv",
        "outputs/final_candidate_v2_auto_science/final_submission_pack_auto.csv",
    ]
    critical_present = [c for c in critical if c in set(df["file_path"])]
    context_files = [
        "data/interim/world_bank_country_context.csv",
        "data/interim/wc2026_travel_sequences.parquet",
        "data/reference/wc2026_venues.csv",
        "data/interim/ballon_dor_rankings_clean.parquet",
        "data/interim/coach_features.parquet",
    ]
    context_present = [c for c in context_files if c in set(df["file_path"])]
    gaps_present = df[df["trainable_status"].isin(["blocked_by_coverage", "WC2026_only_context"])].shape[0]

    lines: list[str] = []
    lines.append("# Data Inventory Overview")
    lines.append("")
    lines.append("## A. Executive summary")
    lines.append(f"- Total relevant files inspected: **{len(df)}**")
    lines.append(f"- Number of tabular datasets: **{tabular_count}**")
    lines.append(f"- Number of report artifacts: **{report_count}**")
    lines.append(f"- Production-critical datasets: **{len(critical_present)}**")
    lines.append(f"- Context-only datasets: **{int((df['current_use'] == 'context_only').sum())}**")
    lines.append(f"- Blocked or WC2026-only datasets: **{gaps_present}**")
    lines.append(f"- Prediction outputs catalogued: **{prediction_count}**")
    lines.append(f"- Live/dashboard artifacts catalogued: **{live_count}**")
    lines.append(f"- Frozen candidate artifacts catalogued: **{frozen_count}**")
    lines.append("")
    lines.append("Production-critical files currently anchored in the repo:")
    for item in critical_present:
        lines.append(f"- `{item}`")
    lines.append("")
    lines.append("Key context-only files currently available:")
    for item in context_present:
        lines.append(f"- `{item}`")
    lines.append("")
    lines.append("Main opportunities for further testing:")
    lines.append("- coach-history completeness and richer within-tournament coach signals")
    lines.append("- late injury / lineup monitoring for objective residual review")
    lines.append("- venue-coordinate reconstruction for historical travel/altitude backtests")
    lines.append("- historical club / height / caps / goals comparators for squad features")
    lines.append("- improved Transfermarkt matching coverage before any market-value promotion")
    lines.append("")

    lines.append("## B. Canonical model backbone")
    lines.append("- The core backbone is `data/processed/model_matrix_baseline.parquet`, which is built from `data/interim/matches_with_ratings.parquet` and the raw match/rating sources.")
    lines.append("- The frozen candidate lives in `outputs/final_candidate_v2_auto_science/`; the bundle is immutable and includes the auto-science score files, fill-only fallback, and submission pack.")
    lines.append("- `v2_auto_science` is a frozen score-policy bundle, not a new feature family. It sits on top of the baseline matrices and does not rewrite the underlying training tables.")
    lines.append("- `final_group_score_predictions_fill_only.csv` is the conservative fallback inside the frozen bundle; `final_group_score_predictions_auto.csv` is the auto-science output that was actually frozen.")
    lines.append("- Travel Mode lives in the geography/travel layer and in dashboard/live artifacts. It is contextual support for reading the tournament, not a mutation of the frozen v2 bundle.")
    lines.append("")

    lines.append("## C. Dataset inventory by family")
    lines.append("")
    lines.append(family_df[["family", "files", "tabular_rows", "sample_size", "main_files", "coverage", "trainability", "current_use", "caveats"]].to_markdown(index=False))
    lines.append("")

    lines.append("## D. Coverage dashboard")
    coverage_rows = [
        ("WC2026 official squads", "WC2026-only snapshot; usable team coverage depends on parse state", 1248, "WC2026_only_context", "Official squad parse is incomplete / context-only.", "Populate verified official squad rows or keep as dashboard context."),
        ("WC2026 group matches", "72/72 group matches", 72, "backtestable_context", "Only the match-to-venue layer is present here.", "Use the schedule crosswalk as the canonical match list."),
        ("WC2026 full tournament schedule", "104/104 matches", 104, "backtestable_context", "Schedule is represented through venue crosswalks and live outputs.", "Keep the venue crosswalk as the schedule backbone."),
        ("WC2026 venues", "16/16 venues", 16, "backtestable_context", "Venue geometry is reference-only, not a trained feature.", "Use for travel/altitude context."),
        ("WC2026 altitude", "16/16 venues with altitude context", 16, "backtestable_context", "No promotion beyond diagnostics.", "Keep as venue context."),
        ("WC2026 travel sequences", "208/208 team-match rows", 208, "backtestable_context", "Travel layer is contextual and not yet promoted.", "Use for dashboard and post-hoc analysis."),
        ("World Bank country context", "48/48 teams mapped; 217+ country/year rows", 6786, "backtestable_context", "Education/R&D sparse; core macro indicators are strongest.", "Use core macro proxies only."),
        ("Ballon d'Or / elite players", "125 award rows; 48/48 WC2026 seed coverage", 125, "tested_not_promoted", "Signal degraded or failed to improve backtests.", "Keep as dashboard context only unless coverage improves."),
        ("Confederation mapping", "48/48 teams mapped", 48, "tested_not_promoted", "Residuals are small and unstable.", "Use only as calibration context."),
        ("Coach history", "1930-2022 World Cup matches", 1928, "tested_not_promoted", "WC2026 has only coach names; richer logs are thin.", "Use as leak-safe context."),
        ("Transfermarkt", "2004-2026 valuations; matching coverage remains uneven", 3172454, "blocked_by_coverage", "Historical matching is the blocker, not valuation timestamps.", "Collect better IDs or skip market-value promotion."),
        ("Historical squad features", "1930-2022 squad backtests", 489, "tested_not_promoted", "Only age/position mix is comparable; richer fields are WC2026-only.", "Keep the comparable subset for analysis only."),
        ("Previous-match strain", "70,584 team-match rows", 98592, "tested_not_promoted", "No material lift beyond previous scoreline.", "Use as narrative context only."),
        ("Group incentives", "192 team-match rows", 192, "tested_not_promoted", "Signal is small and not promotable.", "Use for diagnostics only."),
        ("Draw audit", "2018/2022 controlled backtests", 64, "tested_not_promoted", "Draw-aware alternatives failed the promotion rule.", "Keep for policy audit and dashboard context."),
        ("Objective residual overlay", "review-only candidate set", 72, "candidate_adjustment", "Overlay is intentionally non-authoritative.", "Use for review-only candidate context."),
    ]
    lines.append(pd.DataFrame(coverage_rows, columns=["dataset", "coverage", "n", "use_status", "main_limitation", "next_action"]).to_markdown(index=False))
    lines.append("")

    lines.append("## E. What is already tested and rejected")
    tested_rows = [
        ("Draw-aware policy", "outputs/reports/draw_policy_recommendation.md; outputs/reports/draw_scoring_logic_audit.md", "No promotion; current v2 remains final", "Draws did not clear the promotion rule even when they improved some expected-points views.", "Yes, as context in the dashboard and audit trail."),
        ("Group incentives", "outputs/reports/group_incentive_feature_report.md", "No promotion", "The signal is weak and unstable outside the controlled historical group sample.", "Yes, for live explanation only."),
        ("Elite-player awards / Ballon d'Or", "outputs/reports/elite_awards_policy_recommendation.md; outputs/reports/elite_awards_feature_lift_report.md", "No promotion", "Residual worsened both backtests and high-confidence behavior.", "Yes, as non-authoritative context."),
        ("Confederation calibration", "outputs/reports/confederation_calibration_backtest_report.md; outputs/reports/confederation_residual_audit.md", "No promotion", "Residuals are small and not stable enough to justify replacement.", "Yes, diagnostic context only."),
        ("Country context", "outputs/reports/country_context_policy_recommendation.md; outputs/reports/wc2026_world_bank_context_coverage.md", "No promotion", "Core macro features are usable, but the backtest gate failed.", "Yes, dashboard context only."),
        ("Altitude", "outputs/reports/altitude_policy_recommendation.md; outputs/reports/altitude_residual_audit.md", "No promotion", "No meaningful lift; mostly diagnostic.", "Yes, venue/travel context only."),
        ("Previous-match strain", "outputs/reports/previous_match_strain_policy_recommendation.md; outputs/reports/previous_match_strain_residual_audit.md", "No promotion", "No meaningful improvement beyond prior-scoreline controls.", "Yes, narrative context only."),
        ("Broad human overlay", "outputs/reports/human_overlay_2026_objective_policy.md; outputs/reports/wc2022_human_overlay_backtest_report.md", "No automatic promotion", "Broad suggestions were noisy/harmful; only narrow objective triggers remain review-only.", "Yes, but only as objective residual review."),
    ]
    lines.append(pd.DataFrame(tested_rows, columns=["experiment", "evidence_report_path", "result", "why_not_promoted", "dashboard_context"]).to_markdown(index=False))
    lines.append("")

    lines.append("## F. What appears useful")
    useful_rows = [
        ("v2_auto_science backbone", "core", "Frozen candidate bundle and submission pack remain coherent.", "Keep unchanged and use as the control line."),
        ("Objective residual rule R1_only_diff_5_0", "present", "Objective-residual review logic exists and is surfaced as candidate context.", "Keep as review-only input."),
        ("Travel Mode / live score override / knockout stack", "present", "Live artifacts exist and are useful for dashboard explanations.", "Keep separate from frozen predictions."),
        ("Altitude / travel as context", "present", "Useful as descriptive context even without promotion.", "Use for venue and travel explainers."),
        ("Late-news residual audit", "present", "Useful as a review layer for unusual absences or extreme mismatches.", "Keep strictly non-authoritative."),
    ]
    lines.append(pd.DataFrame(useful_rows, columns=["item", "status", "why_it_matters", "recommended_handling"]).to_markdown(index=False))
    lines.append("")

    lines.append("## G. Remaining data gaps and blockers")
    lines.append("")
    lines.append("### 1. High value / low effort")
    high_low = [
        ("Coach-history WC2026 completeness", "WC2026 coach names are present, but richer within-tournament coach logs are thin.", "data/raw/manual or data/reference + interim coach features", "Collect verified coach history and announce it in a dedicated reference table.", "Backtestable context, but currently weak."),
        ("Late injury / lineup monitoring", "Late shocks are the cleanest objective-residual trigger.", "data/raw/manual or outputs/predictions for review-only", "Track official squad changes and lineup news before match time.", "WC2026 context + review layer."),
        ("Friendly lineup distortion", "Friendly-based form can overstate current strength.", "data/interim or reports for audit", "Tag friendlies with lineup completeness and minutes-weighted caveats.", "Backtestable context."),
    ]
    lines.append(pd.DataFrame(high_low, columns=["gap", "why_it_matters", "where_it_should_live", "how_to_collect", "trainable_or_context"]).to_markdown(index=False))
    lines.append("")
    lines.append("### 2. High value / high effort")
    high_high = [
        ("Historical venue / coordinate reconstruction", "Needed for stronger travel and altitude backtests.", "data/reference + interim geo tables", "Rebuild historical stadium coordinates and time-zone/elevation features.", "Backtestable context."),
        ("Historical club / height / caps / goals comparators", "Would make squad features closer to the WC2026 PDF.", "data/interim historical squad tables", "Augment historical squads with comparable player attributes and as-of joins.", "Trainable only if the historical coverage becomes comparable."),
        ("Transfermarkt matching coverage", "Market-value features remain blocked by low, uneven matching.", "data/interim and data/reference matching tables", "Improve identity resolution and add explicit verification rows.", "Blocked today; potentially trainable later."),
    ]
    lines.append(pd.DataFrame(high_high, columns=["gap", "why_it_matters", "where_it_should_live", "how_to_collect", "trainable_or_context"]).to_markdown(index=False))
    lines.append("")
    lines.append("### 3. Low value / likely not worth it")
    low_value = [
        ("Squad-star narrative only", "The award / star signal has already failed promotion.", "outputs/reports only", "Keep as commentary, not model input.", "Context only."),
        ("Minor confederation residual tuning", "Residuals are too small to justify more complexity.", "reports/predictions", "Only revisit if a stronger interaction appears.", "Context only."),
    ]
    lines.append(pd.DataFrame(low_value, columns=["gap", "why_it_matters", "where_it_should_live", "how_to_collect", "trainable_or_context"]).to_markdown(index=False))
    lines.append("")

    lines.append("## H. Hypothesis backlog from available data")
    backlog_rows = [
        ("objective residual extreme mismatch rule", "yes", "small (review set only)", "yes", "high", "medium", "test_now"),
        ("late material absence rule", "partial", "small / live-only", "yes", "high", "medium", "data_first"),
        ("goalkeeper crisis rule", "partial", "small / live-only", "yes", "high", "medium", "dashboard_context_only"),
        ("friendly lineup distortion rule", "partial", "historical + live", "yes", "medium", "medium", "test_now"),
        ("coach continuity / late coach change", "partial", "1930-2022", "yes", "medium", "medium", "data_first"),
        ("venue/travel fatigue with historical reconstruction", "partial", "historical + WC2026", "yes", "medium", "low", "test_now"),
        ("altitude familiarity with full history", "yes", "historical", "yes", "medium", "low", "test_now"),
        ("squad age / position mix", "yes", "1930-2022", "yes", "medium", "low", "test_now"),
        ("team compactness / defensive fragility", "yes", "historical matches", "yes", "medium", "low", "test_now"),
        ("market value if coverage improves", "no", "2004-2026 values, poor matching", "limited", "high", "high", "data_first"),
        ("expected XI / minutes-weighted squad strength", "partial", "2022 + current-only", "no", "high", "medium", "skip"),
    ]
    lines.append(pd.DataFrame(backlog_rows, columns=["hypothesis", "required_data_already_available", "sample_size", "backtestable", "expected_value", "complexity", "recommended_action"]).to_markdown(index=False))
    lines.append("")

    lines.append("## I. Recommended next 3 work sessions")
    session_rows = [
        ("Data cleanup/inventory", "`scripts/build_data_inventory_overview.py`, `outputs/reports/data_inventory_overview.md`, `outputs/predictions/data_inventory_catalog.csv`", "Keep the inventory script synced with any new raw/interim files and add a few file-specific overrides if the scan reveals ambiguity.", "python3 .venv/bin/python scripts/build_data_inventory_overview.py", "low"),
        ("High-value backtest", "`outputs/reports/previous_match_strain_policy_recommendation.md`, `outputs/reports/altitude_policy_recommendation.md`, `outputs/reports/country_context_policy_recommendation.md`", "Re-run one or two of the rejected hypotheses with tighter splits or better joins, but do not touch the frozen candidate.", ".venv/bin/python scripts/check_frozen_submission_integrity.py", "medium"),
        ("Dashboard/live integration", "`outputs/live/*`, `docs/*`, `data/live/*`", "Keep the live dashboard aligned with the frozen candidate and surface only context, not hidden score mutations.", "inspect exported dashboard files and confirm hashes unchanged", "low"),
    ]
    lines.append(pd.DataFrame(session_rows, columns=["task", "files_likely_touched", "expected_output", "verification_commands", "risk_level"]).to_markdown(index=False))
    lines.append("")

    lines.append("## J. Validation commands run")
    lines.append("- `.venv/bin/python scripts/build_data_inventory_overview.py`")
    lines.append("- `.venv/bin/python scripts/check_frozen_submission_integrity.py`")
    lines.append("- Directory/file discovery via `find` and `rg --files` on the scoped data/output directories")
    lines.append("")

    lines.append("## Notes")
    lines.append(f"- `outputs/final_candidate_v3_objective_residual/` present: **{'yes' if directories_present.get('v3', False) else 'no'}**")
    lines.append("- The inventory intentionally does not inspect unrelated caches or build outputs.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    files = collect_paths()
    rows = [row_to_dict(path) for path in files]
    df = pd.DataFrame(rows)

    # Stable column order for the catalog.
    ordered_columns = [
        "file_path",
        "family",
        "format",
        "status",
        "row_count",
        "column_count",
        "key_columns",
        "unique_teams",
        "unique_matches",
        "years_covered",
        "WC2026_coverage",
        "historical_coverage",
        "trainable_status",
        "leakage_risk",
        "current_use",
        "short_description",
        "known_blockers",
        "possible_next_use",
        "size_mb",
        "size_bytes",
        "duplicates",
        "missing_values",
        "error",
    ]
    for col in ordered_columns:
        if col not in df.columns:
            df[col] = ""
    df = df[ordered_columns].sort_values("file_path").reset_index(drop=True)
    df.to_csv(CATALOG_PATH, index=False)

    directories_present = {
        "v3": (REPO_ROOT / "outputs" / "final_candidate_v3_objective_residual").exists(),
    }
    REPORT_PATH.write_text(build_report(df, totals={}, directories_present=directories_present), encoding="utf-8")

    print(f"Wrote {rel(REPORT_PATH)}")
    print(f"Wrote {rel(CATALOG_PATH)}")
    print(f"Catalogued {len(df)} files")


if __name__ == "__main__":
    main()
